use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

pub fn rms_norm_f32(input: &Tensor, weight: &Tensor, out: &mut Tensor, eps: f32) -> Result<()> {
    validate_rowwise_f32(input, out)?;
    weight.ensure_dtype(DType::F32)?;
    weight.ensure_contiguous()?;

    let dims = input.shape().dims();
    let hidden = *dims.last().ok_or_else(|| {
        FerrisError::new(
            ErrorKind::InvalidShape,
            "rms_norm_f32 requires at least one dimension",
        )
    })?;

    if weight.shape().rank() != 1 || weight.shape().dims()[0] != hidden {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "rms_norm weight must be a rank-1 tensor matching the last input dimension",
        ));
    }

    let rows = input.element_count() / hidden;
    let input_values = input.to_vec_f32()?;
    let weight_values = weight.to_vec_f32()?;
    let mut out_values = vec![0.0f32; input.element_count()];

    for row in 0..rows {
        let start = row * hidden;
        let end = start + hidden;
        let row_slice = &input_values[start..end];

        let mean_square = row_slice.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
        let inv_rms = 1.0 / (mean_square + eps).sqrt();

        for col in 0..hidden {
            out_values[start + col] = row_slice[col] * inv_rms * weight_values[col];
        }
    }

    out.copy_from_f32_slice(&out_values)
}

pub fn softmax_f32(input: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_rowwise_f32(input, out)?;

    let dims = input.shape().dims();
    let width = *dims.last().ok_or_else(|| {
        FerrisError::new(
            ErrorKind::InvalidShape,
            "softmax_f32 requires at least one dimension",
        )
    })?;
    let rows = input.element_count() / width;
    let input_values = input.to_vec_f32()?;
    let mut out_values = vec![0.0f32; input.element_count()];

    for row in 0..rows {
        let start = row * width;
        let end = start + width;
        let row_slice = &input_values[start..end];
        let row_max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for col in 0..width {
            let value = (row_slice[col] - row_max).exp();
            out_values[start + col] = value;
            sum += value;
        }

        for col in 0..width {
            out_values[start + col] /= sum;
        }
    }

    out.copy_from_f32_slice(&out_values)
}

fn validate_rowwise_f32(input: &Tensor, out: &Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if input.shape().rank() == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "operation requires at least one tensor dimension",
        ));
    }

    if input.shape() != out.shape() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "input and output tensors must have the same shape",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ferrisinfer_core::Shape;

    use super::*;

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.iter().zip(rhs.iter()) {
            assert!(
                (left - right).abs() <= tolerance,
                "left={left}, right={right}"
            );
        }
    }

    #[test]
    fn rms_norm_f32_normalizes_last_dimension() {
        let input = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2]).unwrap(),
            vec![3.0, 4.0, 0.0, 5.0],
        )
        .unwrap();
        let weight =
            Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![1.0, 1.5]).unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 2]).unwrap()).unwrap();

        rms_norm_f32(&input, &weight, &mut out, 1e-5).unwrap();

        approx_eq_slice(
            &out.to_vec_f32().unwrap(),
            &[0.84852797, 1.697056, 0.0, 2.1213198],
            1e-5,
        );
    }

    #[test]
    fn softmax_f32_normalizes_each_row() {
        let input = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 3]).unwrap()).unwrap();

        softmax_f32(&input, &mut out).unwrap();

        approx_eq_slice(
            &out.to_vec_f32().unwrap(),
            &[
                0.09003057, 0.24472848, 0.66524094, 0.33333334, 0.33333334, 0.33333334,
            ],
            1e-6,
        );
    }
}
