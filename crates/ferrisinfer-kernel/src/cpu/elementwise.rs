use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

pub fn zero_tensor(tensor: &mut Tensor) -> Result<()> {
    tensor.as_bytes_mut()?.fill(0);
    Ok(())
}

pub fn add_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_same_shape(lhs, rhs, out)?;

    let lhs_values = lhs.as_f32_slice()?;
    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;

    for ((slot, lhs_value), rhs_value) in out_values
        .iter_mut()
        .zip(lhs_values.iter().copied())
        .zip(rhs_values.iter().copied())
    {
        *slot = lhs_value + rhs_value;
    }

    Ok(())
}

pub fn mul_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_same_shape(lhs, rhs, out)?;

    let lhs_values = lhs.as_f32_slice()?;
    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;

    for ((slot, lhs_value), rhs_value) in out_values
        .iter_mut()
        .zip(lhs_values.iter().copied())
        .zip(rhs_values.iter().copied())
    {
        *slot = lhs_value * rhs_value;
    }

    Ok(())
}

pub fn silu_f32(input: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_unary_f32(input, out)?;

    let input_values = input.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;

    for (slot, value) in out_values.iter_mut().zip(input_values.iter().copied()) {
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        *slot = value * sigmoid;
    }

    Ok(())
}

fn validate_same_shape(lhs: &Tensor, rhs: &Tensor, out: &Tensor) -> Result<()> {
    validate_unary_f32(lhs, out)?;
    rhs.ensure_dtype(DType::F32)?;
    rhs.ensure_contiguous()?;

    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "elementwise operands and output must have the same shape",
        ));
    }

    Ok(())
}

fn validate_unary_f32(input: &Tensor, out: &Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

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
    fn add_f32_accumulates_elementwise() {
        let shape = Shape::from_slice(&[4]).unwrap();
        let lhs = Tensor::from_f32_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor::from_f32_vec(shape.clone(), vec![0.5, -1.0, 2.0, 1.5]).unwrap();
        let mut out = Tensor::zeros(DType::F32, shape).unwrap();

        add_f32(&lhs, &rhs, &mut out).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![1.5, 1.0, 5.0, 5.5]);
    }

    #[test]
    fn mul_f32_multiplies_elementwise() {
        let shape = Shape::from_slice(&[3]).unwrap();
        let lhs = Tensor::from_f32_vec(shape.clone(), vec![2.0, -3.0, 4.0]).unwrap();
        let rhs = Tensor::from_f32_vec(shape.clone(), vec![0.5, 2.0, -1.5]).unwrap();
        let mut out = Tensor::zeros(DType::F32, shape).unwrap();

        mul_f32(&lhs, &rhs, &mut out).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![1.0, -6.0, -6.0]);
    }

    #[test]
    fn silu_f32_matches_reference_values() {
        let shape = Shape::from_slice(&[3]).unwrap();
        let input = Tensor::from_f32_vec(shape.clone(), vec![-1.0, 0.0, 1.0]).unwrap();
        let mut out = Tensor::zeros(DType::F32, shape).unwrap();

        silu_f32(&input, &mut out).unwrap();

        approx_eq_slice(
            &out.to_vec_f32().unwrap(),
            &[-0.26894143, 0.0, 0.7310586],
            1e-6,
        );
    }
}
