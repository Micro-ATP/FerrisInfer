use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

pub fn zero_tensor(tensor: &mut Tensor) -> Result<()> {
    tensor.as_bytes_mut()?.fill(0);
    Ok(())
}

pub fn add_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_same_shape(lhs, rhs, out)?;

    let lhs_values = lhs.to_vec_f32()?;
    let rhs_values = rhs.to_vec_f32()?;
    let mut out_values = Vec::with_capacity(out.element_count());

    for (lhs_value, rhs_value) in lhs_values.into_iter().zip(rhs_values) {
        out_values.push(lhs_value + rhs_value);
    }

    out.copy_from_f32_slice(&out_values)
}

pub fn mul_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_same_shape(lhs, rhs, out)?;

    let lhs_values = lhs.to_vec_f32()?;
    let rhs_values = rhs.to_vec_f32()?;
    let mut out_values = Vec::with_capacity(out.element_count());

    for (lhs_value, rhs_value) in lhs_values.into_iter().zip(rhs_values) {
        out_values.push(lhs_value * rhs_value);
    }

    out.copy_from_f32_slice(&out_values)
}

pub fn silu_f32(input: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_unary_f32(input, out)?;

    let input_values = input.to_vec_f32()?;
    let mut out_values = Vec::with_capacity(out.element_count());

    for value in input_values {
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        out_values.push(value * sigmoid);
    }

    out.copy_from_f32_slice(&out_values)
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
