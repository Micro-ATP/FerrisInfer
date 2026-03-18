use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

pub fn matmul_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    lhs.ensure_dtype(DType::F32)?;
    lhs.ensure_contiguous()?;
    rhs.ensure_dtype(DType::F32)?;
    rhs.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if lhs.shape().rank() != 2 || rhs.shape().rank() != 2 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul_f32 currently requires rank-2 tensors",
        ));
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let out_dims = out.shape().dims();

    let m = lhs_dims[0];
    let k = lhs_dims[1];
    let rhs_k = rhs_dims[0];
    let n = rhs_dims[1];

    if rhs_k != k {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul inner dimensions must match",
        ));
    }

    if out_dims != [m, n] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul output shape must be [lhs.rows, rhs.cols]",
        ));
    }

    let lhs_values = lhs.to_vec_f32()?;
    let rhs_values = rhs.to_vec_f32()?;
    let mut out_values = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                let lhs_index = row * k + inner;
                let rhs_index = inner * n + col;
                sum += lhs_values[lhs_index] * rhs_values[rhs_index];
            }
            out_values[row * n + col] = sum;
        }
    }

    out.copy_from_f32_slice(&out_values)
}

#[cfg(test)]
mod tests {
    use ferrisinfer_core::Shape;

    use super::*;

    #[test]
    fn matmul_f32_multiplies_two_matrices() {
        let lhs = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let rhs = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 2]).unwrap(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 2]).unwrap()).unwrap();

        matmul_f32(&lhs, &rhs, &mut out).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_f32_rejects_invalid_output_shape() {
        let lhs =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![1.0, 2.0]).unwrap();
        let rhs =
            Tensor::from_f32_vec(Shape::from_slice(&[2, 1]).unwrap(), vec![3.0, 4.0]).unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 1]).unwrap()).unwrap();

        let error = matmul_f32(&lhs, &rhs, &mut out).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidShape);
    }
}
