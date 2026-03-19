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

    let lhs_values = lhs.as_f32_slice()?;
    let rhs_values = rhs.as_f32_slice()?;
    let mut out_values = vec![0.0f32; m * n];
    let thread_count = preferred_thread_count(m, n, k);

    if thread_count == 1 {
        compute_row_chunk(&lhs_values, &rhs_values, &mut out_values, 0, m, k, n);
    } else {
        let rows_per_chunk = m.div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(rows_per_chunk * n).enumerate() {
                let row_start = chunk_index * rows_per_chunk;
                let row_end = (row_start + rows_per_chunk).min(m);
                let lhs_values = &lhs_values;
                let rhs_values = &rhs_values;

                scope.spawn(move || {
                    compute_row_chunk(lhs_values, rhs_values, out_chunk, row_start, row_end, k, n);
                });
            }
        });
    }

    out.copy_from_f32_slice(&out_values)
}

fn compute_row_chunk(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_chunk: &mut [f32],
    row_start: usize,
    row_end: usize,
    k: usize,
    n: usize,
) {
    for (local_row, row) in (row_start..row_end).enumerate() {
        let lhs_row = &lhs_values[row * k..(row + 1) * k];
        let out_row = &mut out_chunk[local_row * n..(local_row + 1) * n];

        for (inner, lhs_value) in lhs_row.iter().copied().enumerate() {
            if lhs_value == 0.0 {
                continue;
            }

            let rhs_row = &rhs_values[inner * n..(inner + 1) * n];
            for (slot, rhs_value) in out_row.iter_mut().zip(rhs_row.iter().copied()) {
                *slot += lhs_value * rhs_value;
            }
        }
    }
}

fn preferred_thread_count(rows: usize, cols: usize, inner: usize) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    if rows < 2 {
        return 1;
    }

    let total_work = rows.saturating_mul(cols).saturating_mul(inner);
    if total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(rows))
        .unwrap_or(1)
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
