use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    fn cblas_sgemv(
        order: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

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
    let out_values = out.as_f32_slice_mut()?;

    if try_accelerate_matmul(lhs_values, rhs_values, out_values, m, k, n) {
        return Ok(());
    }

    if m == 1 {
        compute_single_row_matmul(lhs_values, rhs_values, out_values, n);
        return Ok(());
    }

    out_values.fill(0.0);
    let thread_count = preferred_thread_count(m, n, k);

    if thread_count == 1 {
        compute_row_chunk(&lhs_values, &rhs_values, out_values, 0, m, k, n);
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
    Ok(())
}

pub fn matmul_rhs_transposed_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    lhs.ensure_dtype(DType::F32)?;
    lhs.ensure_contiguous()?;
    rhs.ensure_dtype(DType::F32)?;
    rhs.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if lhs.shape().rank() != 2 || rhs.shape().rank() != 2 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul_rhs_transposed_f32 currently requires rank-2 tensors",
        ));
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let out_dims = out.shape().dims();

    let m = lhs_dims[0];
    let k = lhs_dims[1];
    let n = rhs_dims[0];
    let rhs_k = rhs_dims[1];

    if rhs_k != k {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "transposed matmul inner dimensions must match",
        ));
    }

    if out_dims != [m, n] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "transposed matmul output shape must be [lhs.rows, rhs.rows]",
        ));
    }

    let lhs_values = lhs.as_f32_slice()?;
    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;

    if try_accelerate_matmul_rhs_transposed(lhs_values, rhs_values, out_values, m, k, n) {
        return Ok(());
    }

    let thread_count = preferred_transposed_thread_count(m, n, k);
    if thread_count == 1 {
        compute_transposed_row_chunk(lhs_values, rhs_values, out_values, k, n, 0, m);
    } else {
        let rows_per_chunk = m.div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(rows_per_chunk * n).enumerate() {
                let row_start = chunk_index * rows_per_chunk;
                let row_end = (row_start + rows_per_chunk).min(m);
                let lhs_values = &lhs_values;
                let rhs_values = &rhs_values;

                scope.spawn(move || {
                    compute_transposed_row_chunk(
                        lhs_values, rhs_values, out_chunk, k, n, row_start, row_end,
                    );
                });
            }
        });
    }

    Ok(())
}

pub fn matmul_rhs_transposed_argmax_f32(lhs: &Tensor, rhs: &Tensor) -> Result<(usize, f32)> {
    lhs.ensure_dtype(DType::F32)?;
    lhs.ensure_contiguous()?;
    rhs.ensure_dtype(DType::F32)?;
    rhs.ensure_contiguous()?;

    if lhs.shape().rank() != 2 || rhs.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul_rhs_transposed_argmax_f32 requires rank-2 tensors",
        ));
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    if lhs_dims[0] != 1 || rhs_dims[1] != lhs_dims[1] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul_rhs_transposed_argmax_f32 expects lhs [1, k] and rhs [n, k]",
        ));
    }

    let input_width = lhs_dims[1];
    let out_width = rhs_dims[0];
    if out_width == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "matmul_rhs_transposed_argmax_f32 requires a non-empty output width",
        ));
    }

    let input_values = lhs.as_f32_slice()?;
    let weight_values = rhs.as_f32_slice()?;

    let aggregate = if let Some(aggregate) = try_accelerate_matmul_rhs_transposed_argmax(
        input_values,
        weight_values,
        input_width,
        out_width,
    ) {
        aggregate
    } else {
        let thread_count = preferred_projection_thread_count(out_width, input_width);
        if thread_count == 1 {
            compute_projection_argmax_chunk(input_values, weight_values, input_width, 0, out_width)
        } else {
            let cols_per_chunk = out_width.div_ceil(thread_count);
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for chunk_index in 0..thread_count {
                    let col_start = chunk_index * cols_per_chunk;
                    if col_start >= out_width {
                        break;
                    }

                    let col_end = (col_start + cols_per_chunk).min(out_width);
                    let input_values = &input_values;
                    let weight_values = &weight_values;
                    handles.push(scope.spawn(move || {
                        compute_projection_argmax_chunk(
                            input_values,
                            weight_values,
                            input_width,
                            col_start,
                            col_end,
                        )
                    }));
                }

                let mut aggregate = None;
                for handle in handles {
                    let chunk = handle.join().expect("scoped thread panicked");
                    aggregate = Some(match aggregate {
                        Some(current) => merge_projection_argmax_chunks(current, chunk),
                        None => chunk,
                    });
                }

                aggregate.expect("at least one projection argmax chunk")
            })
        }
    };

    Ok((aggregate.best_index, 1.0 / aggregate.scaled_sum))
}

#[cfg(target_os = "macos")]
fn try_accelerate_matmul(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_values: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Ok(m) = i32::try_from(m) else {
        return false;
    };
    let Ok(k) = i32::try_from(k) else {
        return false;
    };
    let Ok(n) = i32::try_from(n) else {
        return false;
    };

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m,
            n,
            k,
            1.0,
            lhs_values.as_ptr(),
            k,
            rhs_values.as_ptr(),
            n,
            0.0,
            out_values.as_mut_ptr(),
            n,
        );
    }
    true
}

#[cfg(not(target_os = "macos"))]
fn try_accelerate_matmul(
    _lhs_values: &[f32],
    _rhs_values: &[f32],
    _out_values: &mut [f32],
    _m: usize,
    _k: usize,
    _n: usize,
) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn try_accelerate_matmul_rhs_transposed(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_values: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Ok(m) = i32::try_from(m) else {
        return false;
    };
    let Ok(k) = i32::try_from(k) else {
        return false;
    };
    let Ok(n) = i32::try_from(n) else {
        return false;
    };

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m,
            n,
            k,
            1.0,
            lhs_values.as_ptr(),
            k,
            rhs_values.as_ptr(),
            k,
            0.0,
            out_values.as_mut_ptr(),
            n,
        );
    }
    true
}

#[cfg(not(target_os = "macos"))]
fn try_accelerate_matmul_rhs_transposed(
    _lhs_values: &[f32],
    _rhs_values: &[f32],
    _out_values: &mut [f32],
    _m: usize,
    _k: usize,
    _n: usize,
) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn try_accelerate_matmul_rhs_transposed_argmax(
    lhs_values: &[f32],
    rhs_values: &[f32],
    input_width: usize,
    out_width: usize,
) -> Option<ProjectionArgmax> {
    let Ok(input_width_i32) = i32::try_from(input_width) else {
        return None;
    };
    let Ok(out_width_i32) = i32::try_from(out_width) else {
        return None;
    };

    let mut logits = vec![0.0f32; out_width];
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            out_width_i32,
            input_width_i32,
            1.0,
            rhs_values.as_ptr(),
            input_width_i32,
            lhs_values.as_ptr(),
            1,
            0.0,
            logits.as_mut_ptr(),
            1,
        );
    }
    Some(projection_argmax_from_logits(&logits))
}

#[cfg(not(target_os = "macos"))]
fn try_accelerate_matmul_rhs_transposed_argmax(
    _lhs_values: &[f32],
    _rhs_values: &[f32],
    _input_width: usize,
    _out_width: usize,
) -> Option<ProjectionArgmax> {
    None
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

fn compute_single_row_matmul(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_values: &mut [f32],
    cols: usize,
) {
    let thread_count = preferred_single_row_thread_count(cols, lhs_values.len());
    if thread_count == 1 {
        compute_single_row_chunk(lhs_values, rhs_values, out_values, cols, 0, cols);
        return;
    }

    let cols_per_chunk = cols.div_ceil(thread_count);
    std::thread::scope(|scope| {
        for (chunk_index, out_chunk) in out_values.chunks_mut(cols_per_chunk).enumerate() {
            let col_start = chunk_index * cols_per_chunk;
            let col_end = (col_start + cols_per_chunk).min(cols);
            let lhs_values = &lhs_values;
            let rhs_values = &rhs_values;

            scope.spawn(move || {
                compute_single_row_chunk(
                    lhs_values, rhs_values, out_chunk, cols, col_start, col_end,
                );
            });
        }
    });
}

fn compute_single_row_chunk(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_chunk: &mut [f32],
    total_cols: usize,
    col_start: usize,
    col_end: usize,
) {
    out_chunk.fill(0.0);

    for (inner, lhs_value) in lhs_values.iter().copied().enumerate() {
        if lhs_value == 0.0 {
            continue;
        }

        let rhs_row = &rhs_values[inner * total_cols + col_start..inner * total_cols + col_end];
        for (slot, rhs_value) in out_chunk.iter_mut().zip(rhs_row.iter().copied()) {
            *slot += lhs_value * rhs_value;
        }
    }
}

fn compute_transposed_row_chunk(
    lhs_values: &[f32],
    rhs_values: &[f32],
    out_chunk: &mut [f32],
    input_width: usize,
    out_width: usize,
    row_start: usize,
    row_end: usize,
) {
    for (local_row, row) in (row_start..row_end).enumerate() {
        let input_row = &lhs_values[row * input_width..(row + 1) * input_width];
        let out_row = &mut out_chunk[local_row * out_width..(local_row + 1) * out_width];

        for out_col in 0..out_width {
            let mut sum = 0.0f32;
            let rhs_row = &rhs_values[out_col * input_width..(out_col + 1) * input_width];
            for (input_value, rhs_value) in input_row.iter().zip(rhs_row.iter()) {
                sum += input_value * rhs_value;
            }
            out_row[out_col] = sum;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProjectionArgmax {
    best_index: usize,
    max_logit: f32,
    scaled_sum: f32,
}

fn compute_projection_argmax_chunk(
    input_values: &[f32],
    weight_values: &[f32],
    input_width: usize,
    col_start: usize,
    col_end: usize,
) -> ProjectionArgmax {
    let mut best_index = col_start;
    let mut max_logit = f32::NEG_INFINITY;
    let mut scaled_sum = 0.0f32;

    for out_col in col_start..col_end {
        let mut sum = 0.0f32;
        let weight_row = &weight_values[out_col * input_width..(out_col + 1) * input_width];
        for (input_value, weight_value) in input_values.iter().zip(weight_row.iter()) {
            sum += input_value * weight_value;
        }

        if sum > max_logit {
            scaled_sum = if max_logit.is_finite() {
                scaled_sum * (max_logit - sum).exp() + 1.0
            } else {
                1.0
            };
            max_logit = sum;
            best_index = out_col;
        } else {
            scaled_sum += (sum - max_logit).exp();
        }
    }

    ProjectionArgmax {
        best_index,
        max_logit,
        scaled_sum,
    }
}

fn merge_projection_argmax_chunks(
    left: ProjectionArgmax,
    right: ProjectionArgmax,
) -> ProjectionArgmax {
    if left.max_logit >= right.max_logit {
        ProjectionArgmax {
            best_index: left.best_index,
            max_logit: left.max_logit,
            scaled_sum: left.scaled_sum
                + right.scaled_sum * (right.max_logit - left.max_logit).exp(),
        }
    } else {
        ProjectionArgmax {
            best_index: right.best_index,
            max_logit: right.max_logit,
            scaled_sum: right.scaled_sum
                + left.scaled_sum * (left.max_logit - right.max_logit).exp(),
        }
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn projection_argmax_from_logits(logits: &[f32]) -> ProjectionArgmax {
    debug_assert!(!logits.is_empty());

    let mut best_index = 0usize;
    let mut max_logit = f32::NEG_INFINITY;
    let mut scaled_sum = 0.0f32;

    for (index, logit) in logits.iter().copied().enumerate() {
        if logit > max_logit {
            scaled_sum = if max_logit.is_finite() {
                scaled_sum * (max_logit - logit).exp() + 1.0
            } else {
                1.0
            };
            max_logit = logit;
            best_index = index;
        } else {
            scaled_sum += (logit - max_logit).exp();
        }
    }

    ProjectionArgmax {
        best_index,
        max_logit,
        scaled_sum,
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

fn preferred_single_row_thread_count(cols: usize, inner: usize) -> usize {
    const MIN_PARALLEL_WORK: usize = 8_388_608;
    const MIN_PARALLEL_COLS: usize = 8192;

    if cols < MIN_PARALLEL_COLS {
        return 1;
    }

    let total_work = cols.saturating_mul(inner);
    if total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(cols))
        .unwrap_or(1)
}

fn preferred_transposed_thread_count(rows: usize, cols: usize, inner: usize) -> usize {
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

fn preferred_projection_thread_count(out_width: usize, input_width: usize) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    if out_width < 2 {
        return 1;
    }

    let total_work = out_width.saturating_mul(input_width);
    if total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(out_width))
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

    #[test]
    fn matmul_rhs_transposed_f32_multiplies_weight_rows_against_input() {
        let lhs = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let rhs = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 3]).unwrap(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 2]).unwrap()).unwrap();

        matmul_rhs_transposed_f32(&lhs, &rhs, &mut out).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![50.0, 68.0, 122.0, 167.0]);
    }

    #[test]
    fn matmul_rhs_transposed_argmax_f32_returns_best_index_and_probability() {
        let lhs =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![1.0, 2.0]).unwrap();
        let rhs = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let (best_index, probability) = matmul_rhs_transposed_argmax_f32(&lhs, &rhs).unwrap();

        assert_eq!(best_index, 2);
        assert!((probability - 0.66524094).abs() <= 1e-6);
    }
}
