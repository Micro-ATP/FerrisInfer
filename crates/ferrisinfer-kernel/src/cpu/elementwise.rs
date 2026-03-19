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
    let thread_count = preferred_elementwise_thread_count(out_values.len());

    if thread_count == 1 {
        for ((slot, lhs_value), rhs_value) in out_values
            .iter_mut()
            .zip(lhs_values.iter().copied())
            .zip(rhs_values.iter().copied())
        {
            *slot = lhs_value + rhs_value;
        }
    } else {
        let chunk_len = out_values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_len).enumerate() {
                let start = chunk_index * chunk_len;
                let end = start + out_chunk.len();
                let lhs_chunk = &lhs_values[start..end];
                let rhs_chunk = &rhs_values[start..end];
                scope.spawn(move || {
                    for ((slot, lhs_value), rhs_value) in out_chunk
                        .iter_mut()
                        .zip(lhs_chunk.iter().copied())
                        .zip(rhs_chunk.iter().copied())
                    {
                        *slot = lhs_value + rhs_value;
                    }
                });
            }
        });
    }

    Ok(())
}

pub fn add_f32_in_place(out: &mut Tensor, rhs: &Tensor) -> Result<()> {
    validate_binary_in_place(out, rhs)?;

    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let thread_count = preferred_elementwise_thread_count(out_values.len());

    if thread_count == 1 {
        for (slot, rhs_value) in out_values.iter_mut().zip(rhs_values.iter().copied()) {
            *slot += rhs_value;
        }
    } else {
        let chunk_len = out_values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_len).enumerate() {
                let start = chunk_index * chunk_len;
                let end = start + out_chunk.len();
                let rhs_chunk = &rhs_values[start..end];
                scope.spawn(move || {
                    for (slot, rhs_value) in out_chunk.iter_mut().zip(rhs_chunk.iter().copied()) {
                        *slot += rhs_value;
                    }
                });
            }
        });
    }

    Ok(())
}

pub fn mul_f32(lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_same_shape(lhs, rhs, out)?;

    let lhs_values = lhs.as_f32_slice()?;
    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let thread_count = preferred_elementwise_thread_count(out_values.len());

    if thread_count == 1 {
        for ((slot, lhs_value), rhs_value) in out_values
            .iter_mut()
            .zip(lhs_values.iter().copied())
            .zip(rhs_values.iter().copied())
        {
            *slot = lhs_value * rhs_value;
        }
    } else {
        let chunk_len = out_values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_len).enumerate() {
                let start = chunk_index * chunk_len;
                let end = start + out_chunk.len();
                let lhs_chunk = &lhs_values[start..end];
                let rhs_chunk = &rhs_values[start..end];
                scope.spawn(move || {
                    for ((slot, lhs_value), rhs_value) in out_chunk
                        .iter_mut()
                        .zip(lhs_chunk.iter().copied())
                        .zip(rhs_chunk.iter().copied())
                    {
                        *slot = lhs_value * rhs_value;
                    }
                });
            }
        });
    }

    Ok(())
}

pub fn mul_f32_in_place(out: &mut Tensor, rhs: &Tensor) -> Result<()> {
    validate_binary_in_place(out, rhs)?;

    let rhs_values = rhs.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let thread_count = preferred_elementwise_thread_count(out_values.len());

    if thread_count == 1 {
        for (slot, rhs_value) in out_values.iter_mut().zip(rhs_values.iter().copied()) {
            *slot *= rhs_value;
        }
    } else {
        let chunk_len = out_values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_len).enumerate() {
                let start = chunk_index * chunk_len;
                let end = start + out_chunk.len();
                let rhs_chunk = &rhs_values[start..end];
                scope.spawn(move || {
                    for (slot, rhs_value) in out_chunk.iter_mut().zip(rhs_chunk.iter().copied()) {
                        *slot *= rhs_value;
                    }
                });
            }
        });
    }

    Ok(())
}

pub fn silu_f32(input: &Tensor, out: &mut Tensor) -> Result<()> {
    validate_unary_f32(input, out)?;

    let input_values = input.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let thread_count = preferred_elementwise_thread_count(out_values.len());

    if thread_count == 1 {
        for (slot, value) in out_values.iter_mut().zip(input_values.iter().copied()) {
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            *slot = value * sigmoid;
        }
    } else {
        let chunk_len = out_values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_len).enumerate() {
                let start = chunk_index * chunk_len;
                let end = start + out_chunk.len();
                let input_chunk = &input_values[start..end];
                scope.spawn(move || {
                    for (slot, value) in out_chunk.iter_mut().zip(input_chunk.iter().copied()) {
                        let sigmoid = 1.0 / (1.0 + (-value).exp());
                        *slot = value * sigmoid;
                    }
                });
            }
        });
    }

    Ok(())
}

pub fn silu_f32_in_place(tensor: &mut Tensor) -> Result<()> {
    tensor.ensure_dtype(DType::F32)?;
    tensor.ensure_contiguous()?;

    let values = tensor.as_f32_slice_mut()?;
    let thread_count = preferred_elementwise_thread_count(values.len());

    if thread_count == 1 {
        for value in values {
            let sigmoid = 1.0 / (1.0 + (-*value).exp());
            *value *= sigmoid;
        }
    } else {
        let chunk_len = values.len().div_ceil(thread_count);
        std::thread::scope(|scope| {
            for chunk in values.chunks_mut(chunk_len) {
                scope.spawn(move || {
                    for value in chunk {
                        let sigmoid = 1.0 / (1.0 + (-*value).exp());
                        *value *= sigmoid;
                    }
                });
            }
        });
    }

    Ok(())
}

fn preferred_elementwise_thread_count(element_count: usize) -> usize {
    const MIN_PARALLEL_ELEMENTS: usize = 131_072;

    if element_count < MIN_PARALLEL_ELEMENTS {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
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

fn validate_binary_in_place(out: &Tensor, rhs: &Tensor) -> Result<()> {
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;
    rhs.ensure_dtype(DType::F32)?;
    rhs.ensure_contiguous()?;

    if out.shape() != rhs.shape() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "in-place elementwise operands must have the same shape",
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
    fn add_f32_in_place_accumulates_rhs_into_out() {
        let shape = Shape::from_slice(&[3]).unwrap();
        let rhs = Tensor::from_f32_vec(shape.clone(), vec![0.5, -1.0, 2.0]).unwrap();
        let mut out = Tensor::from_f32_vec(shape, vec![1.0, 2.0, 3.0]).unwrap();

        add_f32_in_place(&mut out, &rhs).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![1.5, 1.0, 5.0]);
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
    fn mul_f32_in_place_multiplies_rhs_into_out() {
        let shape = Shape::from_slice(&[3]).unwrap();
        let rhs = Tensor::from_f32_vec(shape.clone(), vec![0.5, 2.0, -1.5]).unwrap();
        let mut out = Tensor::from_f32_vec(shape, vec![2.0, -3.0, 4.0]).unwrap();

        mul_f32_in_place(&mut out, &rhs).unwrap();

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

    #[test]
    fn silu_f32_in_place_updates_tensor_values() {
        let shape = Shape::from_slice(&[3]).unwrap();
        let mut tensor = Tensor::from_f32_vec(shape, vec![-1.0, 0.0, 1.0]).unwrap();

        silu_f32_in_place(&mut tensor).unwrap();

        approx_eq_slice(
            &tensor.to_vec_f32().unwrap(),
            &[-0.26894143, 0.0, 0.7310586],
            1e-6,
        );
    }
}
