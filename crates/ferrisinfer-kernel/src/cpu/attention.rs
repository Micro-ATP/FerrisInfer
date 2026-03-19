use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

pub fn embedding_gather_f32(embedding: &Tensor, token_ids: &[u32], out: &mut Tensor) -> Result<()> {
    embedding.ensure_dtype(DType::F32)?;
    embedding.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if embedding.shape().rank() != 2 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "embedding_gather_f32 requires rank-2 tensors",
        ));
    }

    let embedding_dims = embedding.shape().dims();
    let out_dims = out.shape().dims();
    let vocab_size = embedding_dims[0];
    let hidden_size = embedding_dims[1];

    if out_dims[0] != token_ids.len() || out_dims[1] != hidden_size {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "embedding output shape must be [token_ids.len(), hidden_size]",
        ));
    }

    let embedding_values = embedding.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;

    for (row, token_id) in token_ids.iter().copied().enumerate() {
        let token_index = token_id as usize;
        if token_index >= vocab_size {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!("token id {token_index} is out of vocabulary range {vocab_size}"),
            ));
        }

        let src_start = token_index * hidden_size;
        let src_end = src_start + hidden_size;
        let dst_start = row * hidden_size;
        let dst_end = dst_start + hidden_size;
        out_values[dst_start..dst_end].copy_from_slice(&embedding_values[src_start..src_end]);
    }

    Ok(())
}

pub fn split_heads_f32(input: &Tensor, num_heads: usize, out: &mut Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if input.shape().rank() != 2 || out.shape().rank() != 3 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "split_heads_f32 requires input rank 2 and output rank 3",
        ));
    }

    let input_dims = input.shape().dims();
    let seq_len = input_dims[0];
    let hidden_size = input_dims[1];

    if hidden_size % num_heads != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "hidden size must be divisible by number of heads",
        ));
    }

    let head_dim = hidden_size / num_heads;
    if out.shape().dims() != [seq_len, num_heads, head_dim] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "split head output shape must be [seq_len, num_heads, head_dim]",
        ));
    }

    out.copy_from_f32_slice(input.as_f32_slice()?)
}

pub fn merge_heads_f32(input: &Tensor, out: &mut Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if input.shape().rank() != 3 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "merge_heads_f32 requires input rank 3 and output rank 2",
        ));
    }

    let input_dims = input.shape().dims();
    let seq_len = input_dims[0];
    let num_heads = input_dims[1];
    let head_dim = input_dims[2];
    let hidden_size = num_heads * head_dim;

    if out.shape().dims() != [seq_len, hidden_size] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "merge head output shape must be [seq_len, hidden_size]",
        ));
    }

    out.copy_from_f32_slice(input.as_f32_slice()?)
}

pub fn rope_f32(
    query: &mut Tensor,
    key: &mut Tensor,
    position_offset: usize,
    rotary_dims: usize,
    theta: f32,
) -> Result<()> {
    validate_attention_tensor(query)?;
    validate_attention_tensor(key)?;

    let query_dims = query.shape().dims();
    let key_dims = key.shape().dims();
    let seq_len = query_dims[0];
    let head_dim = query_dims[2];

    if key_dims[0] != seq_len || key_dims[2] != head_dim {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "query and key tensors must share sequence length and head dimension for RoPE",
        ));
    }

    if rotary_dims == 0 {
        return Ok(());
    }

    if rotary_dims > head_dim || rotary_dims % 2 != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "rotary_dims must be an even value no larger than head_dim",
        ));
    }

    if theta <= 0.0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidConfig,
            "rope theta must be positive",
        ));
    }

    let mut query_values = query.as_f32_slice()?.to_vec();
    let mut key_values = key.as_f32_slice()?.to_vec();
    apply_rope_in_place(
        &mut query_values,
        query_dims[0],
        query_dims[1],
        head_dim,
        position_offset,
        rotary_dims,
        theta,
    );
    apply_rope_in_place(
        &mut key_values,
        key_dims[0],
        key_dims[1],
        head_dim,
        position_offset,
        rotary_dims,
        theta,
    );

    query.copy_from_f32_slice(&query_values)?;
    key.copy_from_f32_slice(&key_values)
}

pub fn prefixed_causal_attention_f32(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cache_len: usize,
    out: &mut Tensor,
) -> Result<()> {
    validate_attention_tensor(query)?;
    validate_attention_tensor(key)?;
    validate_attention_tensor(value)?;
    validate_attention_tensor(out)?;

    let query_dims = query.shape().dims();
    let key_dims = key.shape().dims();
    let value_dims = value.shape().dims();
    let out_dims = out.shape().dims();

    let query_len = query_dims[0];
    let num_query_heads = query_dims[1];
    let head_dim = query_dims[2];
    let num_kv_heads = key_dims[1];
    let expected_kv_len = cache_len
        .checked_add(query_len)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "attention length overflow"))?;

    if out_dims != query_dims {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention output tensor must match query tensor shape",
        ));
    }

    if key_dims[0] != expected_kv_len
        || value_dims[0] != expected_kv_len
        || value_dims[1] != num_kv_heads
        || key_dims[2] != head_dim
        || value_dims[2] != head_dim
    {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention key/value tensors must match cache + query length and head dimension",
        ));
    }

    if num_query_heads % num_kv_heads != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "number of query heads must be divisible by number of KV heads",
        ));
    }

    let queries_per_kv_head = num_query_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let query_values = query.as_f32_slice()?;
    let key_values = key.as_f32_slice()?;
    let value_values = value.as_f32_slice()?;
    let mut out_values = vec![0.0f32; out.element_count()];

    let thread_count =
        preferred_attention_thread_count(query_len, num_query_heads, head_dim, expected_kv_len);
    if thread_count == 1 {
        compute_attention_chunk(
            &query_values,
            &key_values,
            &value_values,
            &mut out_values,
            0,
            query_len,
            cache_len,
            num_query_heads,
            num_kv_heads,
            queries_per_kv_head,
            head_dim,
            scale,
        );
    } else {
        let positions_per_chunk = query_len.div_ceil(thread_count);
        let chunk_width = positions_per_chunk * num_query_heads * head_dim;

        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_width).enumerate() {
                let position_start = chunk_index * positions_per_chunk;
                let position_end = (position_start + positions_per_chunk).min(query_len);
                let query_values = &query_values;
                let key_values = &key_values;
                let value_values = &value_values;

                scope.spawn(move || {
                    compute_attention_chunk(
                        query_values,
                        key_values,
                        value_values,
                        out_chunk,
                        position_start,
                        position_end,
                        cache_len,
                        num_query_heads,
                        num_kv_heads,
                        queries_per_kv_head,
                        head_dim,
                        scale,
                    );
                });
            }
        });
    }

    out.copy_from_f32_slice(&out_values)
}

pub fn decode_causal_attention_f32(
    query: &Tensor,
    cached_key: &Tensor,
    cached_value: &Tensor,
    cache_len: usize,
    key_slot: &Tensor,
    value_slot: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    validate_attention_tensor(query)?;
    validate_attention_tensor(cached_key)?;
    validate_attention_tensor(cached_value)?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;
    key_slot.ensure_dtype(DType::F32)?;
    key_slot.ensure_contiguous()?;
    value_slot.ensure_dtype(DType::F32)?;
    value_slot.ensure_contiguous()?;

    let query_dims = query.shape().dims();
    let cache_key_dims = cached_key.shape().dims();
    let cache_value_dims = cached_value.shape().dims();
    let out_dims = out.shape().dims();
    let num_query_heads = query_dims[1];
    let head_dim = query_dims[2];
    let num_kv_heads = cache_key_dims[1];

    if query_dims[0] != 1 || out_dims != query_dims {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decode attention requires query/output shape [1, num_heads, head_dim]",
        ));
    }

    if cache_key_dims[0] < cache_len
        || cache_value_dims[0] < cache_len
        || cache_value_dims[1] != num_kv_heads
        || cache_key_dims[2] != head_dim
        || cache_value_dims[2] != head_dim
    {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decode attention cache tensors must cover cache_len and match head dimensions",
        ));
    }

    if key_slot.shape().dims() != [num_kv_heads, head_dim]
        || value_slot.shape().dims() != [num_kv_heads, head_dim]
    {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decode attention slot tensors must have shape [num_kv_heads, head_dim]",
        ));
    }

    if num_query_heads % num_kv_heads != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "number of query heads must be divisible by number of KV heads",
        ));
    }

    let query_values = query.as_f32_slice()?;
    let key_slot_values = key_slot.as_f32_slice()?;
    let value_slot_values = value_slot.as_f32_slice()?;
    let cached_key_bytes = cached_key.as_bytes();
    let cached_value_bytes = cached_value.as_bytes();
    let queries_per_kv_head = num_query_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out_values = vec![0.0f32; out.element_count()];

    let thread_count =
        preferred_decode_attention_thread_count(num_query_heads, head_dim, cache_len + 1);
    if thread_count == 1 {
        compute_decode_attention_head_chunk(
            &query_values,
            cached_key_bytes,
            cached_value_bytes,
            &key_slot_values,
            &value_slot_values,
            &mut out_values,
            0,
            num_query_heads,
            cache_len,
            num_kv_heads,
            queries_per_kv_head,
            head_dim,
            scale,
        );
    } else {
        let heads_per_chunk = num_query_heads.div_ceil(thread_count);
        let chunk_width = heads_per_chunk * head_dim;

        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_width).enumerate() {
                let head_start = chunk_index * heads_per_chunk;
                let head_end = (head_start + heads_per_chunk).min(num_query_heads);
                let query_values = &query_values;
                let key_slot_values = &key_slot_values;
                let value_slot_values = &value_slot_values;

                scope.spawn(move || {
                    compute_decode_attention_head_chunk(
                        query_values,
                        cached_key_bytes,
                        cached_value_bytes,
                        key_slot_values,
                        value_slot_values,
                        out_chunk,
                        head_start,
                        head_end,
                        cache_len,
                        num_kv_heads,
                        queries_per_kv_head,
                        head_dim,
                        scale,
                    );
                });
            }
        });
    }

    out.copy_from_f32_slice(&out_values)
}

pub fn causal_self_attention_f32(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    prefixed_causal_attention_f32(query, key, value, 0, out)
}

fn validate_attention_tensor(tensor: &Tensor) -> Result<()> {
    tensor.ensure_dtype(DType::F32)?;
    tensor.ensure_contiguous()?;

    if tensor.shape().rank() != 3 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention tensors must have shape [seq_len, num_heads, head_dim]",
        ));
    }

    Ok(())
}

fn attention_index(
    position: usize,
    head: usize,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
) -> usize {
    ((position * num_heads + head) * head_dim) + dim
}

fn read_packed_f32(bytes: &[u8], index: usize) -> f32 {
    let start = index * DType::F32.size_in_bytes();
    let mut array = [0u8; 4];
    array.copy_from_slice(&bytes[start..start + 4]);
    f32::from_le_bytes(array)
}

#[allow(clippy::too_many_arguments)]
fn compute_attention_chunk(
    query_values: &[f32],
    key_values: &[f32],
    value_values: &[f32],
    out_chunk: &mut [f32],
    position_start: usize,
    position_end: usize,
    cache_len: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    queries_per_kv_head: usize,
    head_dim: usize,
    scale: f32,
) {
    for (local_position, query_position) in (position_start..position_end).enumerate() {
        let max_attended_position = cache_len + query_position;

        for query_head in 0..num_query_heads {
            let kv_head = query_head / queries_per_kv_head;
            let mut scores = Vec::with_capacity(max_attended_position + 1);
            let mut max_score = f32::NEG_INFINITY;

            for attended_position in 0..=max_attended_position {
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    let query_index =
                        attention_index(query_position, query_head, dim, num_query_heads, head_dim);
                    let key_index =
                        attention_index(attended_position, kv_head, dim, num_kv_heads, head_dim);
                    score += query_values[query_index] * key_values[key_index];
                }
                score *= scale;
                max_score = max_score.max(score);
                scores.push(score);
            }

            let mut score_sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                score_sum += *score;
            }

            for dim in 0..head_dim {
                let out_index =
                    attention_index(local_position, query_head, dim, num_query_heads, head_dim);
                let mut weighted_sum = 0.0f32;

                for (attended_position, normalized_score) in scores.iter().copied().enumerate() {
                    let value_index =
                        attention_index(attended_position, kv_head, dim, num_kv_heads, head_dim);
                    weighted_sum += (normalized_score / score_sum) * value_values[value_index];
                }

                out_chunk[out_index] = weighted_sum;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_decode_attention_head_chunk(
    query_values: &[f32],
    cached_key_bytes: &[u8],
    cached_value_bytes: &[u8],
    key_slot_values: &[f32],
    value_slot_values: &[f32],
    out_chunk: &mut [f32],
    head_start: usize,
    head_end: usize,
    cache_len: usize,
    num_kv_heads: usize,
    queries_per_kv_head: usize,
    head_dim: usize,
    scale: f32,
) {
    let attended_len = cache_len + 1;

    for (local_head_index, query_head) in (head_start..head_end).enumerate() {
        let kv_head = query_head / queries_per_kv_head;
        let query_base = query_head * head_dim;
        let slot_base = kv_head * head_dim;
        let mut scores = Vec::with_capacity(attended_len);
        let mut max_score = f32::NEG_INFINITY;

        for attended_position in 0..cache_len {
            let mut score = 0.0f32;
            for dim in 0..head_dim {
                let key_index =
                    attention_index(attended_position, kv_head, dim, num_kv_heads, head_dim);
                score += query_values[query_base + dim] * read_packed_f32(cached_key_bytes, key_index);
            }
            score *= scale;
            max_score = max_score.max(score);
            scores.push(score);
        }

        let mut slot_score = 0.0f32;
        for dim in 0..head_dim {
            slot_score += query_values[query_base + dim] * key_slot_values[slot_base + dim];
        }
        slot_score *= scale;
        max_score = max_score.max(slot_score);
        scores.push(slot_score);

        let mut score_sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            score_sum += *score;
        }

        for dim in 0..head_dim {
            let out_index = local_head_index * head_dim + dim;
            let mut weighted_sum = 0.0f32;

            for (attended_position, normalized_score) in
                scores[..cache_len].iter().copied().enumerate()
            {
                let value_index =
                    attention_index(attended_position, kv_head, dim, num_kv_heads, head_dim);
                weighted_sum +=
                    (normalized_score / score_sum) * read_packed_f32(cached_value_bytes, value_index);
            }

            weighted_sum +=
                (scores[cache_len] / score_sum) * value_slot_values[slot_base + dim];
            out_chunk[out_index] = weighted_sum;
        }
    }
}

fn preferred_attention_thread_count(
    query_len: usize,
    num_query_heads: usize,
    head_dim: usize,
    attended_len: usize,
) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    if query_len < 2 {
        return 1;
    }

    let total_work = query_len
        .saturating_mul(num_query_heads)
        .saturating_mul(head_dim)
        .saturating_mul(attended_len);
    if total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(query_len))
        .unwrap_or(1)
}

fn preferred_decode_attention_thread_count(
    num_query_heads: usize,
    head_dim: usize,
    attended_len: usize,
) -> usize {
    const MIN_PARALLEL_WORK: usize = 32_768;

    let total_work = num_query_heads
        .saturating_mul(head_dim)
        .saturating_mul(attended_len);
    if total_work < MIN_PARALLEL_WORK || num_query_heads < 2 {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(num_query_heads))
        .unwrap_or(1)
}

fn apply_rope_in_place(
    values: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
    rotary_dims: usize,
    theta: f32,
) {
    for position in 0..seq_len {
        let absolute_position = (position_offset + position) as f32;

        for head in 0..num_heads {
            for dim in (0..rotary_dims).step_by(2) {
                let angle = absolute_position / theta.powf(dim as f32 / head_dim as f32);
                let cosine = angle.cos();
                let sine = angle.sin();
                let base = ((position * num_heads + head) * head_dim) + dim;
                rotate_pair(values, base, cosine, sine);
            }
        }
    }
}

fn rotate_pair(values: &mut [f32], base: usize, cosine: f32, sine: f32) {
    let even = values[base];
    let odd = values[base + 1];
    values[base] = even * cosine - odd * sine;
    values[base + 1] = even * sine + odd * cosine;
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
    fn embedding_gather_f32_selects_requested_rows() {
        let embedding = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 2]).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 2]).unwrap()).unwrap();

        embedding_gather_f32(&embedding, &[2, 0], &mut out).unwrap();

        assert_eq!(out.to_vec_f32().unwrap(), vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn rope_f32_rotates_query_and_key_pairs() {
        let mut query = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut key = query.clone();

        rope_f32(&mut query, &mut key, 0, 2, 1.0).unwrap();

        approx_eq_slice(
            &query.to_vec_f32().unwrap(),
            &[1.0, 0.0, 0.5403023, 0.84147096],
            1e-6,
        );
        approx_eq_slice(
            &key.to_vec_f32().unwrap(),
            &[1.0, 0.0, 0.5403023, 0.84147096],
            1e-6,
        );
    }

    #[test]
    fn causal_self_attention_f32_respects_causal_mask() {
        let query = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let key = query.clone();
        let value = query.clone();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 1, 2]).unwrap()).unwrap();

        causal_self_attention_f32(&query, &key, &value, &mut out).unwrap();

        approx_eq_slice(
            &out.to_vec_f32().unwrap(),
            &[1.0, 0.0, 0.33023846, 0.66976154],
            1e-6,
        );
    }

    #[test]
    fn prefixed_causal_attention_f32_uses_cached_prefix() {
        let query =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1, 2]).unwrap(), vec![0.0, 1.0]).unwrap();
        let key = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let value = key.clone();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[1, 1, 2]).unwrap()).unwrap();

        prefixed_causal_attention_f32(&query, &key, &value, 1, &mut out).unwrap();

        approx_eq_slice(&out.to_vec_f32().unwrap(), &[0.33023846, 0.66976154], 1e-6);
    }

    #[test]
    fn decode_causal_attention_f32_matches_prefixed_reference() {
        let query = Tensor::from_f32_vec(
            Shape::from_slice(&[1, 2, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let cache_key = Tensor::from_f32_vec(
            Shape::from_slice(&[4, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let cache_value = cache_key.clone();
        let key_slot =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![0.0, 1.0]).unwrap();
        let value_slot = key_slot.clone();
        let full_key = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let full_value = full_key.clone();
        let mut decode_out =
            Tensor::zeros(DType::F32, Shape::from_slice(&[1, 2, 2]).unwrap()).unwrap();
        let mut prefixed_out =
            Tensor::zeros(DType::F32, Shape::from_slice(&[1, 2, 2]).unwrap()).unwrap();

        decode_causal_attention_f32(
            &query,
            &cache_key,
            &cache_value,
            1,
            &key_slot,
            &value_slot,
            &mut decode_out,
        )
        .unwrap();
        prefixed_causal_attention_f32(&query, &full_key, &full_value, 1, &mut prefixed_out)
            .unwrap();

        approx_eq_slice(
            &decode_out.to_vec_f32().unwrap(),
            &prefixed_out.to_vec_f32().unwrap(),
            1e-6,
        );
    }

    #[test]
    fn causal_self_attention_f32_supports_grouped_query_attention() {
        let query = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2, 2]).unwrap(),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        let key = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let value = key.clone();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[2, 2, 2]).unwrap()).unwrap();

        causal_self_attention_f32(&query, &key, &value, &mut out).unwrap();

        approx_eq_slice(
            &out.to_vec_f32().unwrap(),
            &[
                1.0, 0.0, 1.0, 0.0, 0.33023846, 0.66976154, 0.33023846, 0.66976154,
            ],
            1e-6,
        );
    }
}
