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
    let thread_count = preferred_embedding_thread_count(token_ids.len(), hidden_size);

    if thread_count == 1 {
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
    } else {
        let rows_per_chunk = token_ids.len().div_ceil(thread_count);
        let error_slot = std::sync::Mutex::new(None);

        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values
                .chunks_mut(rows_per_chunk * hidden_size)
                .enumerate()
            {
                let row_start = chunk_index * rows_per_chunk;
                let row_end = row_start + out_chunk.len() / hidden_size;
                let token_chunk = &token_ids[row_start..row_end];
                let embedding_values = &embedding_values;
                let error_slot = &error_slot;

                scope.spawn(move || {
                    for (local_row, token_id) in token_chunk.iter().copied().enumerate() {
                        let token_index = token_id as usize;
                        if token_index >= vocab_size {
                            let mut error = error_slot.lock().expect("embedding gather mutex poisoned");
                            if error.is_none() {
                                *error = Some(FerrisError::new(
                                    ErrorKind::InvalidShape,
                                    format!(
                                        "token id {token_index} is out of vocabulary range {vocab_size}"
                                    ),
                                ));
                            }
                            return;
                        }

                        let src_start = token_index * hidden_size;
                        let src_end = src_start + hidden_size;
                        let dst_start = local_row * hidden_size;
                        let dst_end = dst_start + hidden_size;
                        out_chunk[dst_start..dst_end]
                            .copy_from_slice(&embedding_values[src_start..src_end]);
                    }
                });
            }
        });

        if let Some(error) = error_slot
            .into_inner()
            .expect("embedding gather mutex poisoned")
        {
            return Err(error);
        }
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
    let query_len = query_dims[0];
    let query_heads = query_dims[1];
    let key_len = key_dims[0];
    let key_heads = key_dims[1];
    let head_dim = query_dims[2];

    if key_len != query_len || key_dims[2] != head_dim {
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

    apply_rope_pair_in_place(
        query.as_f32_slice_mut()?,
        key.as_f32_slice_mut()?,
        query_len,
        query_heads,
        key_heads,
        head_dim,
        position_offset,
        rotary_dims,
        theta,
    );
    Ok(())
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
    let out_values = out.as_f32_slice_mut()?;

    let thread_count =
        preferred_attention_thread_count(query_len, num_query_heads, head_dim, expected_kv_len);
    if thread_count == 1 {
        compute_attention_chunk(
            &query_values,
            &key_values,
            &value_values,
            out_values,
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
    Ok(())
}

pub fn cached_prefixed_causal_attention_f32(
    query: &Tensor,
    cached_key: &Tensor,
    cached_value: &Tensor,
    cache_len: usize,
    key_suffix: &Tensor,
    value_suffix: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    validate_attention_tensor(query)?;
    validate_attention_tensor(cached_key)?;
    validate_attention_tensor(cached_value)?;
    validate_attention_tensor(key_suffix)?;
    validate_attention_tensor(value_suffix)?;
    validate_attention_tensor(out)?;

    let query_dims = query.shape().dims();
    let cache_key_dims = cached_key.shape().dims();
    let cache_value_dims = cached_value.shape().dims();
    let key_suffix_dims = key_suffix.shape().dims();
    let value_suffix_dims = value_suffix.shape().dims();
    let out_dims = out.shape().dims();

    let query_len = query_dims[0];
    let num_query_heads = query_dims[1];
    let head_dim = query_dims[2];
    let num_kv_heads = key_suffix_dims[1];

    if out_dims != query_dims {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention output tensor must match query tensor shape",
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
            "cached attention tensors must cover cache_len and match head dimensions",
        ));
    }

    if key_suffix_dims[0] != query_len
        || value_suffix_dims[0] != query_len
        || value_suffix_dims[1] != num_kv_heads
        || key_suffix_dims[2] != head_dim
        || value_suffix_dims[2] != head_dim
    {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention suffix tensors must match query length and head dimension",
        ));
    }

    if value_suffix_dims != key_suffix_dims {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "attention key/value suffix tensors must have matching shapes",
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
    let key_suffix_values = key_suffix.as_f32_slice()?;
    let value_suffix_values = value_suffix.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let cached_key_values = cached_key.as_f32_slice().ok();
    let cached_value_values = cached_value.as_f32_slice().ok();
    let cached_key_bytes = cached_key.as_bytes();
    let cached_value_bytes = cached_value.as_bytes();

    let attended_len = cache_len
        .checked_add(query_len)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "attention length overflow"))?;
    let thread_count =
        preferred_attention_thread_count(query_len, num_query_heads, head_dim, attended_len);
    if thread_count == 1 {
        if let (Some(cached_key_values), Some(cached_value_values)) =
            (cached_key_values, cached_value_values)
        {
            compute_cached_prefixed_attention_chunk_f32(
                query_values,
                cached_key_values,
                cached_value_values,
                key_suffix_values,
                value_suffix_values,
                out_values,
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
            compute_cached_prefixed_attention_chunk_packed(
                query_values,
                cached_key_bytes,
                cached_value_bytes,
                key_suffix_values,
                value_suffix_values,
                out_values,
                0,
                query_len,
                cache_len,
                num_query_heads,
                num_kv_heads,
                queries_per_kv_head,
                head_dim,
                scale,
            );
        }
    } else {
        let positions_per_chunk = query_len.div_ceil(thread_count);
        let chunk_width = positions_per_chunk * num_query_heads * head_dim;

        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in out_values.chunks_mut(chunk_width).enumerate() {
                let position_start = chunk_index * positions_per_chunk;
                let position_end = (position_start + positions_per_chunk).min(query_len);
                let query_values = &query_values;
                let key_suffix_values = &key_suffix_values;
                let value_suffix_values = &value_suffix_values;
                let cached_key_values = cached_key_values;
                let cached_value_values = cached_value_values;

                scope.spawn(move || {
                    if let (Some(cached_key_values), Some(cached_value_values)) =
                        (cached_key_values, cached_value_values)
                    {
                        compute_cached_prefixed_attention_chunk_f32(
                            query_values,
                            cached_key_values,
                            cached_value_values,
                            key_suffix_values,
                            value_suffix_values,
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
                    } else {
                        compute_cached_prefixed_attention_chunk_packed(
                            query_values,
                            cached_key_bytes,
                            cached_value_bytes,
                            key_suffix_values,
                            value_suffix_values,
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
                    }
                });
            }
        });
    }

    Ok(())
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
    let cached_key_values = cached_key.as_f32_slice().ok();
    let cached_value_values = cached_value.as_f32_slice().ok();
    let cached_key_bytes = cached_key.as_bytes();
    let cached_value_bytes = cached_value.as_bytes();
    let queries_per_kv_head = num_query_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out_values = out.as_f32_slice_mut()?;

    let thread_count =
        preferred_decode_attention_thread_count(num_query_heads, head_dim, cache_len + 1);
    if thread_count == 1 {
        if let (Some(cached_key_values), Some(cached_value_values)) =
            (cached_key_values, cached_value_values)
        {
            compute_decode_attention_head_chunk_f32(
                &query_values,
                cached_key_values,
                cached_value_values,
                &key_slot_values,
                &value_slot_values,
                out_values,
                0,
                num_query_heads,
                cache_len,
                num_kv_heads,
                queries_per_kv_head,
                head_dim,
                scale,
            );
        } else {
            compute_decode_attention_head_chunk_packed(
                &query_values,
                cached_key_bytes,
                cached_value_bytes,
                &key_slot_values,
                &value_slot_values,
                out_values,
                0,
                num_query_heads,
                cache_len,
                num_kv_heads,
                queries_per_kv_head,
                head_dim,
                scale,
            );
        }
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
                let cached_key_values = cached_key_values;
                let cached_value_values = cached_value_values;

                scope.spawn(move || {
                    if let (Some(cached_key_values), Some(cached_value_values)) =
                        (cached_key_values, cached_value_values)
                    {
                        compute_decode_attention_head_chunk_f32(
                            query_values,
                            cached_key_values,
                            cached_value_values,
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
                    } else {
                        compute_decode_attention_head_chunk_packed(
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
                    }
                });
            }
        });
    }
    Ok(())
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
            let query_base =
                attention_index(query_position, query_head, 0, num_query_heads, head_dim);
            let query_row = &query_values[query_base..query_base + head_dim];
            let out_base =
                attention_index(local_position, query_head, 0, num_query_heads, head_dim);
            let out_row = &mut out_chunk[out_base..out_base + head_dim];
            out_row.fill(0.0);
            let mut max_score = f32::NEG_INFINITY;
            let mut score_sum = 0.0f32;
            for attended_position in 0..=max_attended_position {
                let key_base =
                    attention_index(attended_position, kv_head, 0, num_kv_heads, head_dim);
                let key_row = &key_values[key_base..key_base + head_dim];
                let score = dot_product(query_row, key_row) * scale;
                let weight = update_online_softmax_accumulator(
                    out_row,
                    &mut max_score,
                    &mut score_sum,
                    score,
                );

                let value_base =
                    attention_index(attended_position, kv_head, 0, num_kv_heads, head_dim);
                let value_row = &value_values[value_base..value_base + head_dim];
                accumulate_weighted_values(out_row, value_row, weight);
            }

            let inv_score_sum = 1.0 / score_sum;
            for value in out_row.iter_mut() {
                *value *= inv_score_sum;
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn compute_cached_prefixed_attention_chunk_f32(
    query_values: &[f32],
    cached_key_values: &[f32],
    cached_value_values: &[f32],
    key_suffix_values: &[f32],
    value_suffix_values: &[f32],
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
            let query_base =
                attention_index(query_position, query_head, 0, num_query_heads, head_dim);
            let query_row = &query_values[query_base..query_base + head_dim];
            let out_base =
                attention_index(local_position, query_head, 0, num_query_heads, head_dim);
            let out_row = &mut out_chunk[out_base..out_base + head_dim];
            out_row.fill(0.0);
            let mut max_score = f32::NEG_INFINITY;
            let mut score_sum = 0.0f32;

            for attended_position in 0..=max_attended_position {
                let score = if attended_position < cache_len {
                    dot_product_with_values(
                        query_row,
                        cached_key_values,
                        attended_position,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                    )
                } else {
                    dot_product_with_values(
                        query_row,
                        key_suffix_values,
                        attended_position - cache_len,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                    )
                } * scale;

                let weight = update_online_softmax_accumulator(
                    out_row,
                    &mut max_score,
                    &mut score_sum,
                    score,
                );

                if attended_position < cache_len {
                    accumulate_weighted_values_from_cache(
                        out_row,
                        cached_value_values,
                        attended_position,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                        weight,
                    );
                } else {
                    accumulate_weighted_values_from_cache(
                        out_row,
                        value_suffix_values,
                        attended_position - cache_len,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                        weight,
                    );
                }
            }

            let inv_score_sum = 1.0 / score_sum;
            for value in out_row.iter_mut() {
                *value *= inv_score_sum;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_cached_prefixed_attention_chunk_packed(
    query_values: &[f32],
    cached_key_bytes: &[u8],
    cached_value_bytes: &[u8],
    key_suffix_values: &[f32],
    value_suffix_values: &[f32],
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
            let query_base =
                attention_index(query_position, query_head, 0, num_query_heads, head_dim);
            let query_row = &query_values[query_base..query_base + head_dim];
            let out_base =
                attention_index(local_position, query_head, 0, num_query_heads, head_dim);
            let out_row = &mut out_chunk[out_base..out_base + head_dim];
            out_row.fill(0.0);
            let mut max_score = f32::NEG_INFINITY;
            let mut score_sum = 0.0f32;

            for attended_position in 0..=max_attended_position {
                let score = if attended_position < cache_len {
                    dot_product_with_packed_values(
                        query_row,
                        cached_key_bytes,
                        attended_position,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                    )
                } else {
                    dot_product_with_values(
                        query_row,
                        key_suffix_values,
                        attended_position - cache_len,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                    )
                } * scale;

                let weight = update_online_softmax_accumulator(
                    out_row,
                    &mut max_score,
                    &mut score_sum,
                    score,
                );

                if attended_position < cache_len {
                    accumulate_weighted_packed_values(
                        out_row,
                        cached_value_bytes,
                        attended_position,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                        weight,
                    );
                } else {
                    accumulate_weighted_values_from_cache(
                        out_row,
                        value_suffix_values,
                        attended_position - cache_len,
                        kv_head,
                        num_kv_heads,
                        head_dim,
                        weight,
                    );
                }
            }

            let inv_score_sum = 1.0 / score_sum;
            for value in out_row.iter_mut() {
                *value *= inv_score_sum;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_decode_attention_head_chunk_f32(
    query_values: &[f32],
    cached_key_values: &[f32],
    cached_value_values: &[f32],
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
    for (local_head_index, query_head) in (head_start..head_end).enumerate() {
        let kv_head = query_head / queries_per_kv_head;
        let query_base = query_head * head_dim;
        let slot_base = kv_head * head_dim;
        let query_row = &query_values[query_base..query_base + head_dim];
        let out_base = local_head_index * head_dim;
        let out_row = &mut out_chunk[out_base..out_base + head_dim];
        out_row.fill(0.0);
        let mut max_score = f32::NEG_INFINITY;
        let mut score_sum = 0.0f32;

        for attended_position in 0..cache_len {
            let score = dot_product_with_values(
                query_row,
                cached_key_values,
                attended_position,
                kv_head,
                num_kv_heads,
                head_dim,
            ) * scale;
            let weight =
                update_online_softmax_accumulator(out_row, &mut max_score, &mut score_sum, score);
            accumulate_weighted_values_from_cache(
                out_row,
                cached_value_values,
                attended_position,
                kv_head,
                num_kv_heads,
                head_dim,
                weight,
            );
        }

        let slot_row = &key_slot_values[slot_base..slot_base + head_dim];
        let slot_score = dot_product(query_row, slot_row) * scale;
        let slot_weight =
            update_online_softmax_accumulator(out_row, &mut max_score, &mut score_sum, slot_score);
        let slot_value_row = &value_slot_values[slot_base..slot_base + head_dim];
        accumulate_weighted_values(out_row, slot_value_row, slot_weight);

        let inv_score_sum = 1.0 / score_sum;
        for value in out_row.iter_mut() {
            *value *= inv_score_sum;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_decode_attention_head_chunk_packed(
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
    for (local_head_index, query_head) in (head_start..head_end).enumerate() {
        let kv_head = query_head / queries_per_kv_head;
        let query_base = query_head * head_dim;
        let slot_base = kv_head * head_dim;
        let query_row = &query_values[query_base..query_base + head_dim];
        let out_base = local_head_index * head_dim;
        let out_row = &mut out_chunk[out_base..out_base + head_dim];
        out_row.fill(0.0);
        let mut max_score = f32::NEG_INFINITY;
        let mut score_sum = 0.0f32;

        for attended_position in 0..cache_len {
            let score = dot_product_with_packed_values(
                query_row,
                cached_key_bytes,
                attended_position,
                kv_head,
                num_kv_heads,
                head_dim,
            ) * scale;
            let weight =
                update_online_softmax_accumulator(out_row, &mut max_score, &mut score_sum, score);
            accumulate_weighted_packed_values(
                out_row,
                cached_value_bytes,
                attended_position,
                kv_head,
                num_kv_heads,
                head_dim,
                weight,
            );
        }

        let slot_row = &key_slot_values[slot_base..slot_base + head_dim];
        let slot_score = dot_product(query_row, slot_row) * scale;
        let slot_weight =
            update_online_softmax_accumulator(out_row, &mut max_score, &mut score_sum, slot_score);
        let slot_value_row = &value_slot_values[slot_base..slot_base + head_dim];
        accumulate_weighted_values(out_row, slot_value_row, slot_weight);

        let inv_score_sum = 1.0 / score_sum;
        for value in out_row.iter_mut() {
            *value *= inv_score_sum;
        }
    }
}

fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn dot_product_with_values(
    query_row: &[f32],
    values: &[f32],
    position: usize,
    head: usize,
    num_heads: usize,
    head_dim: usize,
) -> f32 {
    let start = attention_index(position, head, 0, num_heads, head_dim);
    dot_product(query_row, &values[start..start + head_dim])
}

fn dot_product_with_packed_values(
    query_row: &[f32],
    packed_values: &[u8],
    position: usize,
    head: usize,
    num_heads: usize,
    head_dim: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for (dim, query_value) in query_row.iter().copied().enumerate() {
        let index = attention_index(position, head, dim, num_heads, head_dim);
        sum += query_value * read_packed_f32(packed_values, index);
    }
    sum
}

fn update_online_softmax_accumulator(
    out_row: &mut [f32],
    max_score: &mut f32,
    score_sum: &mut f32,
    score: f32,
) -> f32 {
    if score > *max_score {
        let rescale = if max_score.is_finite() {
            (*max_score - score).exp()
        } else {
            0.0
        };
        for value in out_row.iter_mut() {
            *value *= rescale;
        }
        *score_sum *= rescale;
        *max_score = score;
        *score_sum += 1.0;
        1.0
    } else {
        let weight = (score - *max_score).exp();
        *score_sum += weight;
        weight
    }
}

fn accumulate_weighted_values(out_row: &mut [f32], value_row: &[f32], weight: f32) {
    for (out_value, value) in out_row.iter_mut().zip(value_row.iter().copied()) {
        *out_value += weight * value;
    }
}

fn accumulate_weighted_values_from_cache(
    out_row: &mut [f32],
    values: &[f32],
    position: usize,
    head: usize,
    num_heads: usize,
    head_dim: usize,
    weight: f32,
) {
    let start = attention_index(position, head, 0, num_heads, head_dim);
    accumulate_weighted_values(out_row, &values[start..start + head_dim], weight);
}

fn accumulate_weighted_packed_values(
    out_row: &mut [f32],
    packed_values: &[u8],
    position: usize,
    head: usize,
    num_heads: usize,
    head_dim: usize,
    weight: f32,
) {
    for (dim, out_value) in out_row.iter_mut().enumerate() {
        let index = attention_index(position, head, dim, num_heads, head_dim);
        *out_value += weight * read_packed_f32(packed_values, index);
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

fn preferred_embedding_thread_count(rows: usize, hidden_size: usize) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    let total_work = rows.saturating_mul(hidden_size);
    if rows < 2 || total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(rows))
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

fn apply_rope_pair_in_place(
    query_values: &mut [f32],
    key_values: &mut [f32],
    seq_len: usize,
    query_heads: usize,
    key_heads: usize,
    head_dim: usize,
    position_offset: usize,
    rotary_dims: usize,
    theta: f32,
) {
    let rotary_pairs = rotary_dims / 2;
    let inverse_frequency_step = theta.powf(-2.0 / head_dim as f32);

    for position in 0..seq_len {
        let absolute_position = (position_offset + position) as f32;
        let mut inverse_frequency = 1.0f32;

        for pair_index in 0..rotary_pairs {
            let angle = absolute_position * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            let low_dim = pair_index;
            let high_dim = pair_index + rotary_pairs;

            for head in 0..query_heads {
                let base = (position * query_heads + head) * head_dim;
                rotate_rope_half_pair(query_values, base + low_dim, base + high_dim, cosine, sine);
            }

            for head in 0..key_heads {
                let base = (position * key_heads + head) * head_dim;
                rotate_rope_half_pair(key_values, base + low_dim, base + high_dim, cosine, sine);
            }

            inverse_frequency *= inverse_frequency_step;
        }
    }
}

fn rotate_rope_half_pair(
    values: &mut [f32],
    low_index: usize,
    high_index: usize,
    cosine: f32,
    sine: f32,
) {
    let low = values[low_index];
    let high = values[high_index];
    values[low_index] = low * cosine - high * sine;
    values[high_index] = high * cosine + low * sine;
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
    fn cached_prefixed_causal_attention_f32_matches_prefixed_reference() {
        let query = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let cached_key = Tensor::from_f32_vec(
            Shape::from_slice(&[4, 1, 2]).unwrap(),
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let cached_value = cached_key.clone();
        let key_suffix = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let value_suffix = key_suffix.clone();
        let full_key = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 1, 2]).unwrap(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let full_value = full_key.clone();
        let mut cached_out =
            Tensor::zeros(DType::F32, Shape::from_slice(&[2, 1, 2]).unwrap()).unwrap();
        let mut prefixed_out =
            Tensor::zeros(DType::F32, Shape::from_slice(&[2, 1, 2]).unwrap()).unwrap();

        cached_prefixed_causal_attention_f32(
            &query,
            &cached_key,
            &cached_value,
            1,
            &key_suffix,
            &value_suffix,
            &mut cached_out,
        )
        .unwrap();
        prefixed_causal_attention_f32(&query, &full_key, &full_value, 1, &mut prefixed_out)
            .unwrap();

        approx_eq_slice(
            &cached_out.to_vec_f32().unwrap(),
            &prefixed_out.to_vec_f32().unwrap(),
            1e-6,
        );
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
