use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Shape, Tensor};
use ferrisinfer_kernel::cpu::attention::{
    cached_prefixed_causal_attention_f32, causal_self_attention_f32, decode_causal_attention_f32,
    embedding_gather_f32, rope_f32,
};
use ferrisinfer_kernel::cpu::elementwise::{add_f32_in_place, mul_f32_in_place, silu_f32_in_place};
use ferrisinfer_kernel::cpu::matmul::{
    matmul_f32, matmul_rhs_transposed_argmax_f32, matmul_rhs_transposed_f32,
};
use ferrisinfer_kernel::cpu::reduction::rms_norm_f32;
use ferrisinfer_model::{ActivationKind, AttentionLayout, DecoderOnlyModel, ModelConfig, NormKind};

use crate::kv_cache::{KvCache, KvCacheStorageKind};
use crate::sampler::TokenSample;

pub struct ReferenceDecoderBlockWeights<'a> {
    pub attention_norm: &'a Tensor,
    pub wq: &'a Tensor,
    pub wq_bias: Option<&'a Tensor>,
    pub wk: &'a Tensor,
    pub wk_bias: Option<&'a Tensor>,
    pub wv: &'a Tensor,
    pub wv_bias: Option<&'a Tensor>,
    pub wo: &'a Tensor,
    pub wo_bias: Option<&'a Tensor>,
    pub ffn_norm: &'a Tensor,
    pub w1: &'a Tensor,
    pub w1_bias: Option<&'a Tensor>,
    pub w2: &'a Tensor,
    pub w2_bias: Option<&'a Tensor>,
    pub w3: &'a Tensor,
    pub w3_bias: Option<&'a Tensor>,
}

#[derive(Debug, Clone, Copy)]
pub struct ReferenceBlockConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
    pub rotary_dims: usize,
}

pub(crate) struct PrefillWorkspace {
    normed: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention: Tensor,
    gated_product: Tensor,
    gate_projection: Tensor,
}

impl PrefillWorkspace {
    fn for_block(config: ReferenceBlockConfig, seq_len: usize) -> Result<Self> {
        let hidden_size = config.num_heads * config.head_dim;
        let kv_hidden_size = config.num_kv_heads * config.head_dim;

        Ok(Self {
            normed: zeros_2d(seq_len, hidden_size)?,
            query: zeros_2d(seq_len, hidden_size)?,
            key: zeros_2d(seq_len, kv_hidden_size)?,
            value: zeros_2d(seq_len, kv_hidden_size)?,
            attention: zeros_2d(seq_len, hidden_size)?,
            gated_product: zeros_2d(seq_len, config.intermediate_size)?,
            gate_projection: zeros_2d(seq_len, config.intermediate_size)?,
        })
    }
}

pub(crate) struct DecodeWorkspace {
    hidden: Tensor,
    normed: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention: Tensor,
    gated_product: Tensor,
    gate_projection: Tensor,
    logits: Tensor,
}

impl DecodeWorkspace {
    pub(crate) fn new(config: &ModelConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_key_value_heads * config.head_dim();

        Ok(Self {
            hidden: zeros_2d(1, hidden_size)?,
            normed: zeros_2d(1, hidden_size)?,
            query: zeros_2d(1, hidden_size)?,
            key: zeros_2d(1, kv_hidden_size)?,
            value: zeros_2d(1, kv_hidden_size)?,
            attention: zeros_2d(1, hidden_size)?,
            gated_product: zeros_2d(1, config.intermediate_size)?,
            gate_projection: zeros_2d(1, config.intermediate_size)?,
            logits: zeros_2d(1, config.vocab_size)?,
        })
    }

    pub(crate) fn logits(&self) -> &Tensor {
        &self.logits
    }
}
pub fn decoder_block_forward_f32(
    input: &Tensor,
    weights: &ReferenceDecoderBlockWeights<'_>,
    config: ReferenceBlockConfig,
) -> Result<Tensor> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;

    if input.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder block input must have shape [seq_len, hidden_size]",
        ));
    }

    let seq_len = input.shape().dims()[0];
    let hidden_size = config.num_heads * config.head_dim;
    let mut output = zeros_2d(seq_len, hidden_size)?;
    let mut workspace = PrefillWorkspace::for_block(config, seq_len)?;
    decoder_block_forward_buffered_f32(
        input,
        &mut output,
        weights,
        config,
        0,
        None,
        &mut workspace,
    )?;
    Ok(output)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decoder_model_prefill_last_token_logits_with_kv_cache_f32(
    model: &DecoderOnlyModel,
    token_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Tensor> {
    let normalized = decoder_model_prefill_hidden_with_kv_capture_f32(model, token_ids, kv_cache)?;
    let last_hidden = select_last_row_f32(&normalized)?;
    let output_weight = output_weight(model)?;

    let mut logits = zeros_2d(1, model.config().vocab_size)?;
    linear_right_transposed_f32(&last_hidden, output_weight, &mut logits)?;
    Ok(logits)
}

pub(crate) fn decoder_model_prefill_last_token_sample_with_kv_cache_f32(
    model: &DecoderOnlyModel,
    token_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<TokenSample> {
    let normalized = decoder_model_prefill_hidden_with_kv_capture_f32(model, token_ids, kv_cache)?;
    let last_hidden = select_last_row_f32(&normalized)?;
    linear_right_transposed_argmax_f32(&last_hidden, output_weight(model)?)
}

fn decoder_block_forward_buffered_f32(
    input: &Tensor,
    output: &mut Tensor,
    weights: &ReferenceDecoderBlockWeights<'_>,
    config: ReferenceBlockConfig,
    position_offset: usize,
    kv_capture: Option<(&mut KvCache, usize, usize)>,
    workspace: &mut PrefillWorkspace,
) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    output.ensure_dtype(DType::F32)?;
    output.ensure_contiguous()?;

    if input.shape().rank() != 2 || output.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder block forward requires rank-2 input and output tensors",
        ));
    }

    if config.num_heads == 0
        || config.num_kv_heads == 0
        || config.head_dim == 0
        || config.intermediate_size == 0
    {
        return Err(FerrisError::new(
            ErrorKind::InvalidConfig,
            "decoder block dimensions must be greater than zero",
        ));
    }

    if config.num_heads % config.num_kv_heads != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidConfig,
            "decoder block requires num_heads to be divisible by num_kv_heads",
        ));
    }

    let dims = input.shape().dims();
    let seq_len = dims[0];
    let hidden_size = dims[1];
    let query_hidden_size = config.num_heads * config.head_dim;
    let kv_hidden_size = config.num_kv_heads * config.head_dim;

    if hidden_size != query_hidden_size {
        return Err(FerrisError::new(
            ErrorKind::InvalidConfig,
            "hidden size must equal num_heads * head_dim",
        ));
    }

    if output.shape().dims() != [seq_len, hidden_size] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder block output must have shape [seq_len, hidden_size]",
        ));
    }

    workspace
        .normed
        .reshape_in_place(Shape::from_slice(&[seq_len, hidden_size])?)?;
    workspace
        .query
        .reshape_in_place(Shape::from_slice(&[seq_len, hidden_size])?)?;
    workspace
        .key
        .reshape_in_place(Shape::from_slice(&[seq_len, kv_hidden_size])?)?;
    workspace
        .value
        .reshape_in_place(Shape::from_slice(&[seq_len, kv_hidden_size])?)?;
    workspace
        .attention
        .reshape_in_place(Shape::from_slice(&[seq_len, hidden_size])?)?;
    workspace
        .gated_product
        .reshape_in_place(Shape::from_slice(&[seq_len, config.intermediate_size])?)?;
    workspace
        .gate_projection
        .reshape_in_place(Shape::from_slice(&[seq_len, config.intermediate_size])?)?;

    rms_norm_f32(
        input,
        weights.attention_norm,
        &mut workspace.normed,
        config.rms_norm_epsilon,
    )?;

    linear_f32_with_bias(
        &workspace.normed,
        weights.wq,
        weights.wq_bias,
        &mut workspace.query,
    )?;
    linear_f32_with_bias(
        &workspace.normed,
        weights.wk,
        weights.wk_bias,
        &mut workspace.key,
    )?;
    linear_f32_with_bias(
        &workspace.normed,
        weights.wv,
        weights.wv_bias,
        &mut workspace.value,
    )?;

    workspace.query.reshape_in_place(Shape::from_slice(&[
        seq_len,
        config.num_heads,
        config.head_dim,
    ])?)?;
    workspace.key.reshape_in_place(Shape::from_slice(&[
        seq_len,
        config.num_kv_heads,
        config.head_dim,
    ])?)?;
    workspace.value.reshape_in_place(Shape::from_slice(&[
        seq_len,
        config.num_kv_heads,
        config.head_dim,
    ])?)?;
    workspace.attention.reshape_in_place(Shape::from_slice(&[
        seq_len,
        config.num_heads,
        config.head_dim,
    ])?)?;

    rope_f32(
        &mut workspace.query,
        &mut workspace.key,
        position_offset,
        config.rotary_dims,
        config.rope_theta,
    )?;

    match kv_capture {
        Some((kv_cache, layer_index, start_position)) => {
            prefill_causal_attention_with_kv_cache_f32(
                &workspace.query,
                &workspace.key,
                &workspace.value,
                &mut workspace.attention,
                kv_cache,
                layer_index,
                start_position,
            )?;

            kv_cache.write_sequence_uncommitted_f32(
                layer_index,
                start_position,
                &workspace.key,
                &workspace.value,
            )?;
        }
        None => {
            causal_self_attention_f32(
                &workspace.query,
                &workspace.key,
                &workspace.value,
                &mut workspace.attention,
            )?;
        }
    }
    workspace
        .attention
        .reshape_in_place(Shape::from_slice(&[seq_len, hidden_size])?)?;

    linear_f32_with_bias(
        &workspace.attention,
        weights.wo,
        weights.wo_bias,
        &mut workspace.normed,
    )?;
    add_f32_in_place(&mut workspace.normed, input)?;

    rms_norm_f32(
        &workspace.normed,
        weights.ffn_norm,
        &mut workspace.attention,
        config.rms_norm_epsilon,
    )?;

    linear_f32_with_bias(
        &workspace.attention,
        weights.w1,
        weights.w1_bias,
        &mut workspace.gated_product,
    )?;
    linear_f32_with_bias(
        &workspace.attention,
        weights.w3,
        weights.w3_bias,
        &mut workspace.gate_projection,
    )?;

    silu_f32_in_place(&mut workspace.gate_projection)?;
    mul_f32_in_place(&mut workspace.gated_product, &workspace.gate_projection)?;

    linear_f32_with_bias(
        &workspace.gated_product,
        weights.w2,
        weights.w2_bias,
        output,
    )?;
    add_f32_in_place(output, &workspace.normed)?;
    Ok(())
}

fn decoder_model_prefill_hidden_with_kv_capture_f32(
    model: &DecoderOnlyModel,
    token_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Tensor> {
    validate_supported_reference_model(model.config())?;

    if token_ids.is_empty() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "prefill reference path requires at least one token",
        ));
    }

    if token_ids.len() > kv_cache.remaining_tokens() {
        return Err(FerrisError::new(
            ErrorKind::Runtime,
            "batch prefill exceeds KV cache capacity",
        ));
    }

    let config = model.config();
    let block = block_config(config);
    let start_position = kv_cache.used_tokens();
    let embedding = get_weight(model, "tok_embeddings.weight")?;
    let mut hidden = zeros_2d(token_ids.len(), config.hidden_size)?;
    embedding_gather_f32(embedding, token_ids, &mut hidden)?;
    let mut next_hidden = zeros_2d(token_ids.len(), config.hidden_size)?;
    let mut workspace = PrefillWorkspace::for_block(block, token_ids.len())?;

    for layer in 0..config.num_layers {
        let weights = block_weights(model, layer)?;
        decoder_block_forward_buffered_f32(
            &hidden,
            &mut next_hidden,
            &weights,
            block,
            start_position,
            Some((kv_cache, layer, start_position)),
            &mut workspace,
        )?;
        std::mem::swap(&mut hidden, &mut next_hidden);
    }

    rms_norm_f32(
        &hidden,
        get_weight(model, "norm.weight")?,
        &mut next_hidden,
        config.norm.epsilon,
    )?;
    Ok(next_hidden)
}

fn prefill_causal_attention_with_kv_cache_f32(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention: &mut Tensor,
    kv_cache: &KvCache,
    layer_index: usize,
    start_position: usize,
) -> Result<()> {
    if start_position == 0 {
        return causal_self_attention_f32(query, key, value, attention);
    }

    match kv_cache.storage_kind() {
        KvCacheStorageKind::Contiguous => {
            let cache_layer = kv_cache.layer(layer_index)?;
            cached_prefixed_causal_attention_f32(
                query,
                cache_layer.key(),
                cache_layer.value(),
                start_position,
                key,
                value,
                attention,
            )
        }
        KvCacheStorageKind::Paged => {
            let (cached_key, cached_value) =
                kv_cache.read_prefix_f32(layer_index, start_position)?;
            cached_prefixed_causal_attention_f32(
                query,
                &cached_key,
                &cached_value,
                start_position,
                key,
                value,
                attention,
            )
        }
    }
}

fn decode_causal_attention_with_kv_cache_f32(
    query: &Tensor,
    key: &mut Tensor,
    value: &mut Tensor,
    attention: &mut Tensor,
    kv_cache: &mut KvCache,
    layer_index: usize,
    position: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if position == 0 {
        causal_self_attention_f32(query, key, value, attention)?;
    } else {
        match kv_cache.storage_kind() {
            KvCacheStorageKind::Contiguous => {
                key.reshape_in_place(Shape::from_slice(&[num_kv_heads, head_dim])?)?;
                value.reshape_in_place(Shape::from_slice(&[num_kv_heads, head_dim])?)?;

                let cache_layer = kv_cache.layer(layer_index)?;
                decode_causal_attention_f32(
                    query,
                    cache_layer.key(),
                    cache_layer.value(),
                    position,
                    key,
                    value,
                    attention,
                )?;
            }
            KvCacheStorageKind::Paged => {
                let (cached_key, cached_value) = kv_cache.read_prefix_f32(layer_index, position)?;
                cached_prefixed_causal_attention_f32(
                    query,
                    &cached_key,
                    &cached_value,
                    position,
                    key,
                    value,
                    attention,
                )?;
            }
        }
    }

    key.reshape_in_place(Shape::from_slice(&[num_kv_heads, head_dim])?)?;
    value.reshape_in_place(Shape::from_slice(&[num_kv_heads, head_dim])?)?;
    kv_cache.write_uncommitted_f32(layer_index, position, key, value)
}

pub fn decoder_model_forward_f32(model: &DecoderOnlyModel, token_ids: &[u32]) -> Result<Tensor> {
    let normalized = decoder_model_hidden_f32(model, token_ids)?;
    let output_weight = output_weight(model)?;

    let mut logits = zeros_2d(token_ids.len(), model.config().vocab_size)?;
    linear_right_transposed_f32(&normalized, output_weight, &mut logits)?;
    Ok(logits)
}

pub fn decoder_model_last_token_logits_f32(
    model: &DecoderOnlyModel,
    token_ids: &[u32],
) -> Result<Tensor> {
    let normalized = decoder_model_hidden_f32(model, token_ids)?;
    let last_hidden = select_last_row_f32(&normalized)?;
    let output_weight = output_weight(model)?;

    let mut logits = zeros_2d(1, model.config().vocab_size)?;
    linear_right_transposed_f32(&last_hidden, output_weight, &mut logits)?;
    Ok(logits)
}

pub fn decoder_model_token_logits_with_kv_cache_f32(
    model: &DecoderOnlyModel,
    token_id: u32,
    kv_cache: &mut KvCache,
    position: usize,
) -> Result<Tensor> {
    let mut workspace = DecodeWorkspace::new(model.config())?;
    decoder_model_token_logits_with_kv_cache_buffered_f32(
        model,
        token_id,
        kv_cache,
        position,
        &mut workspace,
    )?;
    Ok(workspace.logits)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decoder_model_token_sample_with_kv_cache_f32(
    model: &DecoderOnlyModel,
    token_id: u32,
    kv_cache: &mut KvCache,
    position: usize,
) -> Result<TokenSample> {
    let mut workspace = DecodeWorkspace::new(model.config())?;
    decoder_model_token_sample_with_kv_cache_buffered_f32(
        model,
        token_id,
        kv_cache,
        position,
        &mut workspace,
    )
}

pub(crate) fn decoder_model_token_logits_with_kv_cache_buffered_f32(
    model: &DecoderOnlyModel,
    token_id: u32,
    kv_cache: &mut KvCache,
    position: usize,
    workspace: &mut DecodeWorkspace,
) -> Result<()> {
    validate_supported_reference_model(model.config())?;

    let config = model.config();
    decoder_model_token_hidden_with_kv_cache_buffered_f32(
        model, token_id, kv_cache, position, workspace,
    )?;

    rms_norm_f32(
        &workspace.hidden,
        get_weight(model, "norm.weight")?,
        &mut workspace.normed,
        config.norm.epsilon,
    )?;
    linear_right_transposed_f32(
        &workspace.normed,
        output_weight(model)?,
        &mut workspace.logits,
    )
}

pub(crate) fn decoder_model_token_sample_with_kv_cache_buffered_f32(
    model: &DecoderOnlyModel,
    token_id: u32,
    kv_cache: &mut KvCache,
    position: usize,
    workspace: &mut DecodeWorkspace,
) -> Result<TokenSample> {
    validate_supported_reference_model(model.config())?;

    let config = model.config();
    decoder_model_token_hidden_with_kv_cache_buffered_f32(
        model, token_id, kv_cache, position, workspace,
    )?;

    rms_norm_f32(
        &workspace.hidden,
        get_weight(model, "norm.weight")?,
        &mut workspace.normed,
        config.norm.epsilon,
    )?;

    linear_right_transposed_argmax_f32(&workspace.normed, output_weight(model)?)
}

fn decoder_model_hidden_f32(model: &DecoderOnlyModel, token_ids: &[u32]) -> Result<Tensor> {
    validate_supported_reference_model(model.config())?;

    if token_ids.is_empty() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder_model_hidden_f32 requires at least one token",
        ));
    }

    let config = model.config();
    let block = block_config(config);
    let embedding = get_weight(model, "tok_embeddings.weight")?;
    let mut hidden = zeros_2d(token_ids.len(), config.hidden_size)?;
    embedding_gather_f32(embedding, token_ids, &mut hidden)?;
    let mut next_hidden = zeros_2d(token_ids.len(), config.hidden_size)?;
    let mut workspace = PrefillWorkspace::for_block(block, token_ids.len())?;

    for layer in 0..config.num_layers {
        let weights = block_weights(model, layer)?;
        decoder_block_forward_buffered_f32(
            &hidden,
            &mut next_hidden,
            &weights,
            block,
            0,
            None,
            &mut workspace,
        )?;
        std::mem::swap(&mut hidden, &mut next_hidden);
    }

    rms_norm_f32(
        &hidden,
        get_weight(model, "norm.weight")?,
        &mut next_hidden,
        config.norm.epsilon,
    )?;
    Ok(next_hidden)
}
fn decoder_model_token_hidden_with_kv_cache_buffered_f32(
    model: &DecoderOnlyModel,
    token_id: u32,
    kv_cache: &mut KvCache,
    position: usize,
    workspace: &mut DecodeWorkspace,
) -> Result<()> {
    let config = model.config();
    let hidden_size = config.hidden_size;
    let kv_hidden_size = config.num_key_value_heads * config.head_dim();

    workspace
        .hidden
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    workspace
        .normed
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    workspace
        .query
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    workspace
        .key
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    workspace
        .value
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    workspace
        .attention
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;

    let embedding = get_weight(model, "tok_embeddings.weight")?;
    embedding_gather_f32(embedding, &[token_id], &mut workspace.hidden)?;

    for layer in 0..config.num_layers {
        let weights = block_weights(model, layer)?;
        decoder_block_decode_step_buffered_f32(
            &weights,
            block_config(config),
            kv_cache,
            layer,
            position,
            workspace,
        )?;
    }

    Ok(())
}

fn decoder_block_decode_step_buffered_f32(
    weights: &ReferenceDecoderBlockWeights<'_>,
    config: ReferenceBlockConfig,
    kv_cache: &mut KvCache,
    layer_index: usize,
    position: usize,
    workspace: &mut DecodeWorkspace,
) -> Result<()> {
    let hidden_size = config.num_heads * config.head_dim;
    let kv_hidden_size = config.num_kv_heads * config.head_dim;

    workspace.hidden.ensure_dtype(DType::F32)?;
    workspace.hidden.ensure_contiguous()?;
    if workspace.hidden.shape().dims() != [1, hidden_size] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decode workspace hidden state must have shape [1, hidden_size]",
        ));
    }

    workspace
        .normed
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    rms_norm_f32(
        &workspace.hidden,
        weights.attention_norm,
        &mut workspace.normed,
        config.rms_norm_epsilon,
    )?;

    workspace
        .query
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    workspace
        .key
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    workspace
        .value
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    linear_f32_with_bias(
        &workspace.normed,
        weights.wq,
        weights.wq_bias,
        &mut workspace.query,
    )?;
    linear_f32_with_bias(
        &workspace.normed,
        weights.wk,
        weights.wk_bias,
        &mut workspace.key,
    )?;
    linear_f32_with_bias(
        &workspace.normed,
        weights.wv,
        weights.wv_bias,
        &mut workspace.value,
    )?;

    workspace.query.reshape_in_place(Shape::from_slice(&[
        1,
        config.num_heads,
        config.head_dim,
    ])?)?;
    workspace.key.reshape_in_place(Shape::from_slice(&[
        1,
        config.num_kv_heads,
        config.head_dim,
    ])?)?;
    workspace.value.reshape_in_place(Shape::from_slice(&[
        1,
        config.num_kv_heads,
        config.head_dim,
    ])?)?;
    rope_f32(
        &mut workspace.query,
        &mut workspace.key,
        position,
        config.rotary_dims,
        config.rope_theta,
    )?;

    workspace.attention.reshape_in_place(Shape::from_slice(&[
        1,
        config.num_heads,
        config.head_dim,
    ])?)?;
    decode_causal_attention_with_kv_cache_f32(
        &workspace.query,
        &mut workspace.key,
        &mut workspace.value,
        &mut workspace.attention,
        kv_cache,
        layer_index,
        position,
        config.num_kv_heads,
        config.head_dim,
    )?;

    workspace
        .query
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    workspace
        .key
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    workspace
        .value
        .reshape_in_place(Shape::from_slice(&[1, kv_hidden_size])?)?;
    workspace
        .attention
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;

    linear_f32_with_bias(
        &workspace.attention,
        weights.wo,
        weights.wo_bias,
        &mut workspace.normed,
    )?;
    add_f32_in_place(&mut workspace.normed, &workspace.hidden)?;

    rms_norm_f32(
        &workspace.normed,
        weights.ffn_norm,
        &mut workspace.attention,
        config.rms_norm_epsilon,
    )?;

    workspace
        .gated_product
        .reshape_in_place(Shape::from_slice(&[1, config.intermediate_size])?)?;
    workspace
        .gate_projection
        .reshape_in_place(Shape::from_slice(&[1, config.intermediate_size])?)?;
    linear_f32_with_bias(
        &workspace.attention,
        weights.w1,
        weights.w1_bias,
        &mut workspace.gated_product,
    )?;
    linear_f32_with_bias(
        &workspace.attention,
        weights.w3,
        weights.w3_bias,
        &mut workspace.gate_projection,
    )?;

    silu_f32_in_place(&mut workspace.gate_projection)?;
    mul_f32_in_place(&mut workspace.gated_product, &workspace.gate_projection)?;

    workspace
        .hidden
        .reshape_in_place(Shape::from_slice(&[1, hidden_size])?)?;
    linear_f32_with_bias(
        &workspace.gated_product,
        weights.w2,
        weights.w2_bias,
        &mut workspace.hidden,
    )?;
    add_f32_in_place(&mut workspace.hidden, &workspace.normed)?;
    Ok(())
}

#[allow(dead_code)]
fn decoder_block_decode_step_f32(
    input: &Tensor,
    weights: &ReferenceDecoderBlockWeights<'_>,
    config: ReferenceBlockConfig,
    kv_cache: &mut KvCache,
    layer_index: usize,
    position: usize,
) -> Result<Tensor> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;

    if input.shape().dims() != [1, config.num_heads * config.head_dim] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder block decode step requires input shape [1, hidden_size]",
        ));
    }

    let hidden_size = config.num_heads * config.head_dim;
    let kv_hidden_size = config.num_kv_heads * config.head_dim;

    let mut normed = zeros_2d(1, hidden_size)?;
    rms_norm_f32(
        input,
        weights.attention_norm,
        &mut normed,
        config.rms_norm_epsilon,
    )?;

    let mut query = zeros_2d(1, hidden_size)?;
    let mut key = zeros_2d(1, kv_hidden_size)?;
    let mut value = zeros_2d(1, kv_hidden_size)?;
    linear_f32_with_bias(&normed, weights.wq, weights.wq_bias, &mut query)?;
    linear_f32_with_bias(&normed, weights.wk, weights.wk_bias, &mut key)?;
    linear_f32_with_bias(&normed, weights.wv, weights.wv_bias, &mut value)?;

    let mut query_heads = reshape_heads_2d_to_3d(query, 1, config.num_heads, config.head_dim)?;
    let mut key_heads = reshape_heads_2d_to_3d(key, 1, config.num_kv_heads, config.head_dim)?;
    let mut value_heads = reshape_heads_2d_to_3d(value, 1, config.num_kv_heads, config.head_dim)?;

    rope_f32(
        &mut query_heads,
        &mut key_heads,
        position,
        config.rotary_dims,
        config.rope_theta,
    )?;

    let mut attention_heads = zeros_3d(1, config.num_heads, config.head_dim)?;
    decode_causal_attention_with_kv_cache_f32(
        &query_heads,
        &mut key_heads,
        &mut value_heads,
        &mut attention_heads,
        kv_cache,
        layer_index,
        position,
        config.num_kv_heads,
        config.head_dim,
    )?;

    let attention_merged = reshape_heads_3d_to_2d(attention_heads, 1, hidden_size)?;

    let mut attention_projected = normed;
    linear_f32_with_bias(
        &attention_merged,
        weights.wo,
        weights.wo_bias,
        &mut attention_projected,
    )?;

    let mut residual = attention_projected;
    add_f32_in_place(&mut residual, input)?;

    let mut ffn_input = attention_merged;
    rms_norm_f32(
        &residual,
        weights.ffn_norm,
        &mut ffn_input,
        config.rms_norm_epsilon,
    )?;

    let mut gated_product = zeros_2d(1, config.intermediate_size)?;
    let mut gate_projection = zeros_2d(1, config.intermediate_size)?;
    linear_f32_with_bias(&ffn_input, weights.w1, weights.w1_bias, &mut gated_product)?;
    linear_f32_with_bias(
        &ffn_input,
        weights.w3,
        weights.w3_bias,
        &mut gate_projection,
    )?;

    silu_f32_in_place(&mut gate_projection)?;
    mul_f32_in_place(&mut gated_product, &gate_projection)?;

    let mut mlp_output = ffn_input;
    linear_f32_with_bias(&gated_product, weights.w2, weights.w2_bias, &mut mlp_output)?;

    add_f32_in_place(&mut mlp_output, &residual)?;
    Ok(mlp_output)
}

fn output_weight<'a>(model: &'a DecoderOnlyModel) -> Result<&'a Tensor> {
    if model.config().tie_word_embeddings {
        get_weight(model, "tok_embeddings.weight")
    } else {
        get_weight(model, "output.weight")
    }
}

fn select_last_row_f32(input: &Tensor) -> Result<Tensor> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;

    if input.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "select_last_row_f32 requires a rank-2 tensor",
        ));
    }

    let dims = input.shape().dims();
    let rows = dims[0];
    let cols = dims[1];
    if rows == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "select_last_row_f32 requires at least one row",
        ));
    }

    let values = input.as_f32_slice()?;
    let start = (rows - 1) * cols;
    Tensor::from_f32_vec(
        Shape::from_slice(&[1, cols])?,
        values[start..start + cols].to_vec(),
    )
}

fn validate_supported_reference_model(config: &ModelConfig) -> Result<()> {
    if config.norm.kind != NormKind::RmsNorm {
        return Err(FerrisError::unsupported(
            "reference forward currently supports RMSNorm only",
        ));
    }

    if config.attention.layout != AttentionLayout::SeparateQkv {
        return Err(FerrisError::unsupported(
            "reference forward currently supports separate QKV projections only",
        ));
    }

    if !config.attention.causal {
        return Err(FerrisError::unsupported(
            "reference forward currently supports causal attention only",
        ));
    }

    if config.mlp.hidden_act != ActivationKind::Silu || !config.mlp.gated {
        return Err(FerrisError::unsupported(
            "reference forward currently supports gated SiLU MLP only",
        ));
    }

    Ok(())
}

fn block_config(config: &ModelConfig) -> ReferenceBlockConfig {
    ReferenceBlockConfig {
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim(),
        intermediate_size: config.intermediate_size,
        rms_norm_epsilon: config.norm.epsilon,
        rope_theta: config.rope.theta,
        rotary_dims: config.rope.rotary_dims,
    }
}

fn block_weights<'a>(
    model: &'a DecoderOnlyModel,
    layer: usize,
) -> Result<ReferenceDecoderBlockWeights<'a>> {
    Ok(ReferenceDecoderBlockWeights {
        attention_norm: get_weight(model, &format!("layers.{layer}.attention_norm.weight"))?,
        wq: get_weight(model, &format!("layers.{layer}.attention.wq.weight"))?,
        wq_bias: get_optional_weight(model, &format!("layers.{layer}.attention.wq.bias")),
        wk: get_weight(model, &format!("layers.{layer}.attention.wk.weight"))?,
        wk_bias: get_optional_weight(model, &format!("layers.{layer}.attention.wk.bias")),
        wv: get_weight(model, &format!("layers.{layer}.attention.wv.weight"))?,
        wv_bias: get_optional_weight(model, &format!("layers.{layer}.attention.wv.bias")),
        wo: get_weight(model, &format!("layers.{layer}.attention.wo.weight"))?,
        wo_bias: get_optional_weight(model, &format!("layers.{layer}.attention.wo.bias")),
        ffn_norm: get_weight(model, &format!("layers.{layer}.ffn_norm.weight"))?,
        w1: get_weight(model, &format!("layers.{layer}.feed_forward.w1.weight"))?,
        w1_bias: get_optional_weight(model, &format!("layers.{layer}.feed_forward.w1.bias")),
        w2: get_weight(model, &format!("layers.{layer}.feed_forward.w2.weight"))?,
        w2_bias: get_optional_weight(model, &format!("layers.{layer}.feed_forward.w2.bias")),
        w3: get_weight(model, &format!("layers.{layer}.feed_forward.w3.weight"))?,
        w3_bias: get_optional_weight(model, &format!("layers.{layer}.feed_forward.w3.bias")),
    })
}

fn get_weight<'a>(model: &'a DecoderOnlyModel, name: &str) -> Result<&'a Tensor> {
    model.weights().get(name).ok_or_else(|| {
        FerrisError::new(
            ErrorKind::MissingWeight,
            format!("missing required tensor: {name}"),
        )
    })
}

fn get_optional_weight<'a>(model: &'a DecoderOnlyModel, name: &str) -> Option<&'a Tensor> {
    model.weights().get(name)
}

fn reshape_heads_2d_to_3d(
    tensor: Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    tensor.reshape(Shape::from_slice(&[seq_len, num_heads, head_dim])?)
}

fn reshape_heads_3d_to_2d(tensor: Tensor, seq_len: usize, hidden_size: usize) -> Result<Tensor> {
    tensor.reshape(Shape::from_slice(&[seq_len, hidden_size])?)
}

fn linear_f32(input: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    weight.ensure_dtype(DType::F32)?;
    weight.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if input.shape().rank() != 2 || weight.shape().rank() != 2 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_f32 requires rank-2 tensors",
        ));
    }

    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    let out_dims = out.shape().dims();

    if input_dims[1] != weight_dims[0] || out_dims != [input_dims[0], weight_dims[1]] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_f32 expects input [m, k], weight [k, n], output [m, n]",
        ));
    }

    matmul_f32(input, weight, out)
}

fn linear_f32_with_bias(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
) -> Result<()> {
    linear_f32(input, weight, out)?;

    let Some(bias) = bias else {
        return Ok(());
    };

    bias.ensure_dtype(DType::F32)?;
    bias.ensure_contiguous()?;

    if bias.shape().rank() != 1 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_f32_with_bias requires a rank-1 bias tensor",
        ));
    }

    let out_dims = out.shape().dims();
    let out_rows = out_dims[0];
    let out_cols = out_dims[1];
    if bias.shape().dims()[0] != out_cols {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_f32_with_bias bias width must match output width",
        ));
    }

    let bias_values = bias.as_f32_slice()?;
    let out_values = out.as_f32_slice_mut()?;
    let thread_count = preferred_bias_add_thread_count(out_rows, out_cols);

    if thread_count == 1 {
        for row in 0..out_rows {
            for col in 0..out_cols {
                out_values[row * out_cols + col] += bias_values[col];
            }
        }
    } else {
        let rows_per_chunk = out_rows.div_ceil(thread_count);
        std::thread::scope(|scope| {
            for chunk in out_values.chunks_mut(rows_per_chunk * out_cols) {
                let bias_values = &bias_values;
                scope.spawn(move || {
                    for row in chunk.chunks_mut(out_cols) {
                        for (slot, bias_value) in row.iter_mut().zip(bias_values.iter().copied()) {
                            *slot += bias_value;
                        }
                    }
                });
            }
        });
    }

    Ok(())
}

fn linear_right_transposed_f32(input: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
    matmul_rhs_transposed_f32(input, weight, out)
}

fn linear_right_transposed_argmax_f32(input: &Tensor, weight: &Tensor) -> Result<TokenSample> {
    let (best_index, probability) = matmul_rhs_transposed_argmax_f32(input, weight)?;

    Ok(TokenSample {
        token_id: u32::try_from(best_index).map_err(|_| {
            FerrisError::new(ErrorKind::Runtime, "best token index does not fit into u32")
        })?,
        probability,
    })
}

fn preferred_bias_add_thread_count(rows: usize, cols: usize) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    let total_work = rows.saturating_mul(cols);
    if rows < 2 || total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(rows))
        .unwrap_or(1)
}

fn zeros_2d(rows: usize, cols: usize) -> Result<Tensor> {
    Tensor::zeros(DType::F32, Shape::from_slice(&[rows, cols])?)
}

fn zeros_3d(dim0: usize, dim1: usize, dim2: usize) -> Result<Tensor> {
    Tensor::zeros(DType::F32, Shape::from_slice(&[dim0, dim1, dim2])?)
}

#[cfg(test)]
mod tests {
    use ferrisinfer_model::{
        ActivationKind, ArchitectureKind, AttentionLayout, AttentionSpec, DecoderOnlyModel,
        MlpSpec, ModelConfig, NormKind, NormSpec, RopeScalingKind, RopeSpec, WeightMap,
    };

    use crate::kv_cache::{KvCache, KvCacheConfig};

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
    fn linear_f32_with_bias_adds_bias_per_output_row() {
        let input =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![1.0, 2.0]).unwrap();
        let weight = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let bias = Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![0.5, -1.0]).unwrap();
        let mut out = Tensor::zeros(DType::F32, Shape::from_slice(&[1, 2]).unwrap()).unwrap();

        linear_f32_with_bias(&input, &weight, Some(&bias), &mut out).unwrap();

        approx_eq_slice(&out.to_vec_f32().unwrap(), &[1.5, 1.0], 1e-6);
    }

    #[test]
    fn decoder_block_forward_f32_runs_single_reference_block() {
        let input = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let ones = Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![1.0, 1.0]).unwrap();
        let identity = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let zeros = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 2]).unwrap(),
            vec![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let weights = ReferenceDecoderBlockWeights {
            attention_norm: &ones,
            wq: &identity,
            wq_bias: None,
            wk: &identity,
            wk_bias: None,
            wv: &identity,
            wv_bias: None,
            wo: &identity,
            wo_bias: None,
            ffn_norm: &ones,
            w1: &zeros,
            w1_bias: None,
            w2: &zeros,
            w2_bias: None,
            w3: &zeros,
            w3_bias: None,
        };

        let output = decoder_block_forward_f32(
            &input,
            &weights,
            ReferenceBlockConfig {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                intermediate_size: 2,
                rms_norm_epsilon: 1e-5,
                rope_theta: 10000.0,
                rotary_dims: 0,
            },
        )
        .unwrap();

        approx_eq_slice(
            &output.to_vec_f32().unwrap(),
            &[2.4141994, 0.0, 0.27658173, 2.1376176],
            1e-5,
        );
    }

    #[test]
    fn decoder_block_forward_f32_supports_gqa() {
        let input = Tensor::from_f32_vec(
            Shape::from_slice(&[1, 4]).unwrap(),
            vec![1.4142135, 0.0, 0.0, 1.4142135],
        )
        .unwrap();
        let ones = Tensor::from_f32_vec(Shape::from_slice(&[4]).unwrap(), vec![1.0, 1.0, 1.0, 1.0])
            .unwrap();
        let q_weight = Tensor::from_f32_vec(
            Shape::from_slice(&[4, 4]).unwrap(),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let kv_weight = Tensor::from_f32_vec(
            Shape::from_slice(&[4, 2]).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let wo = Tensor::from_f32_vec(
            Shape::from_slice(&[4, 4]).unwrap(),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let zeros_mlp_in =
            Tensor::from_f32_vec(Shape::from_slice(&[4, 2]).unwrap(), vec![0.0; 8]).unwrap();
        let zeros_mlp_out =
            Tensor::from_f32_vec(Shape::from_slice(&[2, 4]).unwrap(), vec![0.0; 8]).unwrap();

        let weights = ReferenceDecoderBlockWeights {
            attention_norm: &ones,
            wq: &q_weight,
            wq_bias: None,
            wk: &kv_weight,
            wk_bias: None,
            wv: &kv_weight,
            wv_bias: None,
            wo: &wo,
            wo_bias: None,
            ffn_norm: &ones,
            w1: &zeros_mlp_in,
            w1_bias: None,
            w2: &zeros_mlp_out,
            w2_bias: None,
            w3: &zeros_mlp_in,
            w3_bias: None,
        };

        let output = decoder_block_forward_f32(
            &input,
            &weights,
            ReferenceBlockConfig {
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                intermediate_size: 2,
                rms_norm_epsilon: 0.0,
                rope_theta: 10000.0,
                rotary_dims: 0,
            },
        )
        .unwrap();

        approx_eq_slice(
            &output.to_vec_f32().unwrap(),
            &[2.828427, 0.0, 1.4142135, 1.4142135],
            1e-5,
        );
    }

    #[test]
    fn decoder_model_forward_f32_runs_complete_reference_path() {
        let model = tiny_reference_model();
        let logits = decoder_model_forward_f32(&model, &[0, 1]).unwrap();

        approx_eq_slice(
            &logits.to_vec_f32().unwrap(),
            &[1.4142112, 0.0, 0.18146895, 1.4025193],
            1e-5,
        );
    }

    #[test]
    fn decoder_model_last_token_logits_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let full_logits = decoder_model_forward_f32(&model, &[0, 1]).unwrap();
        let last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1]).unwrap();

        approx_eq_slice(
            &last_logits.to_vec_f32().unwrap(),
            &[0.18146895, 1.4025193],
            1e-5,
        );
        approx_eq_slice(
            &full_logits.to_vec_f32().unwrap()[2..4],
            &last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
    }

    #[test]
    fn decoder_model_token_logits_with_kv_cache_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();
        let mut last_logits = None;

        for (position, token_id) in [0u32, 1u32].iter().copied().enumerate() {
            let logits = decoder_model_token_logits_with_kv_cache_f32(
                &model,
                token_id,
                &mut kv_cache,
                position,
            )
            .unwrap();
            kv_cache.advance(1).unwrap();
            last_logits = Some(logits);
        }

        let last_logits = last_logits.unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1]).unwrap();

        approx_eq_slice(
            &last_logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(kv_cache.used_tokens(), 2);
    }

    #[test]
    fn decoder_model_token_logits_with_paged_kv_cache_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new_paged(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();
        let mut last_logits = None;

        for (position, token_id) in [0u32, 1u32].iter().copied().enumerate() {
            let logits = decoder_model_token_logits_with_kv_cache_f32(
                &model,
                token_id,
                &mut kv_cache,
                position,
            )
            .unwrap();
            kv_cache.advance(1).unwrap();
            last_logits = Some(logits);
        }

        let last_logits = last_logits.unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1]).unwrap();

        approx_eq_slice(
            &last_logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(kv_cache.used_tokens(), 2);
        assert_eq!(kv_cache.storage_kind(), KvCacheStorageKind::Paged);
    }

    #[test]
    fn decoder_model_token_sample_with_kv_cache_matches_logits_argmax() {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();
        let mut last_sample = None;
        let mut last_logits = None;

        for (position, token_id) in [0u32, 1u32].iter().copied().enumerate() {
            let sample = decoder_model_token_sample_with_kv_cache_f32(
                &model,
                token_id,
                &mut kv_cache,
                position,
            )
            .unwrap();
            let logits = decoder_model_token_logits_with_kv_cache_f32(
                &model,
                token_id,
                &mut kv_cache,
                position,
            )
            .unwrap();
            kv_cache.advance(1).unwrap();
            last_sample = Some(sample);
            last_logits = Some(logits);
        }

        let sample = last_sample.unwrap();
        let logits = last_logits.unwrap();
        let expected = crate::sampler::argmax_last_token(&logits).unwrap();

        assert_eq!(sample.token_id, expected.token_id);
        assert!((sample.probability - expected.probability).abs() <= 1e-6);
    }

    #[test]
    fn decoder_model_prefill_last_token_logits_with_kv_cache_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        let logits = decoder_model_prefill_last_token_logits_with_kv_cache_f32(
            &model,
            &[0, 1],
            &mut kv_cache,
        )
        .unwrap();
        kv_cache.advance(2).unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1]).unwrap();

        approx_eq_slice(
            &logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(kv_cache.used_tokens(), 2);
    }
    #[test]
    fn decoder_model_prefill_last_token_logits_with_existing_kv_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        let _ = decoder_model_token_logits_with_kv_cache_f32(&model, 0, &mut kv_cache, 0).unwrap();
        kv_cache.advance(1).unwrap();

        let logits = decoder_model_prefill_last_token_logits_with_kv_cache_f32(
            &model,
            &[1, 0],
            &mut kv_cache,
        )
        .unwrap();
        kv_cache.advance(2).unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1, 0]).unwrap();

        approx_eq_slice(
            &logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(kv_cache.used_tokens(), 3);
    }

    #[test]
    fn decoder_model_prefill_last_token_logits_with_existing_paged_kv_matches_full_forward_last_row(
    ) {
        let model = tiny_reference_model();
        let mut kv_cache = KvCache::new_paged(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let _ = decoder_model_token_logits_with_kv_cache_f32(&model, 0, &mut kv_cache, 0).unwrap();
        kv_cache.advance(1).unwrap();

        let logits = decoder_model_prefill_last_token_logits_with_kv_cache_f32(
            &model,
            &[1, 0],
            &mut kv_cache,
        )
        .unwrap();
        kv_cache.advance(2).unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1, 0]).unwrap();

        approx_eq_slice(
            &logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(kv_cache.used_tokens(), 3);
        assert_eq!(kv_cache.storage_kind(), KvCacheStorageKind::Paged);
    }

    #[test]
    fn decoder_model_token_logits_with_imported_paged_prefix_matches_full_forward_last_row() {
        let model = tiny_reference_model();
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        };
        let mut source = KvCache::new_paged(config.clone(), 2).unwrap();
        let mut dest = KvCache::new_paged(config, 2).unwrap();

        let _ =
            decoder_model_prefill_last_token_logits_with_kv_cache_f32(&model, &[0, 1], &mut source)
                .unwrap();
        source.advance(2).unwrap();
        dest.copy_prefix_from(&source, 2).unwrap();

        let logits = decoder_model_token_logits_with_kv_cache_f32(&model, 0, &mut dest, 2).unwrap();
        let full_last_logits = decoder_model_last_token_logits_f32(&model, &[0, 1, 0]).unwrap();

        approx_eq_slice(
            &logits.to_vec_f32().unwrap(),
            &full_last_logits.to_vec_f32().unwrap(),
            1e-5,
        );
        assert_eq!(dest.used_tokens(), 2);
        assert_eq!(dest.storage_kind(), KvCacheStorageKind::Paged);
    }

    #[test]
    fn decoder_model_prefill_last_token_sample_with_kv_cache_matches_logits_argmax() {
        let model = tiny_reference_model();
        let mut sample_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();
        let mut logits_cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        let sample = decoder_model_prefill_last_token_sample_with_kv_cache_f32(
            &model,
            &[0, 1],
            &mut sample_cache,
        )
        .unwrap();
        let logits = decoder_model_prefill_last_token_logits_with_kv_cache_f32(
            &model,
            &[0, 1],
            &mut logits_cache,
        )
        .unwrap();
        let expected = crate::sampler::argmax_last_token(&logits).unwrap();

        assert_eq!(sample.token_id, expected.token_id);
        assert!((sample.probability - expected.probability).abs() <= 1e-6);
    }

    fn tiny_reference_model() -> DecoderOnlyModel {
        let mut weights = WeightMap::new();
        weights.insert(
            "tok_embeddings.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.attention_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![1.0, 1.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wq.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.attention.wk.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.attention.wv.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.attention.wo.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.ffn_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![1.0, 1.0]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w1.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w2.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w3.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![0.0, 0.0, 0.0, 0.0],
            )
            .unwrap(),
        );
        weights.insert(
            "norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![1.0, 1.0]).unwrap(),
        );
        weights.insert(
            "output.weight",
            Tensor::from_f32_vec(
                Shape::from_slice(&[2, 2]).unwrap(),
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );

        DecoderOnlyModel::new(
            ModelConfig {
                architecture: ArchitectureKind::Llama,
                hidden_size: 2,
                intermediate_size: 2,
                num_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                vocab_size: 2,
                max_position_embeddings: 16,
                norm: NormSpec {
                    kind: NormKind::RmsNorm,
                    epsilon: 1e-5,
                },
                rope: RopeSpec {
                    theta: 10000.0,
                    scaling: RopeScalingKind::None,
                    scaling_factor: 1.0,
                    rotary_dims: 0,
                },
                attention: AttentionSpec {
                    layout: AttentionLayout::SeparateQkv,
                    causal: true,
                    use_qk_norm: false,
                    head_dim: 2,
                },
                mlp: MlpSpec {
                    hidden_act: ActivationKind::Silu,
                    gated: true,
                },
                tie_word_embeddings: false,
            },
            weights,
        )
        .unwrap()
    }
}
