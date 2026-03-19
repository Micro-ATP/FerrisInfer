use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Shape, Tensor};
use ferrisinfer_kernel::cpu::attention::{
    causal_self_attention_f32, embedding_gather_f32, merge_heads_f32, rope_f32, split_heads_f32,
};
use ferrisinfer_kernel::cpu::elementwise::{add_f32, mul_f32, silu_f32};
use ferrisinfer_kernel::cpu::matmul::matmul_f32;
use ferrisinfer_kernel::cpu::reduction::rms_norm_f32;
use ferrisinfer_model::{ActivationKind, AttentionLayout, DecoderOnlyModel, ModelConfig, NormKind};

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

    let mut normed = zeros_2d(seq_len, hidden_size)?;
    rms_norm_f32(
        input,
        weights.attention_norm,
        &mut normed,
        config.rms_norm_epsilon,
    )?;

    let mut query = zeros_2d(seq_len, query_hidden_size)?;
    let mut key = zeros_2d(seq_len, kv_hidden_size)?;
    let mut value = zeros_2d(seq_len, kv_hidden_size)?;
    linear_f32_with_bias(&normed, weights.wq, weights.wq_bias, &mut query)?;
    linear_f32_with_bias(&normed, weights.wk, weights.wk_bias, &mut key)?;
    linear_f32_with_bias(&normed, weights.wv, weights.wv_bias, &mut value)?;

    let mut query_heads = zeros_3d(seq_len, config.num_heads, config.head_dim)?;
    let mut key_heads = zeros_3d(seq_len, config.num_kv_heads, config.head_dim)?;
    let mut value_heads = zeros_3d(seq_len, config.num_kv_heads, config.head_dim)?;
    split_heads_f32(&query, config.num_heads, &mut query_heads)?;
    split_heads_f32(&key, config.num_kv_heads, &mut key_heads)?;
    split_heads_f32(&value, config.num_kv_heads, &mut value_heads)?;

    rope_f32(
        &mut query_heads,
        &mut key_heads,
        0,
        config.rotary_dims,
        config.rope_theta,
    )?;

    let mut attention_heads = zeros_3d(seq_len, config.num_heads, config.head_dim)?;
    causal_self_attention_f32(&query_heads, &key_heads, &value_heads, &mut attention_heads)?;

    let mut attention_merged = zeros_2d(seq_len, hidden_size)?;
    merge_heads_f32(&attention_heads, &mut attention_merged)?;

    let mut attention_projected = zeros_2d(seq_len, hidden_size)?;
    linear_f32_with_bias(
        &attention_merged,
        weights.wo,
        weights.wo_bias,
        &mut attention_projected,
    )?;

    let mut residual = zeros_2d(seq_len, hidden_size)?;
    add_f32(input, &attention_projected, &mut residual)?;

    let mut ffn_input = zeros_2d(seq_len, hidden_size)?;
    rms_norm_f32(
        &residual,
        weights.ffn_norm,
        &mut ffn_input,
        config.rms_norm_epsilon,
    )?;

    let mut up_projection = zeros_2d(seq_len, config.intermediate_size)?;
    let mut gate_projection = zeros_2d(seq_len, config.intermediate_size)?;
    linear_f32_with_bias(&ffn_input, weights.w1, weights.w1_bias, &mut up_projection)?;
    linear_f32_with_bias(
        &ffn_input,
        weights.w3,
        weights.w3_bias,
        &mut gate_projection,
    )?;

    let mut gated_activation = zeros_2d(seq_len, config.intermediate_size)?;
    silu_f32(&gate_projection, &mut gated_activation)?;

    let mut gated_product = zeros_2d(seq_len, config.intermediate_size)?;
    mul_f32(&up_projection, &gated_activation, &mut gated_product)?;

    let mut mlp_output = zeros_2d(seq_len, hidden_size)?;
    linear_f32_with_bias(&gated_product, weights.w2, weights.w2_bias, &mut mlp_output)?;

    let mut output = zeros_2d(seq_len, hidden_size)?;
    add_f32(&residual, &mlp_output, &mut output)?;
    Ok(output)
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

fn decoder_model_hidden_f32(model: &DecoderOnlyModel, token_ids: &[u32]) -> Result<Tensor> {
    validate_supported_reference_model(model.config())?;

    if token_ids.is_empty() {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "decoder_model_hidden_f32 requires at least one token",
        ));
    }

    let config = model.config();
    let embedding = get_weight(model, "tok_embeddings.weight")?;
    let mut hidden = zeros_2d(token_ids.len(), config.hidden_size)?;
    embedding_gather_f32(embedding, token_ids, &mut hidden)?;

    for layer in 0..config.num_layers {
        let weights = block_weights(model, layer)?;
        hidden = decoder_block_forward_f32(&hidden, &weights, block_config(config))?;
    }

    let mut normalized = zeros_2d(token_ids.len(), config.hidden_size)?;
    rms_norm_f32(
        &hidden,
        get_weight(model, "norm.weight")?,
        &mut normalized,
        config.norm.epsilon,
    )?;
    Ok(normalized)
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

    let values = input.to_vec_f32()?;
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
    if bias.shape().dims()[0] != out_dims[1] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_f32_with_bias bias width must match output width",
        ));
    }

    let bias_values = bias.to_vec_f32()?;
    let mut out_values = out.to_vec_f32()?;

    for row in 0..out_dims[0] {
        for col in 0..out_dims[1] {
            out_values[row * out_dims[1] + col] += bias_values[col];
        }
    }

    out.copy_from_f32_slice(&out_values)
}

fn linear_right_transposed_f32(input: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
    input.ensure_dtype(DType::F32)?;
    input.ensure_contiguous()?;
    weight.ensure_dtype(DType::F32)?;
    weight.ensure_contiguous()?;
    out.ensure_dtype(DType::F32)?;
    out.ensure_contiguous()?;

    if input.shape().rank() != 2 || weight.shape().rank() != 2 || out.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_right_transposed_f32 requires rank-2 tensors",
        ));
    }

    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    let out_dims = out.shape().dims();

    let rows = input_dims[0];
    let input_width = input_dims[1];
    let out_width = weight_dims[0];

    if weight_dims[1] != input_width || out_dims != [rows, out_width] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "linear_right_transposed_f32 expects input [m, k], weight [n, k], output [m, n]",
        ));
    }

    let input_values = input.to_vec_f32()?;
    let weight_values = weight.to_vec_f32()?;
    let mut out_values = vec![0.0f32; out.element_count()];
    let thread_count = preferred_transposed_linear_thread_count(rows, out_width, input_width);

    if thread_count == 1 {
        compute_linear_right_transposed_chunk(
            &input_values,
            &weight_values,
            &mut out_values,
            rows,
            input_width,
            out_width,
            0,
            out_width,
        );
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
                    let mut chunk_values = vec![0.0f32; rows * (col_end - col_start)];
                    compute_linear_right_transposed_chunk(
                        input_values,
                        weight_values,
                        &mut chunk_values,
                        rows,
                        input_width,
                        col_end - col_start,
                        col_start,
                        col_end,
                    );
                    (col_start, col_end, chunk_values)
                }));
            }

            for handle in handles {
                let (col_start, col_end, chunk_values) =
                    handle.join().expect("scoped thread panicked");
                let chunk_width = col_end - col_start;

                for row in 0..rows {
                    let dst_start = row * out_width + col_start;
                    let dst_end = dst_start + chunk_width;
                    let src_start = row * chunk_width;
                    let src_end = src_start + chunk_width;
                    out_values[dst_start..dst_end]
                        .copy_from_slice(&chunk_values[src_start..src_end]);
                }
            }
        });
    }

    out.copy_from_f32_slice(&out_values)
}

fn compute_linear_right_transposed_chunk(
    input_values: &[f32],
    weight_values: &[f32],
    out_values: &mut [f32],
    rows: usize,
    input_width: usize,
    out_width: usize,
    col_start: usize,
    col_end: usize,
) {
    for row in 0..rows {
        for (local_col, out_col) in (col_start..col_end).enumerate() {
            let mut sum = 0.0f32;
            for inner in 0..input_width {
                sum += input_values[row * input_width + inner]
                    * weight_values[out_col * input_width + inner];
            }
            out_values[row * out_width + local_col] = sum;
        }
    }
}

fn preferred_transposed_linear_thread_count(
    rows: usize,
    out_width: usize,
    input_width: usize,
) -> usize {
    const MIN_PARALLEL_WORK: usize = 131_072;

    if out_width < 2 {
        return 1;
    }

    let total_work = rows.saturating_mul(out_width).saturating_mul(input_width);
    if total_work < MIN_PARALLEL_WORK {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(out_width))
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
