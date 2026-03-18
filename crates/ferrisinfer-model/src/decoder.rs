use ferrisinfer_core::{ErrorKind, FerrisError, Result};

use crate::config::ModelConfig;
use crate::weights::WeightMap;

#[derive(Debug)]
pub struct DecoderOnlyModel {
    config: ModelConfig,
    weights: WeightMap,
}

impl DecoderOnlyModel {
    pub fn new(config: ModelConfig, weights: WeightMap) -> Result<Self> {
        config.validate()?;

        let model = Self { config, weights };
        model.validate_weights()?;

        Ok(model)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn weights(&self) -> &WeightMap {
        &self.weights
    }

    pub fn required_weight_keys(&self) -> Vec<String> {
        let mut keys = vec!["tok_embeddings.weight".to_string()];

        for layer in 0..self.config.num_layers {
            keys.push(format!("layers.{layer}.attention_norm.weight"));
            keys.push(format!("layers.{layer}.attention.wq.weight"));
            keys.push(format!("layers.{layer}.attention.wk.weight"));
            keys.push(format!("layers.{layer}.attention.wv.weight"));
            keys.push(format!("layers.{layer}.attention.wo.weight"));
            keys.push(format!("layers.{layer}.ffn_norm.weight"));
            keys.push(format!("layers.{layer}.feed_forward.w1.weight"));
            keys.push(format!("layers.{layer}.feed_forward.w2.weight"));
            keys.push(format!("layers.{layer}.feed_forward.w3.weight"));
        }

        keys.push("norm.weight".to_string());

        if !self.config.tie_word_embeddings {
            keys.push("output.weight".to_string());
        }

        keys
    }

    pub fn validate_weights(&self) -> Result<()> {
        for key in self.required_weight_keys() {
            if !self.weights.contains(&key) {
                return Err(FerrisError::new(
                    ErrorKind::MissingWeight,
                    format!("missing required tensor: {key}"),
                ));
            }
        }

        self.validate_weight_shapes()
    }

    fn validate_weight_shapes(&self) -> Result<()> {
        let hidden = self.config.hidden_size;
        let intermediate = self.config.intermediate_size;
        let kv_hidden = self.config.num_key_value_heads * self.config.head_dim();

        self.ensure_weight_shape("tok_embeddings.weight", &[self.config.vocab_size, hidden])?;

        for layer in 0..self.config.num_layers {
            self.ensure_weight_shape(&format!("layers.{layer}.attention_norm.weight"), &[hidden])?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.attention.wq.weight"),
                &[hidden, hidden],
            )?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.attention.wk.weight"),
                &[hidden, kv_hidden],
            )?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.attention.wv.weight"),
                &[hidden, kv_hidden],
            )?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.attention.wo.weight"),
                &[hidden, hidden],
            )?;
            self.ensure_weight_shape(&format!("layers.{layer}.ffn_norm.weight"), &[hidden])?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.feed_forward.w1.weight"),
                &[hidden, intermediate],
            )?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.feed_forward.w2.weight"),
                &[intermediate, hidden],
            )?;
            self.ensure_weight_shape(
                &format!("layers.{layer}.feed_forward.w3.weight"),
                &[hidden, intermediate],
            )?;
        }

        self.ensure_weight_shape("norm.weight", &[hidden])?;

        if !self.config.tie_word_embeddings {
            self.ensure_weight_shape("output.weight", &[self.config.vocab_size, hidden])?;
        }

        Ok(())
    }

    fn ensure_weight_shape(&self, name: &str, expected: &[usize]) -> Result<()> {
        let tensor = self.weights.get(name).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::MissingWeight,
                format!("missing required tensor: {name}"),
            )
        })?;

        if tensor.shape().dims() != expected {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor '{name}' expected shape {:?} but found {:?}",
                    expected,
                    tensor.shape().dims()
                ),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ferrisinfer_core::{Shape, Tensor};

    use crate::{
        ActivationKind, ArchitectureKind, AttentionLayout, AttentionSpec, MlpSpec, ModelConfig,
        NormKind, NormSpec, RopeScalingKind, RopeSpec, WeightMap,
    };

    use super::*;

    #[test]
    fn decoder_model_rejects_invalid_weight_shape() {
        let config = tiny_config();
        let mut weights = tiny_weights();
        weights.insert(
            "layers.0.attention.wk.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2, 2]).unwrap(), vec![0.0; 4]).unwrap(),
        );

        let error = DecoderOnlyModel::new(config, weights).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidShape);
    }

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: ArchitectureKind::Qwen2,
            hidden_size: 4,
            intermediate_size: 8,
            num_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            vocab_size: 16,
            max_position_embeddings: 128,
            norm: NormSpec {
                kind: NormKind::RmsNorm,
                epsilon: 1e-6,
            },
            rope: RopeSpec {
                theta: 1_000_000.0,
                scaling: RopeScalingKind::None,
                scaling_factor: 1.0,
                rotary_dims: 2,
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
            tie_word_embeddings: true,
        }
    }

    fn tiny_weights() -> WeightMap {
        let mut weights = WeightMap::new();
        weights.insert(
            "tok_embeddings.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[16, 4]).unwrap(), vec![0.0; 64]).unwrap(),
        );
        weights.insert(
            "layers.0.attention_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4]).unwrap(), vec![0.0; 4]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wq.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 4]).unwrap(), vec![0.0; 16]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wk.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 2]).unwrap(), vec![0.0; 8]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wv.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 2]).unwrap(), vec![0.0; 8]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wo.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 4]).unwrap(), vec![0.0; 16]).unwrap(),
        );
        weights.insert(
            "layers.0.ffn_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4]).unwrap(), vec![0.0; 4]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w1.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 8]).unwrap(), vec![0.0; 32]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w2.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[8, 4]).unwrap(), vec![0.0; 32]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w3.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4, 8]).unwrap(), vec![0.0; 32]).unwrap(),
        );
        weights.insert(
            "norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[4]).unwrap(), vec![0.0; 4]).unwrap(),
        );
        weights
    }
}
