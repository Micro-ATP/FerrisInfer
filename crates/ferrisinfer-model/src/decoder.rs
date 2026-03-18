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

        Ok(())
    }
}
