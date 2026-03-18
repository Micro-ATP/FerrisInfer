use std::sync::Arc;

use ferrisinfer_core::{DType, Result};
use ferrisinfer_io::Tokenizer;
use ferrisinfer_model::DecoderOnlyModel;

use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::sampler::SamplerConfig;

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_sequence_length: usize,
    pub max_generated_tokens: usize,
    pub sampler: SamplerConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 4096,
            max_generated_tokens: 512,
            sampler: SamplerConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
}

pub struct Session {
    config: SessionConfig,
    model: Arc<DecoderOnlyModel>,
    tokenizer: Arc<dyn Tokenizer>,
    kv_cache: KvCache,
    position: usize,
}

impl Session {
    pub fn new(
        model: Arc<DecoderOnlyModel>,
        tokenizer: Arc<dyn Tokenizer>,
        config: SessionConfig,
    ) -> Result<Self> {
        let model_config = model.config();
        let kv_cache = KvCache::new(KvCacheConfig {
            num_layers: model_config.num_layers,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim(),
            max_sequence_length: config.max_sequence_length,
            dtype: DType::F32,
        })?;

        Ok(Self {
            config,
            model,
            tokenizer,
            kv_cache,
            position: 0,
        })
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn model(&self) -> &DecoderOnlyModel {
        self.model.as_ref()
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn prefill(&mut self, prompt: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenizer.encode(prompt, true)?;
        self.kv_cache.advance(tokens.len())?;
        self.position += tokens.len();
        Ok(tokens)
    }
}
