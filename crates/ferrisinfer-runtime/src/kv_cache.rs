use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result};

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_sequence_length: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct KvCache {
    config: KvCacheConfig,
    used_tokens: usize,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        if config.num_layers == 0
            || config.num_kv_heads == 0
            || config.head_dim == 0
            || config.max_sequence_length == 0
        {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "KV cache dimensions must be greater than zero",
            ));
        }

        Ok(Self {
            config,
            used_tokens: 0,
        })
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    pub fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.config
            .max_sequence_length
            .saturating_sub(self.used_tokens)
    }

    pub fn advance(&mut self, tokens: usize) -> Result<()> {
        let next = self
            .used_tokens
            .checked_add(tokens)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache position overflow"))?;

        if next > self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "KV cache capacity exceeded",
            ));
        }

        self.used_tokens = next;
        Ok(())
    }
}
