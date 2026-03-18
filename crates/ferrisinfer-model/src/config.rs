use ferrisinfer_core::{ErrorKind, FerrisError, Result};

use crate::spec::{ArchitectureKind, AttentionSpec, MlpSpec, NormSpec, RopeSpec};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ArchitectureKind,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub norm: NormSpec,
    pub rope: RopeSpec,
    pub attention: AttentionSpec,
    pub mlp: MlpSpec,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "model dimensions must be greater than zero",
            ));
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }

        if self.attention.head_dim != self.head_dim() {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "attention.head_dim must match hidden_size / num_attention_heads",
            ));
        }

        if self.rope.rotary_dims > self.head_dim() {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "rope.rotary_dims cannot exceed attention head dimension",
            ));
        }

        Ok(())
    }
}
