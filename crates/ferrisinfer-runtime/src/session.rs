use std::sync::Arc;

use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result};
use ferrisinfer_io::Tokenizer;
use ferrisinfer_model::DecoderOnlyModel;

use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::reference::decoder_model_last_token_logits_f32;
use crate::sampler::{argmax_last_token, SamplerConfig, TokenSample};

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
    pub add_bos: bool,
    pub stop_token_id: Option<u32>,
}

impl GenerationRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_new_tokens: 128,
            add_bos: false,
            stop_token_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    MaxNewTokens,
    StopToken,
    SessionLimit,
    SequenceLength,
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub prompt_token_ids: Vec<u32>,
    pub generated_tokens: Vec<TokenSample>,
    pub generated_text: String,
    pub finish_reason: GenerationFinishReason,
}

impl GenerationOutput {
    pub fn generated_token_ids(&self) -> Vec<u32> {
        self.generated_tokens
            .iter()
            .map(|sample| sample.token_id)
            .collect()
    }
}

pub struct Session {
    config: SessionConfig,
    model: Arc<DecoderOnlyModel>,
    tokenizer: Arc<dyn Tokenizer>,
    kv_cache: KvCache,
    position: usize,
    token_history: Vec<u32>,
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
            token_history: Vec::new(),
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

    pub fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
        self.token_history.clear();
    }

    pub fn prefill(&mut self, prompt: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenizer.encode(prompt, true)?;
        self.prefill_tokens(&tokens)?;
        Ok(tokens)
    }

    pub fn prefill_tokens(&mut self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "prefill_tokens requires at least one token",
            ));
        }

        self.kv_cache.advance(token_ids.len())?;
        self.position = self
            .position
            .checked_add(token_ids.len())
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "session position overflow"))?;
        self.token_history.extend_from_slice(token_ids);
        Ok(())
    }

    pub fn step_reference(&mut self) -> Result<TokenSample> {
        if self.token_history.is_empty() {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "step_reference requires at least one prefetched token",
            ));
        }

        if self.position >= self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "cannot generate beyond max sequence length",
            ));
        }

        let logits = decoder_model_last_token_logits_f32(self.model.as_ref(), &self.token_history)?;
        let sample = argmax_last_token(&logits)?;
        self.prefill_tokens(&[sample.token_id])?;
        Ok(sample)
    }

    pub fn generate_reference(&mut self, request: &GenerationRequest) -> Result<GenerationOutput> {
        self.reset();
        let prompt_token_ids = self.tokenizer.encode(&request.prompt, request.add_bos)?;
        self.prefill_tokens(&prompt_token_ids)?;
        self.generate_reference_from_prefilled(
            prompt_token_ids,
            request.max_new_tokens,
            request.stop_token_id,
        )
    }

    pub fn generate_reference_from_tokens(
        &mut self,
        prompt_token_ids: &[u32],
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
    ) -> Result<GenerationOutput> {
        self.reset();
        self.prefill_tokens(prompt_token_ids)?;
        self.generate_reference_from_prefilled(
            prompt_token_ids.to_vec(),
            max_new_tokens,
            stop_token_id,
        )
    }

    fn generate_reference_from_prefilled(
        &mut self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
    ) -> Result<GenerationOutput> {
        let generation_budget = max_new_tokens.min(self.config.max_generated_tokens);
        let mut generated_tokens = Vec::with_capacity(generation_budget);
        let mut finish_reason = if generation_budget < max_new_tokens {
            GenerationFinishReason::SessionLimit
        } else {
            GenerationFinishReason::MaxNewTokens
        };

        for _ in 0..generation_budget {
            if self.position >= self.config.max_sequence_length {
                finish_reason = GenerationFinishReason::SequenceLength;
                break;
            }

            let sample = self.step_reference()?;
            generated_tokens.push(sample);

            if stop_token_id.is_some_and(|stop| stop == sample.token_id) {
                finish_reason = GenerationFinishReason::StopToken;
                break;
            }
        }

        let generated_text = self.tokenizer.decode(
            &generated_tokens
                .iter()
                .map(|sample| sample.token_id)
                .collect::<Vec<_>>(),
        )?;

        Ok(GenerationOutput {
            prompt_token_ids,
            generated_tokens,
            generated_text,
            finish_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ferrisinfer_core::{Shape, Tensor};
    use ferrisinfer_io::TokenizerKind;
    use ferrisinfer_model::{
        ActivationKind, ArchitectureKind, AttentionLayout, AttentionSpec, DecoderOnlyModel,
        MlpSpec, ModelConfig, NormKind, NormSpec, RopeScalingKind, RopeSpec, WeightMap,
    };

    use super::*;

    #[derive(Debug)]
    struct DummyTokenizer;

    impl Tokenizer for DummyTokenizer {
        fn kind(&self) -> TokenizerKind {
            TokenizerKind::BytePair
        }

        fn vocab_size(&self) -> usize {
            2
        }

        fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
            let mut tokens = Vec::new();
            if add_bos {
                tokens.push(0);
            }

            for ch in text.chars() {
                let token = match ch {
                    'a' | 'A' => 0,
                    'b' | 'B' => 1,
                    _ => {
                        return Err(FerrisError::new(
                            ErrorKind::Parse,
                            format!("unsupported dummy token '{ch}'"),
                        ));
                    }
                };
                tokens.push(token);
            }

            Ok(tokens)
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            let mut decoded = String::new();
            for &token in tokens {
                match token {
                    0 => decoded.push('A'),
                    1 => decoded.push('B'),
                    other => {
                        return Err(FerrisError::new(
                            ErrorKind::Parse,
                            format!("unsupported dummy token id {other}"),
                        ));
                    }
                }
            }
            Ok(decoded)
        }
    }

    #[test]
    fn prefill_tokens_tracks_history_and_cache_usage() {
        let mut session = test_session(SessionConfig::default());

        session.prefill_tokens(&[0]).unwrap();

        assert_eq!(session.position(), 1);
        assert_eq!(session.kv_cache().used_tokens(), 1);
        assert_eq!(session.token_history(), &[0]);
    }

    #[test]
    fn generate_reference_from_tokens_stops_on_stop_token() {
        let mut session = test_session(SessionConfig::default());

        let output = session
            .generate_reference_from_tokens(&[0], 4, Some(1))
            .unwrap();

        assert_eq!(output.prompt_token_ids, vec![0]);
        assert_eq!(output.generated_token_ids(), vec![1]);
        assert_eq!(output.generated_text, "B");
        assert_eq!(output.finish_reason, GenerationFinishReason::StopToken);
        assert_eq!(session.token_history(), &[0, 1]);
        assert_eq!(session.kv_cache().used_tokens(), 2);
    }

    #[test]
    fn generate_reference_respects_session_generation_limit() {
        let mut session = test_session(SessionConfig {
            max_sequence_length: 8,
            max_generated_tokens: 1,
            sampler: SamplerConfig::default(),
        });

        let output = session
            .generate_reference(&GenerationRequest {
                prompt: "a".to_string(),
                max_new_tokens: 3,
                add_bos: false,
                stop_token_id: None,
            })
            .unwrap();

        assert_eq!(output.prompt_token_ids, vec![0]);
        assert_eq!(output.generated_token_ids(), vec![1]);
        assert_eq!(output.generated_text, "B");
        assert_eq!(output.finish_reason, GenerationFinishReason::SessionLimit);
    }

    fn test_session(config: SessionConfig) -> Session {
        Session::new(
            Arc::new(always_one_model()),
            Arc::new(DummyTokenizer) as Arc<dyn Tokenizer>,
            config,
        )
        .unwrap()
    }

    fn always_one_model() -> DecoderOnlyModel {
        let mut weights = WeightMap::new();
        weights.insert(
            "tok_embeddings.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2, 1]).unwrap(), vec![1.0, 1.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1]).unwrap(), vec![1.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wq.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wk.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wv.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.attention.wo.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.ffn_norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1]).unwrap(), vec![1.0]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w1.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w2.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "layers.0.feed_forward.w3.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1, 1]).unwrap(), vec![0.0]).unwrap(),
        );
        weights.insert(
            "norm.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[1]).unwrap(), vec![1.0]).unwrap(),
        );
        weights.insert(
            "output.weight",
            Tensor::from_f32_vec(Shape::from_slice(&[2, 1]).unwrap(), vec![0.0, 1.0]).unwrap(),
        );

        DecoderOnlyModel::new(
            ModelConfig {
                architecture: ArchitectureKind::Qwen2,
                hidden_size: 1,
                intermediate_size: 1,
                num_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                vocab_size: 2,
                max_position_embeddings: 16,
                norm: NormSpec {
                    kind: NormKind::RmsNorm,
                    epsilon: 1e-6,
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
                    head_dim: 1,
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
