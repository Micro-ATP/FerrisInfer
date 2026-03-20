use std::sync::Arc;

use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result};
use ferrisinfer_io::Tokenizer;
use ferrisinfer_model::DecoderOnlyModel;

use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::paged_kv::PrefixHandle;
use crate::reference::{
    decoder_model_prefill_last_token_logits_with_kv_cache_f32,
    decoder_model_prefill_last_token_sample_with_kv_cache_f32,
    decoder_model_token_logits_with_kv_cache_buffered_f32,
    decoder_model_token_sample_with_kv_cache_buffered_f32, DecodeWorkspace,
};
use crate::sampler::{sample_last_token, SamplerConfig, SamplerState, TokenSample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKvCacheConfig {
    Contiguous,
    Paged { page_size: usize },
}

impl Default for SessionKvCacheConfig {
    fn default() -> Self {
        Self::Contiguous
    }
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_sequence_length: usize,
    pub max_generated_tokens: usize,
    pub sampler: SamplerConfig,
    pub kv_cache: SessionKvCacheConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 4096,
            max_generated_tokens: 512,
            sampler: SamplerConfig::default(),
            kv_cache: SessionKvCacheConfig::default(),
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
    last_sample: Option<TokenSample>,
    sampler_state: SamplerState,
    decode_workspace: DecodeWorkspace,
}

impl Session {
    pub fn new(
        model: Arc<DecoderOnlyModel>,
        tokenizer: Arc<dyn Tokenizer>,
        config: SessionConfig,
    ) -> Result<Self> {
        let sampler_seed = config.sampler.seed;
        let (kv_cache, decode_workspace) = {
            let model_config = model.config();
            let kv_config = KvCacheConfig {
                num_layers: model_config.num_layers,
                num_kv_heads: model_config.num_key_value_heads,
                head_dim: model_config.head_dim(),
                max_sequence_length: config.max_sequence_length,
                dtype: DType::F32,
            };
            let kv_cache = match config.kv_cache {
                SessionKvCacheConfig::Contiguous => KvCache::new_contiguous(kv_config)?,
                SessionKvCacheConfig::Paged { page_size } => {
                    KvCache::new_paged(kv_config, page_size)?
                }
            };
            let decode_workspace = DecodeWorkspace::new(model_config)?;
            (kv_cache, decode_workspace)
        };

        Ok(Self {
            config,
            model,
            tokenizer,
            kv_cache,
            position: 0,
            token_history: Vec::new(),
            last_sample: None,
            sampler_state: SamplerState::new(sampler_seed),
            decode_workspace,
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

    pub fn prefix_handle(&self, token_count: usize) -> Result<Option<PrefixHandle>> {
        if token_count > self.position {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "session prefix handle length {token_count} exceeds committed tokens {}",
                    self.position
                ),
            ));
        }

        self.kv_cache.prefix_handle(token_count)
    }

    pub fn copy_prefix_from_session(&mut self, source: &Session, token_count: usize) -> Result<()> {
        if token_count > source.token_history.len() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "session prefix length {token_count} exceeds source token history {}",
                    source.token_history.len()
                ),
            ));
        }

        self.reset();
        self.kv_cache
            .copy_prefix_from(&source.kv_cache, token_count)?;
        self.position = token_count;
        self.token_history
            .extend_from_slice(&source.token_history[..token_count]);

        if token_count == source.position
            && Arc::ptr_eq(&self.model, &source.model)
            && sampler_configs_match(&self.config.sampler, &source.config.sampler)
        {
            self.last_sample = source.last_sample;
            self.sampler_state = source.sampler_state;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
        self.token_history.clear();
        self.last_sample = None;
        self.sampler_state = SamplerState::new(self.config.sampler.seed);
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

        if self.position != self.kv_cache.used_tokens() {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "session position and KV cache usage diverged",
            ));
        }

        if token_ids.len() > 1 {
            self.process_reference_prefill_tokens(token_ids)?;
            return Ok(());
        }

        for &token_id in token_ids {
            self.process_reference_token(token_id)?;
        }
        Ok(())
    }

    pub fn step_reference(&mut self) -> Result<TokenSample> {
        if self.token_history.is_empty() {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "step_reference requires at least one prefetched token",
            ));
        }

        let sample = self.last_sample.ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Runtime,
                "step_reference requires a sampled token from a prefetched token",
            )
        })?;
        self.process_reference_token(sample.token_id)?;
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

    fn process_reference_token(&mut self, token_id: u32) -> Result<()> {
        if self.position >= self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "cannot process tokens beyond max sequence length",
            ));
        }

        if self.position != self.kv_cache.used_tokens() {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "session position and KV cache usage diverged",
            ));
        }

        let sample = if self.config.sampler.is_greedy() {
            decoder_model_token_sample_with_kv_cache_buffered_f32(
                self.model.as_ref(),
                token_id,
                &mut self.kv_cache,
                self.position,
                &mut self.decode_workspace,
            )?
        } else {
            decoder_model_token_logits_with_kv_cache_buffered_f32(
                self.model.as_ref(),
                token_id,
                &mut self.kv_cache,
                self.position,
                &mut self.decode_workspace,
            )?;
            sample_last_token(
                self.decode_workspace.logits(),
                &self.config.sampler,
                &mut self.sampler_state,
            )?
        };
        self.kv_cache.advance(1)?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "session position overflow"))?;
        self.token_history.push(token_id);
        self.last_sample = Some(sample);
        Ok(())
    }

    fn process_reference_prefill_tokens(&mut self, token_ids: &[u32]) -> Result<()> {
        let next_position = self
            .position
            .checked_add(token_ids.len())
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "session position overflow"))?;
        if next_position > self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "cannot process tokens beyond max sequence length",
            ));
        }

        let sample = if self.config.sampler.is_greedy() {
            decoder_model_prefill_last_token_sample_with_kv_cache_f32(
                self.model.as_ref(),
                token_ids,
                &mut self.kv_cache,
            )?
        } else {
            let logits = decoder_model_prefill_last_token_logits_with_kv_cache_f32(
                self.model.as_ref(),
                token_ids,
                &mut self.kv_cache,
            )?;
            sample_last_token(&logits, &self.config.sampler, &mut self.sampler_state)?
        };
        self.kv_cache.advance(token_ids.len())?;
        self.position = next_position;
        self.token_history.extend_from_slice(token_ids);
        self.last_sample = Some(sample);
        Ok(())
    }
}

fn sampler_configs_match(left: &SamplerConfig, right: &SamplerConfig) -> bool {
    left.temperature == right.temperature
        && left.top_k == right.top_k
        && left.top_p == right.top_p
        && left.repetition_penalty == right.repetition_penalty
        && left.seed == right.seed
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
    fn prefill_tokens_batches_initial_prompt_and_preserves_last_sample() {
        let mut session = test_session(SessionConfig::default());

        session.prefill_tokens(&[0, 1]).unwrap();
        let sample = session.step_reference().unwrap();

        assert_eq!(session.position(), 3);
        assert_eq!(session.kv_cache().used_tokens(), 3);
        assert_eq!(session.token_history(), &[0, 1, 1]);
        assert_eq!(sample.token_id, 1);
    }
    #[test]
    fn prefill_tokens_batches_appended_prompt_and_preserves_last_sample() {
        let mut session = test_session(SessionConfig::default());

        session.prefill_tokens(&[0]).unwrap();
        session.prefill_tokens(&[1, 0]).unwrap();
        let sample = session.step_reference().unwrap();

        assert_eq!(session.position(), 4);
        assert_eq!(session.kv_cache().used_tokens(), 4);
        assert_eq!(session.token_history(), &[0, 1, 0, 1]);
        assert_eq!(sample.token_id, 1);
    }

    #[test]
    fn paged_kv_cache_prefill_and_decode_remain_functional() {
        let mut session = test_session(SessionConfig {
            kv_cache: SessionKvCacheConfig::Paged { page_size: 2 },
            ..SessionConfig::default()
        });

        session.prefill_tokens(&[0, 1, 0]).unwrap();
        let sample = session.step_reference().unwrap();

        assert_eq!(
            session.kv_cache().storage_kind(),
            crate::kv_cache::KvCacheStorageKind::Paged
        );
        assert_eq!(session.position(), 4);
        assert_eq!(session.kv_cache().used_tokens(), 4);
        assert_eq!(session.token_history(), &[0, 1, 0, 1]);
        assert_eq!(sample.token_id, 1);
    }

    #[test]
    fn paged_session_exposes_prefix_handle() {
        let mut session = test_session(SessionConfig {
            kv_cache: SessionKvCacheConfig::Paged { page_size: 2 },
            ..SessionConfig::default()
        });

        session.prefill_tokens(&[0, 1, 0]).unwrap();

        let handle = session.prefix_handle(3).unwrap().unwrap();
        assert_eq!(handle.token_count(), 3);
        assert_eq!(handle.page_size(), 2);
        assert_eq!(handle.layer_block_table(0).unwrap().len(), 2);
    }

    #[test]
    fn copy_prefix_from_session_preserves_decode_state_for_exact_prefix() {
        let config = SessionConfig {
            kv_cache: SessionKvCacheConfig::Paged { page_size: 2 },
            ..SessionConfig::default()
        };
        let model = Arc::new(always_one_model());
        let tokenizer = Arc::new(DummyTokenizer) as Arc<dyn Tokenizer>;
        let mut source =
            Session::new(Arc::clone(&model), Arc::clone(&tokenizer), config.clone()).unwrap();
        let mut dest = Session::new(model, tokenizer, config).unwrap();

        source.prefill_tokens(&[0, 1, 0]).unwrap();
        dest.copy_prefix_from_session(&source, 3).unwrap();
        let sample = dest.step_reference().unwrap();

        assert_eq!(dest.position(), 4);
        assert_eq!(dest.kv_cache().used_tokens(), 4);
        assert_eq!(dest.token_history(), &[0, 1, 0, 1]);
        assert_eq!(sample.token_id, 1);
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
            ..SessionConfig::default()
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
