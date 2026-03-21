use std::sync::Arc;

use ferrisinfer_core::{ErrorKind, FerrisError, Result};
use ferrisinfer_io::Tokenizer;
use ferrisinfer_model::DecoderOnlyModel;

use crate::block_manager::{ManagedBlockId, PrefixBlockManager, SharedPrefixAssignment};
use crate::prefix_index::{PrefixIndex, PrefixIndexConfig, PrefixIndexEntry};
use crate::sampler::TokenSample;
use crate::sequence::{
    RequestId, SchedulerBatchKind, SchedulerTick, SequenceFinishReason, SequenceId, SequencePhase,
    SequenceState,
};
use crate::session::{Session, SessionConfig};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_prefill_chunk_tokens: usize,
    pub max_prefill_batch_tokens: usize,
    pub max_consecutive_prefill_ticks: usize,
    pub min_prefix_share_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            max_prefill_chunk_tokens: 256,
            max_prefill_batch_tokens: 1024,
            max_consecutive_prefill_ticks: 2,
            min_prefix_share_tokens: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequenceSubmitRequest {
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub stop_token_id: Option<u32>,
}

impl SequenceSubmitRequest {
    pub fn new(prompt_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            max_new_tokens: 128,
            stop_token_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequenceExecutionUpdate {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub batch_kind: SchedulerBatchKind,
    pub reused_prompt_tokens: usize,
    pub appended_prompt_tokens: usize,
    pub generated_token: Option<TokenSample>,
    pub phase: SequencePhase,
    pub finish_reason: Option<SequenceFinishReason>,
    pub released_prefix_blocks: Vec<ManagedBlockId>,
}

#[derive(Debug, Clone)]
pub struct SchedulerExecutionReport {
    pub tick: SchedulerTick,
    pub updates: Vec<SequenceExecutionUpdate>,
}

struct SequenceEntry {
    state: SequenceState,
    session: Option<Session>,
}

impl SequenceEntry {
    fn session(&self) -> Result<&Session> {
        self.session.as_ref().ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Runtime,
                format!(
                    "scheduler sequence {} no longer retains an active session",
                    self.state.sequence_id().raw()
                ),
            )
        })
    }

    fn session_mut(&mut self) -> Result<&mut Session> {
        let sequence_id = self.state.sequence_id();
        self.session.as_mut().ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Runtime,
                format!(
                    "scheduler sequence {} no longer retains an active session",
                    sequence_id.raw()
                ),
            )
        })
    }

    fn release_session(&mut self) {
        self.session = None;
    }
}

pub struct ReferenceScheduler {
    model: Arc<DecoderOnlyModel>,
    tokenizer: Arc<dyn Tokenizer>,
    session_config: SessionConfig,
    config: SchedulerConfig,
    next_request_id: u64,
    next_sequence_id: u64,
    consecutive_prefill_ticks: usize,
    prefix_index: PrefixIndex,
    prefix_block_manager: PrefixBlockManager,
    sequences: Vec<SequenceEntry>,
}

impl ReferenceScheduler {
    pub fn new(
        model: Arc<DecoderOnlyModel>,
        tokenizer: Arc<dyn Tokenizer>,
        session_config: SessionConfig,
        config: SchedulerConfig,
    ) -> Result<Self> {
        if config.max_batch_size == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "scheduler max_batch_size must be greater than zero",
            ));
        }
        if config.max_prefill_chunk_tokens == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "scheduler max_prefill_chunk_tokens must be greater than zero",
            ));
        }
        if config.max_prefill_batch_tokens == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "scheduler max_prefill_batch_tokens must be greater than zero",
            ));
        }
        if config.max_consecutive_prefill_ticks == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "scheduler max_consecutive_prefill_ticks must be greater than zero",
            ));
        }

        let prefix_index = PrefixIndex::new(PrefixIndexConfig {
            min_prefix_share_tokens: config.min_prefix_share_tokens,
        });

        Ok(Self {
            model,
            tokenizer,
            session_config,
            config,
            next_request_id: 0,
            next_sequence_id: 0,
            consecutive_prefill_ticks: 0,
            prefix_index,
            prefix_block_manager: PrefixBlockManager::new(),
            sequences: Vec::new(),
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn session_config(&self) -> &SessionConfig {
        &self.session_config
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn resident_session_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|entry| entry.session.is_some())
            .count()
    }

    pub fn has_pending(&self) -> bool {
        self.sequences
            .iter()
            .any(|entry| !entry.state.is_finished())
    }

    pub fn sequence_state(&self, sequence_id: SequenceId) -> Option<&SequenceState> {
        self.sequences
            .iter()
            .find(|entry| entry.state.sequence_id() == sequence_id)
            .map(|entry| &entry.state)
    }

    pub fn sequence_states(&self) -> impl Iterator<Item = &SequenceState> {
        self.sequences.iter().map(|entry| &entry.state)
    }

    pub fn submit(&mut self, request: SequenceSubmitRequest) -> Result<SequenceId> {
        if request.prompt_token_ids.is_empty() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "scheduler requests require at least one prompt token",
            ));
        }

        if request.prompt_token_ids.len() > self.session_config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "scheduler request prompt exceeds session max sequence length",
            ));
        }

        let request_id = self.next_request_id();
        let sequence_id = self.next_sequence_id();
        let generation_budget = request
            .max_new_tokens
            .min(self.session_config.max_generated_tokens);
        let state = SequenceState::with_generation_budget(
            request_id,
            sequence_id,
            request.prompt_token_ids,
            request.max_new_tokens,
            generation_budget,
            request.stop_token_id,
        );
        let session = Session::new(
            Arc::clone(&self.model),
            Arc::clone(&self.tokenizer),
            self.session_config.clone(),
        )?;

        self.sequences.push(SequenceEntry {
            state,
            session: Some(session),
        });
        self.prefix_block_manager
            .update_shared_prefix(sequence_id, None);
        self.refresh_sequence_prefix_metadata(sequence_id)?;
        self.rebuild_prefix_index();
        Ok(sequence_id)
    }

    pub fn plan_next_tick(&self) -> Option<SchedulerTick> {
        let prefill_ids = self.collect_prefill_sequence_ids();
        let decode_ids = self.collect_decode_sequence_ids();

        if prefill_ids.is_empty() {
            return (!decode_ids.is_empty())
                .then(|| SchedulerTick::new(SchedulerBatchKind::Decode, decode_ids));
        }
        if decode_ids.is_empty() {
            return Some(SchedulerTick::new(SchedulerBatchKind::Prefill, prefill_ids));
        }
        if self.should_schedule_decode(prefill_ids.len(), decode_ids.len()) {
            return Some(SchedulerTick::new(SchedulerBatchKind::Decode, decode_ids));
        }

        Some(SchedulerTick::new(SchedulerBatchKind::Prefill, prefill_ids))
    }

    fn collect_prefill_sequence_ids(&self) -> Vec<SequenceId> {
        let mut sequence_ids = Vec::new();
        let mut total_prompt_tokens = 0usize;

        for entry in self
            .sequences
            .iter()
            .filter(|entry| entry.state.phase() == SequencePhase::Prefill)
        {
            let chunk_tokens = entry
                .state
                .next_prefill_chunk_len(self.config.max_prefill_chunk_tokens);
            if chunk_tokens == 0 {
                continue;
            }
            if !sequence_ids.is_empty()
                && total_prompt_tokens.saturating_add(chunk_tokens)
                    > self.config.max_prefill_batch_tokens
            {
                break;
            }

            total_prompt_tokens = total_prompt_tokens.saturating_add(chunk_tokens);
            sequence_ids.push(entry.state.sequence_id());
            if sequence_ids.len() >= self.config.max_batch_size
                || total_prompt_tokens >= self.config.max_prefill_batch_tokens
            {
                break;
            }
        }

        sequence_ids
    }

    fn collect_decode_sequence_ids(&self) -> Vec<SequenceId> {
        self.sequences
            .iter()
            .filter(|entry| entry.state.phase() == SequencePhase::Decode)
            .take(self.config.max_batch_size)
            .map(|entry| entry.state.sequence_id())
            .collect::<Vec<_>>()
    }

    fn should_schedule_decode(&self, prefill_batch_size: usize, decode_batch_size: usize) -> bool {
        if decode_batch_size == 0 {
            return false;
        }
        if prefill_batch_size == 0 {
            return true;
        }
        if self.consecutive_prefill_ticks >= self.config.max_consecutive_prefill_ticks {
            return true;
        }

        self.consecutive_prefill_ticks > 0
            && decode_batch_size == self.config.max_batch_size
            && prefill_batch_size < self.config.max_batch_size
    }

    fn best_prefix_share_candidate(
        &self,
        sequence_id: SequenceId,
    ) -> Option<(usize, usize, usize)> {
        if self.config.min_prefix_share_tokens == 0 {
            return None;
        }

        let target_index = self.find_entry_index(sequence_id).ok()?;
        let target = self.sequences.get(target_index)?;
        if target.state.cached_prompt_tokens() != 0
            || target.state.phase() != SequencePhase::Prefill
        {
            return None;
        }

        let candidate = self
            .prefix_index
            .best_match(sequence_id, target.state.prompt_tokens())?;
        let source_index = self.find_entry_index(candidate.sequence_id).ok()?;
        Some((target_index, source_index, candidate.shared_prefix_tokens))
    }

    fn maybe_share_prefix(&mut self, sequence_id: SequenceId) -> Result<usize> {
        self.rebuild_prefix_index();
        let Some((target_index, source_index, shared_prefix_tokens)) =
            self.best_prefix_share_candidate(sequence_id)
        else {
            return Ok(0);
        };

        if shared_prefix_tokens == 0 {
            return Ok(0);
        }

        let shared_prefix_assignment =
            self.capture_shared_prefix_assignment(source_index, shared_prefix_tokens)?;

        if target_index < source_index {
            let (left, right) = self.sequences.split_at_mut(source_index);
            let target = &mut left[target_index];
            let source = &right[0];
            target
                .session_mut()?
                .copy_prefix_from_session(source.session()?, shared_prefix_tokens)?;
            target.state.record_prefill(shared_prefix_tokens);
        } else {
            let (left, right) = self.sequences.split_at_mut(target_index);
            let source = &left[source_index];
            let target = &mut right[0];
            target
                .session_mut()?
                .copy_prefix_from_session(source.session()?, shared_prefix_tokens)?;
            target.state.record_prefill(shared_prefix_tokens);
        }

        self.prefix_block_manager
            .update_shared_prefix(sequence_id, shared_prefix_assignment);
        self.refresh_sequence_prefix_metadata(sequence_id)?;
        self.rebuild_prefix_index();
        Ok(shared_prefix_tokens)
    }

    pub fn execute_next_tick(&mut self) -> Result<Option<SchedulerExecutionReport>> {
        let Some(tick) = self.plan_next_tick() else {
            return Ok(None);
        };

        let mut updates = Vec::with_capacity(tick.sequence_ids().len());
        for sequence_id in tick.sequence_ids().iter().copied() {
            let update = match tick.batch_kind() {
                SchedulerBatchKind::Prefill => self.execute_prefill(sequence_id)?,
                SchedulerBatchKind::Decode => self.execute_decode(sequence_id)?,
            };
            self.refresh_sequence_prefix_metadata(sequence_id)?;
            self.rebuild_prefix_index();
            updates.push(update);
        }

        self.consecutive_prefill_ticks = match tick.batch_kind() {
            SchedulerBatchKind::Prefill => self.consecutive_prefill_ticks.saturating_add(1),
            SchedulerBatchKind::Decode => 0,
        };

        Ok(Some(SchedulerExecutionReport { tick, updates }))
    }

    fn execute_prefill(&mut self, sequence_id: SequenceId) -> Result<SequenceExecutionUpdate> {
        let max_prefill_chunk_tokens = self.config.max_prefill_chunk_tokens;
        let reused_prompt_tokens = self.maybe_share_prefix(sequence_id)?;
        let (request_id, appended_prompt_tokens, phase, finish_reason) = {
            let entry = self.find_entry_mut(sequence_id)?;
            let appended_prompt_tokens =
                entry.state.next_prefill_chunk_len(max_prefill_chunk_tokens);
            if appended_prompt_tokens > 0 {
                let prompt_chunk = entry
                    .state
                    .next_prefill_chunk(max_prefill_chunk_tokens)
                    .to_vec();
                entry.session_mut()?.prefill_tokens(&prompt_chunk)?;
                entry.state.record_prefill(appended_prompt_tokens);
            }

            let finish_reason = if entry.state.generation_budget_reached() {
                Some(entry.state.generation_budget_finish_reason())
            } else if entry.session()?.position() >= entry.session()?.config().max_sequence_length {
                Some(SequenceFinishReason::SequenceLength)
            } else {
                None
            };

            if let Some(reason) = finish_reason {
                entry.state.finish(reason);
            }

            (
                entry.state.request_id(),
                appended_prompt_tokens,
                entry.state.phase(),
                finish_reason,
            )
        };
        let released_prefix_blocks = if finish_reason.is_some() {
            self.release_sequence_resources(sequence_id)?
        } else {
            Vec::new()
        };

        Ok(SequenceExecutionUpdate {
            request_id,
            sequence_id,
            batch_kind: SchedulerBatchKind::Prefill,
            reused_prompt_tokens,
            appended_prompt_tokens,
            generated_token: None,
            phase,
            finish_reason,
            released_prefix_blocks,
        })
    }

    fn execute_decode(&mut self, sequence_id: SequenceId) -> Result<SequenceExecutionUpdate> {
        let (request_id, generated_token, phase, finish_reason) = {
            let entry = self.find_entry_mut(sequence_id)?;

            let finish_reason =
                if entry.session()?.position() >= entry.session()?.config().max_sequence_length {
                    Some(SequenceFinishReason::SequenceLength)
                } else {
                    None
                };
            if let Some(reason) = finish_reason {
                entry.state.finish(reason);
                (
                    entry.state.request_id(),
                    None,
                    entry.state.phase(),
                    Some(reason),
                )
            } else {
                let generated = entry.session_mut()?.step_reference()?;
                entry.state.record_generated_token(generated.token_id);

                let finish_reason = if entry
                    .state
                    .stop_token_id()
                    .is_some_and(|stop| stop == generated.token_id)
                {
                    Some(SequenceFinishReason::StopToken)
                } else if entry.state.generation_budget_reached() {
                    Some(entry.state.generation_budget_finish_reason())
                } else if entry.session()?.position()
                    >= entry.session()?.config().max_sequence_length
                {
                    Some(SequenceFinishReason::SequenceLength)
                } else {
                    None
                };

                if let Some(reason) = finish_reason {
                    entry.state.finish(reason);
                }

                (
                    entry.state.request_id(),
                    Some(generated),
                    entry.state.phase(),
                    finish_reason,
                )
            }
        };
        let released_prefix_blocks = if finish_reason.is_some() {
            self.release_sequence_resources(sequence_id)?
        } else {
            Vec::new()
        };

        Ok(SequenceExecutionUpdate {
            request_id,
            sequence_id,
            batch_kind: SchedulerBatchKind::Decode,
            reused_prompt_tokens: 0,
            appended_prompt_tokens: 0,
            generated_token,
            phase,
            finish_reason,
            released_prefix_blocks,
        })
    }

    fn capture_shared_prefix_assignment(
        &self,
        source_index: usize,
        token_count: usize,
    ) -> Result<Option<SharedPrefixAssignment>> {
        let source = self.sequences.get(source_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Runtime,
                format!("scheduler source sequence index {source_index} is missing"),
            )
        })?;

        Ok(source
            .session()?
            .prefix_handle(token_count)?
            .map(|handle| SharedPrefixAssignment::new(source.state.sequence_id(), handle)))
    }

    fn rebuild_prefix_index(&mut self) {
        self.prefix_index.clear();
        for entry in &self.sequences {
            if entry.state.is_finished() {
                continue;
            }

            self.prefix_index.upsert(PrefixIndexEntry {
                sequence_id: entry.state.sequence_id(),
                prompt_tokens: entry.state.prompt_tokens().to_vec(),
                cached_prompt_tokens: entry.state.cached_prompt_tokens(),
                session_position: entry.state.cached_tokens(),
            });
        }
    }

    fn refresh_sequence_prefix_metadata(&mut self, sequence_id: SequenceId) -> Result<()> {
        let prefix_handle = {
            let entry = self.find_entry(sequence_id)?;
            if entry.state.is_finished() {
                None
            } else {
                entry
                    .session()?
                    .prefix_handle(entry.state.cached_prompt_tokens())?
            }
        };
        self.prefix_block_manager
            .update_owned_prefix(sequence_id, prefix_handle);
        Ok(())
    }

    fn release_sequence_resources(
        &mut self,
        sequence_id: SequenceId,
    ) -> Result<Vec<ManagedBlockId>> {
        {
            let entry = self.find_entry_mut(sequence_id)?;
            entry.release_session();
        }
        Ok(self.prefix_block_manager.remove_sequence(sequence_id))
    }

    fn find_entry(&self, sequence_id: SequenceId) -> Result<&SequenceEntry> {
        self.sequences
            .iter()
            .find(|entry| entry.state.sequence_id() == sequence_id)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("scheduler sequence {} is missing", sequence_id.raw()),
                )
            })
    }

    fn find_entry_mut(&mut self, sequence_id: SequenceId) -> Result<&mut SequenceEntry> {
        self.sequences
            .iter_mut()
            .find(|entry| entry.state.sequence_id() == sequence_id)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("scheduler sequence {} is missing", sequence_id.raw()),
                )
            })
    }

    fn find_entry_index(&self, sequence_id: SequenceId) -> Result<usize> {
        self.sequences
            .iter()
            .position(|entry| entry.state.sequence_id() == sequence_id)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("scheduler sequence {} is missing", sequence_id.raw()),
                )
            })
    }

    fn next_request_id(&mut self) -> RequestId {
        let id = RequestId::new(self.next_request_id);
        self.next_request_id = self.next_request_id.saturating_add(1);
        id
    }

    fn next_sequence_id(&mut self) -> SequenceId {
        let id = SequenceId::new(self.next_sequence_id);
        self.next_sequence_id = self.next_sequence_id.saturating_add(1);
        id
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ferrisinfer_core::{Shape, Tensor};
    use ferrisinfer_io::{Tokenizer, TokenizerKind};
    use ferrisinfer_model::{
        ActivationKind, ArchitectureKind, AttentionLayout, AttentionSpec, DecoderOnlyModel,
        MlpSpec, ModelConfig, NormKind, NormSpec, RopeScalingKind, RopeSpec, WeightMap,
    };

    use super::*;
    use crate::block_manager::ManagedBlockId;
    use crate::paged_kv::KvBlockId;
    use crate::sampler::SamplerConfig;

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
    fn reference_scheduler_prefills_then_decodes_single_sequence() {
        let mut scheduler = test_scheduler(SessionConfig::default(), SchedulerConfig::default());
        let sequence_id = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let prefill_tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(prefill_tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(prefill_tick.sequence_ids(), &[sequence_id]);

        let prefill_report = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(
            prefill_report.tick.batch_kind(),
            SchedulerBatchKind::Prefill
        );
        assert_eq!(prefill_report.updates[0].reused_prompt_tokens, 0);
        assert_eq!(prefill_report.updates[0].appended_prompt_tokens, 2);
        assert_eq!(prefill_report.updates[0].generated_token, None);
        assert_eq!(prefill_report.updates[0].phase, SequencePhase::Decode);
        assert_eq!(prefill_report.updates[0].finish_reason, None);

        let decode_report = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(decode_report.tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(decode_report.updates[0].sequence_id, sequence_id);
        assert_eq!(
            decode_report.updates[0].generated_token.unwrap().token_id,
            1
        );
        assert_eq!(
            decode_report.updates[0].finish_reason,
            Some(SequenceFinishReason::MaxNewTokens)
        );

        let state = scheduler.sequence_state(sequence_id).unwrap();
        assert_eq!(state.phase(), SequencePhase::Finished);
        assert_eq!(state.generated_tokens(), &[1]);
        assert_eq!(
            state.finish_reason(),
            Some(SequenceFinishReason::MaxNewTokens)
        );
        assert!(scheduler.execute_next_tick().unwrap().is_none());
    }

    #[test]
    fn reference_scheduler_prefill_has_priority_over_decode() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 1,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0],
                max_new_tokens: 2,
                stop_token_id: None,
            })
            .unwrap();

        scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(
            scheduler.plan_next_tick().unwrap().batch_kind(),
            SchedulerBatchKind::Decode
        );

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(tick.sequence_ids(), &[second]);

        let first_state = scheduler.sequence_state(first).unwrap();
        assert_eq!(first_state.phase(), SequencePhase::Decode);
    }

    #[test]
    fn reference_scheduler_chunked_prefill_spreads_prompt_across_ticks() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 1,
                max_prefill_chunk_tokens: 2,
                ..SchedulerConfig::default()
            },
        );
        let sequence_id = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let first = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(first.tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(first.updates[0].appended_prompt_tokens, 2);
        assert_eq!(first.updates[0].phase, SequencePhase::Prefill);

        let second = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(second.tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(second.updates[0].appended_prompt_tokens, 2);
        assert_eq!(second.updates[0].phase, SequencePhase::Prefill);

        let third = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(third.tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(third.updates[0].appended_prompt_tokens, 1);
        assert_eq!(third.updates[0].phase, SequencePhase::Decode);

        let decode = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(decode.tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(decode.updates[0].generated_token.unwrap().token_id, 1);
        assert_eq!(
            scheduler
                .sequence_state(sequence_id)
                .unwrap()
                .finish_reason(),
            Some(SequenceFinishReason::MaxNewTokens)
        );
    }

    #[test]
    fn reference_scheduler_reuses_prompt_prefix_from_existing_sequence() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                kv_cache: crate::session::SessionKvCacheConfig::Paged { page_size: 2 },
                ..SessionConfig::default()
            },
            SchedulerConfig {
                max_batch_size: 1,
                min_prefix_share_tokens: 2,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        let report = scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(report.tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(report.tick.sequence_ids(), &[second]);
        assert_eq!(report.updates[0].sequence_id, second);
        assert_eq!(report.updates[0].reused_prompt_tokens, 3);
        assert_eq!(report.updates[0].appended_prompt_tokens, 1);
        assert_eq!(report.updates[0].phase, SequencePhase::Decode);
        assert_eq!(
            scheduler.sequence_state(first).unwrap().phase(),
            SequencePhase::Decode
        );
    }

    #[test]
    fn reference_scheduler_tracks_shared_prefix_block_metadata() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                kv_cache: crate::session::SessionKvCacheConfig::Paged { page_size: 2 },
                ..SessionConfig::default()
            },
            SchedulerConfig {
                max_batch_size: 1,
                min_prefix_share_tokens: 2,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(
            scheduler
                .prefix_block_manager
                .block_ref_count(ManagedBlockId::new(first, 0, KvBlockId::new(0))),
            2
        );
        assert_eq!(
            scheduler
                .prefix_block_manager
                .block_ref_count(ManagedBlockId::new(first, 0, KvBlockId::new(1))),
            2
        );
        assert!(scheduler
            .prefix_block_manager
            .shared_prefix(second)
            .is_some());
    }

    #[test]
    fn reference_scheduler_releases_finished_paged_sequence_resources() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                kv_cache: crate::session::SessionKvCacheConfig::Paged { page_size: 2 },
                ..SessionConfig::default()
            },
            SchedulerConfig::default(),
        );
        let sequence_id = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        assert_eq!(scheduler.resident_session_count(), 1);
        let prefill = scheduler.execute_next_tick().unwrap().unwrap();
        assert!(prefill.updates[0].released_prefix_blocks.is_empty());

        let decode = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(
            decode.updates[0].finish_reason,
            Some(SequenceFinishReason::MaxNewTokens)
        );
        assert_eq!(scheduler.resident_session_count(), 0);
        assert_eq!(
            scheduler
                .prefix_block_manager
                .block_ref_count(ManagedBlockId::new(sequence_id, 0, KvBlockId::new(0))),
            0
        );
        assert_eq!(decode.updates[0].released_prefix_blocks.len(), 1);
        assert_eq!(
            decode.updates[0].released_prefix_blocks[0],
            ManagedBlockId::new(sequence_id, 0, KvBlockId::new(0))
        );
    }

    #[test]
    fn reference_scheduler_releases_shared_prefix_refs_when_target_finishes() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                kv_cache: crate::session::SessionKvCacheConfig::Paged { page_size: 2 },
                ..SessionConfig::default()
            },
            SchedulerConfig {
                max_batch_size: 2,
                min_prefix_share_tokens: 2,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1],
                max_new_tokens: 3,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let decode = scheduler.execute_next_tick().unwrap().unwrap();
        let second_update = decode
            .updates
            .iter()
            .find(|update| update.sequence_id == second)
            .unwrap();

        assert_eq!(
            second_update.finish_reason,
            Some(SequenceFinishReason::MaxNewTokens)
        );
        assert_eq!(scheduler.resident_session_count(), 1);
        assert_eq!(
            scheduler
                .prefix_block_manager
                .block_ref_count(ManagedBlockId::new(first, 0, KvBlockId::new(0))),
            1
        );
        assert_eq!(
            scheduler
                .prefix_block_manager
                .block_ref_count(ManagedBlockId::new(first, 0, KvBlockId::new(1))),
            1
        );
        assert!(!second_update.released_prefix_blocks.is_empty());
    }

    #[test]
    fn reference_scheduler_does_not_reuse_full_prompt_after_source_has_decoded() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                kv_cache: crate::session::SessionKvCacheConfig::Paged { page_size: 2 },
                ..SessionConfig::default()
            },
            SchedulerConfig {
                max_batch_size: 1,
                min_prefix_share_tokens: 2,
                ..SchedulerConfig::default()
            },
        );
        scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0],
                max_new_tokens: 2,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        let report = scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(report.updates[0].sequence_id, second);
        assert_eq!(report.updates[0].reused_prompt_tokens, 2);
        assert_eq!(report.updates[0].appended_prompt_tokens, 1);
    }

    #[test]
    fn reference_scheduler_breaks_prefill_streak_with_decode_when_decode_is_waiting() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 1,
                max_prefill_chunk_tokens: 2,
                max_consecutive_prefill_ticks: 2,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1],
                max_new_tokens: 2,
                stop_token_id: None,
            })
            .unwrap();

        scheduler.execute_next_tick().unwrap().unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![1, 0, 1, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(tick.sequence_ids(), &[first]);

        let report = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(report.tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(report.updates[0].sequence_id, first);
        assert_eq!(report.updates[0].generated_token.unwrap().token_id, 1);
        assert_eq!(
            scheduler.sequence_state(second).unwrap().phase(),
            SequencePhase::Prefill
        );
    }

    #[test]
    fn reference_scheduler_prefill_batch_respects_token_budget_and_packs_remainder_chunks() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 4,
                max_prefill_chunk_tokens: 4,
                max_prefill_batch_tokens: 5,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();
        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![1, 0, 1, 0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let first_tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(first_tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(first_tick.sequence_ids(), &[first]);
        scheduler.execute_next_tick().unwrap().unwrap();

        let second_tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(second_tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(second_tick.sequence_ids(), &[first, second]);
    }

    #[test]
    fn reference_scheduler_prefers_full_decode_batch_between_prefill_chunks() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 2,
                max_prefill_chunk_tokens: 2,
                max_consecutive_prefill_ticks: 8,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0],
                max_new_tokens: 2,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![1],
                max_new_tokens: 2,
                stop_token_id: None,
            })
            .unwrap();
        scheduler.execute_next_tick().unwrap().unwrap();

        let third = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0, 1, 0, 1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        let tick = scheduler.plan_next_tick().unwrap();
        assert_eq!(tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(tick.sequence_ids(), &[first, second]);

        let report = scheduler.execute_next_tick().unwrap().unwrap();
        assert_eq!(report.tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(report.tick.sequence_ids(), &[first, second]);
        assert_eq!(
            scheduler.sequence_state(third).unwrap().phase(),
            SequencePhase::Prefill
        );
    }

    #[test]
    fn reference_scheduler_decode_tick_batches_multiple_sequences_after_staggered_prefill() {
        let mut scheduler = test_scheduler(
            SessionConfig::default(),
            SchedulerConfig {
                max_batch_size: 2,
                ..SchedulerConfig::default()
            },
        );
        let first = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        scheduler.execute_next_tick().unwrap().unwrap();

        let second = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![1],
                max_new_tokens: 1,
                stop_token_id: None,
            })
            .unwrap();

        scheduler.execute_next_tick().unwrap().unwrap();
        let report = scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(report.tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(report.tick.sequence_ids(), &[first, second]);
        assert_eq!(report.updates.len(), 2);
        assert!(report
            .updates
            .iter()
            .all(|update| update.generated_token.is_some()));
        assert!(report
            .updates
            .iter()
            .all(|update| update.finish_reason == Some(SequenceFinishReason::MaxNewTokens)));
    }

    #[test]
    fn reference_scheduler_reports_session_limit_finish_reason() {
        let mut scheduler = test_scheduler(
            SessionConfig {
                max_sequence_length: 8,
                max_generated_tokens: 1,
                sampler: SamplerConfig::default(),
                ..SessionConfig::default()
            },
            SchedulerConfig::default(),
        );
        let sequence_id = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0],
                max_new_tokens: 3,
                stop_token_id: None,
            })
            .unwrap();

        scheduler.execute_next_tick().unwrap().unwrap();
        let report = scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(
            report.updates[0].finish_reason,
            Some(SequenceFinishReason::SessionLimit)
        );
        assert_eq!(
            scheduler
                .sequence_state(sequence_id)
                .unwrap()
                .finish_reason(),
            Some(SequenceFinishReason::SessionLimit)
        );
    }

    #[test]
    fn reference_scheduler_can_finish_immediately_after_prefill() {
        let mut scheduler = test_scheduler(SessionConfig::default(), SchedulerConfig::default());
        let sequence_id = scheduler
            .submit(SequenceSubmitRequest {
                prompt_token_ids: vec![0],
                max_new_tokens: 0,
                stop_token_id: None,
            })
            .unwrap();

        let report = scheduler.execute_next_tick().unwrap().unwrap();

        assert_eq!(report.tick.batch_kind(), SchedulerBatchKind::Prefill);
        assert_eq!(
            report.updates[0].finish_reason,
            Some(SequenceFinishReason::MaxNewTokens)
        );
        assert_eq!(
            scheduler.sequence_state(sequence_id).unwrap().phase(),
            SequencePhase::Finished
        );
    }

    fn test_scheduler(
        session_config: SessionConfig,
        scheduler_config: SchedulerConfig,
    ) -> ReferenceScheduler {
        ReferenceScheduler::new(
            Arc::new(always_one_model()),
            Arc::new(DummyTokenizer) as Arc<dyn Tokenizer>,
            session_config,
            scheduler_config,
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
