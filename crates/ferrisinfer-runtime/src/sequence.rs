#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(u64);

impl RequestId {
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    Prefill,
    Decode,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceFinishReason {
    StopToken,
    MaxNewTokens,
    SessionLimit,
    SequenceLength,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SequenceState {
    request_id: RequestId,
    sequence_id: SequenceId,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    cached_tokens: usize,
    requested_max_new_tokens: usize,
    max_new_tokens: usize,
    stop_token_id: Option<u32>,
    phase: SequencePhase,
    finish_reason: Option<SequenceFinishReason>,
}

impl SequenceState {
    pub fn new(
        request_id: RequestId,
        sequence_id: SequenceId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        stop_token_id: Option<u32>,
    ) -> Self {
        Self::with_generation_budget(
            request_id,
            sequence_id,
            prompt_tokens,
            max_new_tokens,
            max_new_tokens,
            stop_token_id,
        )
    }

    pub fn with_generation_budget(
        request_id: RequestId,
        sequence_id: SequenceId,
        prompt_tokens: Vec<u32>,
        requested_max_new_tokens: usize,
        generation_budget: usize,
        stop_token_id: Option<u32>,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            cached_tokens: 0,
            requested_max_new_tokens,
            max_new_tokens: generation_budget,
            stop_token_id,
            phase: SequencePhase::Prefill,
            finish_reason: None,
        }
    }

    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    pub fn generated_len(&self) -> usize {
        self.generated_tokens.len()
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn cached_prompt_tokens(&self) -> usize {
        self.cached_tokens.min(self.prompt_tokens.len())
    }

    pub fn pending_prompt_tokens(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(self.cached_prompt_tokens())
    }

    pub fn next_prefill_chunk_len(&self, max_chunk_tokens: usize) -> usize {
        self.pending_prompt_tokens().min(max_chunk_tokens)
    }

    pub fn next_prefill_chunk(&self, max_chunk_tokens: usize) -> &[u32] {
        let start = self.cached_prompt_tokens();
        let end = start + self.next_prefill_chunk_len(max_chunk_tokens);
        &self.prompt_tokens[start..end]
    }

    pub fn requested_max_new_tokens(&self) -> usize {
        self.requested_max_new_tokens
    }

    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    pub fn stop_token_id(&self) -> Option<u32> {
        self.stop_token_id
    }

    pub fn phase(&self) -> SequencePhase {
        self.phase
    }

    pub fn finish_reason(&self) -> Option<SequenceFinishReason> {
        self.finish_reason
    }

    pub fn is_finished(&self) -> bool {
        self.phase == SequencePhase::Finished
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    pub fn generation_budget_reached(&self) -> bool {
        self.generated_tokens.len() >= self.max_new_tokens
    }

    pub fn generation_budget_finish_reason(&self) -> SequenceFinishReason {
        if self.requested_max_new_tokens > self.max_new_tokens {
            SequenceFinishReason::SessionLimit
        } else {
            SequenceFinishReason::MaxNewTokens
        }
    }

    pub fn record_prefill(&mut self, appended_tokens: usize) {
        self.cached_tokens = self.cached_tokens.saturating_add(appended_tokens);
        if self.phase != SequencePhase::Finished {
            self.phase = if self.pending_prompt_tokens() == 0 {
                SequencePhase::Decode
            } else {
                SequencePhase::Prefill
            };
        }
    }

    pub fn record_generated_token(&mut self, token_id: u32) {
        self.generated_tokens.push(token_id);
        self.cached_tokens = self.cached_tokens.saturating_add(1);
        if self.phase != SequencePhase::Finished {
            self.phase = SequencePhase::Decode;
        }
    }

    pub fn finish(&mut self, reason: SequenceFinishReason) {
        self.phase = SequencePhase::Finished;
        self.finish_reason = Some(reason);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerBatchKind {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerTick {
    batch_kind: SchedulerBatchKind,
    sequence_ids: Vec<SequenceId>,
}

impl SchedulerTick {
    pub fn new(batch_kind: SchedulerBatchKind, sequence_ids: Vec<SequenceId>) -> Self {
        Self {
            batch_kind,
            sequence_ids,
        }
    }

    pub fn batch_kind(&self) -> SchedulerBatchKind {
        self.batch_kind
    }

    pub fn sequence_ids(&self) -> &[SequenceId] {
        &self.sequence_ids
    }

    pub fn is_empty(&self) -> bool {
        self.sequence_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_state_tracks_prefill_decode_and_finish() {
        let mut sequence = SequenceState::new(
            RequestId::new(7),
            SequenceId::new(11),
            vec![1, 2, 3],
            16,
            Some(99),
        );

        assert_eq!(sequence.request_id().raw(), 7);
        assert_eq!(sequence.sequence_id().raw(), 11);
        assert_eq!(sequence.prompt_tokens(), &[1, 2, 3]);
        assert_eq!(sequence.generated_tokens(), &[]);
        assert_eq!(sequence.generated_len(), 0);
        assert_eq!(sequence.cached_tokens(), 0);
        assert_eq!(sequence.cached_prompt_tokens(), 0);
        assert_eq!(sequence.pending_prompt_tokens(), 3);
        assert_eq!(sequence.requested_max_new_tokens(), 16);
        assert_eq!(sequence.max_new_tokens(), 16);
        assert_eq!(sequence.phase(), SequencePhase::Prefill);
        assert_eq!(sequence.finish_reason(), None);
        assert!(!sequence.is_finished());
        assert!(!sequence.generation_budget_reached());

        sequence.record_prefill(3);
        assert_eq!(sequence.cached_tokens(), 3);
        assert_eq!(sequence.cached_prompt_tokens(), 3);
        assert_eq!(sequence.pending_prompt_tokens(), 0);
        assert_eq!(sequence.phase(), SequencePhase::Decode);

        sequence.record_generated_token(42);
        assert_eq!(sequence.generated_tokens(), &[42]);
        assert_eq!(sequence.generated_len(), 1);
        assert_eq!(sequence.cached_tokens(), 4);
        assert_eq!(sequence.total_tokens(), 4);

        sequence.finish(SequenceFinishReason::StopToken);
        assert_eq!(sequence.phase(), SequencePhase::Finished);
        assert_eq!(
            sequence.finish_reason(),
            Some(SequenceFinishReason::StopToken)
        );
        assert!(sequence.is_finished());
    }

    #[test]
    fn sequence_state_stays_in_prefill_until_prompt_is_fully_cached() {
        let mut sequence = SequenceState::new(
            RequestId::new(3),
            SequenceId::new(5),
            vec![10, 11, 12, 13, 14],
            8,
            None,
        );

        assert_eq!(sequence.next_prefill_chunk_len(2), 2);
        assert_eq!(sequence.next_prefill_chunk(2), &[10, 11]);

        sequence.record_prefill(2);
        assert_eq!(sequence.cached_prompt_tokens(), 2);
        assert_eq!(sequence.pending_prompt_tokens(), 3);
        assert_eq!(sequence.phase(), SequencePhase::Prefill);
        assert_eq!(sequence.next_prefill_chunk(2), &[12, 13]);

        sequence.record_prefill(2);
        assert_eq!(sequence.cached_prompt_tokens(), 4);
        assert_eq!(sequence.pending_prompt_tokens(), 1);
        assert_eq!(sequence.phase(), SequencePhase::Prefill);
        assert_eq!(sequence.next_prefill_chunk(2), &[14]);

        sequence.record_prefill(1);
        assert_eq!(sequence.cached_prompt_tokens(), 5);
        assert_eq!(sequence.pending_prompt_tokens(), 0);
        assert_eq!(sequence.phase(), SequencePhase::Decode);
    }

    #[test]
    fn sequence_state_distinguishes_requested_limit_from_session_budget() {
        let mut sequence = SequenceState::with_generation_budget(
            RequestId::new(1),
            SequenceId::new(2),
            vec![7],
            4,
            1,
            None,
        );

        assert_eq!(sequence.requested_max_new_tokens(), 4);
        assert_eq!(sequence.max_new_tokens(), 1);
        assert_eq!(
            sequence.generation_budget_finish_reason(),
            SequenceFinishReason::SessionLimit
        );

        sequence.record_prefill(1);
        sequence.record_generated_token(99);

        assert!(sequence.generation_budget_reached());
    }

    #[test]
    fn scheduler_tick_reports_batch_shape() {
        let tick = SchedulerTick::new(
            SchedulerBatchKind::Decode,
            vec![SequenceId::new(1), SequenceId::new(2)],
        );

        assert_eq!(tick.batch_kind(), SchedulerBatchKind::Decode);
        assert_eq!(
            tick.sequence_ids(),
            &[SequenceId::new(1), SequenceId::new(2)]
        );
        assert!(!tick.is_empty());
    }
}
