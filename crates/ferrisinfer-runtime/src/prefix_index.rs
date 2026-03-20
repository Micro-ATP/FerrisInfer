use std::collections::HashMap;

use crate::sequence::SequenceId;

#[derive(Debug, Clone)]
pub struct PrefixIndexConfig {
    pub min_prefix_share_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PrefixIndexEntry {
    pub sequence_id: SequenceId,
    pub prompt_tokens: Vec<u32>,
    pub cached_prompt_tokens: usize,
    pub session_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatchCandidate {
    pub sequence_id: SequenceId,
    pub shared_prefix_tokens: usize,
}

#[derive(Debug, Default)]
pub struct PrefixIndex {
    min_prefix_share_tokens: usize,
    buckets: HashMap<Vec<u32>, Vec<PrefixIndexEntry>>,
}

impl PrefixIndex {
    pub fn new(config: PrefixIndexConfig) -> Self {
        Self {
            min_prefix_share_tokens: config.min_prefix_share_tokens,
            buckets: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    pub fn upsert(&mut self, entry: PrefixIndexEntry) {
        if self.min_prefix_share_tokens == 0
            || entry.prompt_tokens.len() < self.min_prefix_share_tokens
            || entry.cached_prompt_tokens < self.min_prefix_share_tokens
        {
            return;
        }

        let bucket_key = entry.prompt_tokens[..self.min_prefix_share_tokens].to_vec();
        self.buckets.entry(bucket_key).or_default().push(entry);
    }

    pub fn best_match(
        &self,
        target_sequence_id: SequenceId,
        target_prompt_tokens: &[u32],
    ) -> Option<PrefixMatchCandidate> {
        if self.min_prefix_share_tokens == 0
            || target_prompt_tokens.len() < self.min_prefix_share_tokens
        {
            return None;
        }

        let bucket_key = target_prompt_tokens[..self.min_prefix_share_tokens].to_vec();
        let candidates = self.buckets.get(&bucket_key)?;

        candidates
            .iter()
            .filter(|candidate| candidate.sequence_id != target_sequence_id)
            .filter_map(|candidate| {
                let shared_prefix_tokens =
                    common_prefix_len(target_prompt_tokens, candidate.prompt_tokens.as_slice())
                        .min(candidate.cached_prompt_tokens);
                let normalized_shared_prefix_tokens = normalize_shared_prefix_len(
                    shared_prefix_tokens,
                    target_prompt_tokens.len(),
                    candidate.session_position,
                );
                (normalized_shared_prefix_tokens >= self.min_prefix_share_tokens).then_some(
                    PrefixMatchCandidate {
                        sequence_id: candidate.sequence_id,
                        shared_prefix_tokens: normalized_shared_prefix_tokens,
                    },
                )
            })
            .max_by_key(|candidate| candidate.shared_prefix_tokens)
    }
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn normalize_shared_prefix_len(
    shared_prefix_tokens: usize,
    target_prompt_len: usize,
    source_session_position: usize,
) -> usize {
    if shared_prefix_tokens == target_prompt_len && source_session_position > shared_prefix_tokens {
        shared_prefix_tokens.saturating_sub(1)
    } else {
        shared_prefix_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_index_groups_candidates_by_minimum_prefix_bucket() {
        let mut index = PrefixIndex::new(PrefixIndexConfig {
            min_prefix_share_tokens: 2,
        });
        index.upsert(PrefixIndexEntry {
            sequence_id: SequenceId::new(1),
            prompt_tokens: vec![10, 11, 12, 13],
            cached_prompt_tokens: 4,
            session_position: 4,
        });
        index.upsert(PrefixIndexEntry {
            sequence_id: SequenceId::new(2),
            prompt_tokens: vec![10, 11, 99],
            cached_prompt_tokens: 3,
            session_position: 3,
        });
        index.upsert(PrefixIndexEntry {
            sequence_id: SequenceId::new(3),
            prompt_tokens: vec![10, 12, 13],
            cached_prompt_tokens: 3,
            session_position: 3,
        });

        let candidate = index
            .best_match(SequenceId::new(7), &[10, 11, 12, 99])
            .unwrap();
        assert_eq!(candidate.sequence_id, SequenceId::new(1));
        assert_eq!(candidate.shared_prefix_tokens, 3);
    }

    #[test]
    fn prefix_index_avoids_full_prompt_share_when_source_has_decode_token() {
        let mut index = PrefixIndex::new(PrefixIndexConfig {
            min_prefix_share_tokens: 2,
        });
        index.upsert(PrefixIndexEntry {
            sequence_id: SequenceId::new(1),
            prompt_tokens: vec![1, 2, 3],
            cached_prompt_tokens: 3,
            session_position: 4,
        });

        let candidate = index.best_match(SequenceId::new(9), &[1, 2, 3]).unwrap();
        assert_eq!(candidate.sequence_id, SequenceId::new(1));
        assert_eq!(candidate.shared_prefix_tokens, 2);
    }
}
