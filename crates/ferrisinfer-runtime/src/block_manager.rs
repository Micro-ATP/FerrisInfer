use std::collections::HashMap;

use crate::paged_kv::{KvBlockId, PrefixHandle};
use crate::sequence::SequenceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagedBlockId {
    owner_sequence_id: SequenceId,
    layer_index: usize,
    block_id: KvBlockId,
}

impl ManagedBlockId {
    pub fn new(owner_sequence_id: SequenceId, layer_index: usize, block_id: KvBlockId) -> Self {
        Self {
            owner_sequence_id,
            layer_index,
            block_id,
        }
    }

    pub fn owner_sequence_id(self) -> SequenceId {
        self.owner_sequence_id
    }

    pub fn layer_index(self) -> usize {
        self.layer_index
    }

    pub fn block_id(self) -> KvBlockId {
        self.block_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedPrefixAssignment {
    source_sequence_id: SequenceId,
    handle: PrefixHandle,
}

impl SharedPrefixAssignment {
    pub fn new(source_sequence_id: SequenceId, handle: PrefixHandle) -> Self {
        Self {
            source_sequence_id,
            handle,
        }
    }

    pub fn source_sequence_id(&self) -> SequenceId {
        self.source_sequence_id
    }

    pub fn handle(&self) -> &PrefixHandle {
        &self.handle
    }
}

#[derive(Debug, Default)]
pub struct PrefixBlockManager {
    owned_prefixes: HashMap<SequenceId, PrefixHandle>,
    shared_prefixes: HashMap<SequenceId, SharedPrefixAssignment>,
    ref_counts: HashMap<ManagedBlockId, usize>,
}

impl PrefixBlockManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.owned_prefixes.clear();
        self.shared_prefixes.clear();
        self.ref_counts.clear();
    }

    pub fn owned_prefix(&self, sequence_id: SequenceId) -> Option<&PrefixHandle> {
        self.owned_prefixes.get(&sequence_id)
    }

    pub fn shared_prefix(&self, sequence_id: SequenceId) -> Option<&SharedPrefixAssignment> {
        self.shared_prefixes.get(&sequence_id)
    }

    pub fn update_owned_prefix(&mut self, sequence_id: SequenceId, handle: Option<PrefixHandle>) {
        if let Some(previous) = self.owned_prefixes.remove(&sequence_id) {
            self.apply_handle_refs(sequence_id, &previous, false);
        }

        if let Some(handle) = handle {
            self.apply_handle_refs(sequence_id, &handle, true);
            self.owned_prefixes.insert(sequence_id, handle);
        }
    }

    pub fn update_shared_prefix(
        &mut self,
        sequence_id: SequenceId,
        assignment: Option<SharedPrefixAssignment>,
    ) {
        if let Some(previous) = self.shared_prefixes.remove(&sequence_id) {
            self.apply_handle_refs(previous.source_sequence_id(), previous.handle(), false);
        }

        if let Some(assignment) = assignment {
            self.apply_handle_refs(assignment.source_sequence_id(), assignment.handle(), true);
            self.shared_prefixes.insert(sequence_id, assignment);
        }
    }

    pub fn remove_sequence(&mut self, sequence_id: SequenceId) {
        self.update_shared_prefix(sequence_id, None);
        self.update_owned_prefix(sequence_id, None);
    }

    pub fn block_ref_count(&self, block_id: ManagedBlockId) -> usize {
        self.ref_counts.get(&block_id).copied().unwrap_or(0)
    }

    fn apply_handle_refs(
        &mut self,
        owner_sequence_id: SequenceId,
        handle: &PrefixHandle,
        increment: bool,
    ) {
        for layer_index in 0..handle.layer_count() {
            let Some(entries) = handle.layer_block_table(layer_index) else {
                continue;
            };

            for entry in entries {
                let block_id =
                    ManagedBlockId::new(owner_sequence_id, layer_index, entry.block_id());
                if increment {
                    *self.ref_counts.entry(block_id).or_insert(0) += 1;
                } else {
                    let should_remove = if let Some(count) = self.ref_counts.get_mut(&block_id) {
                        *count = count.saturating_sub(1);
                        *count == 0
                    } else {
                        false
                    };
                    if should_remove {
                        self.ref_counts.remove(&block_id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::{KvBlockTableEntry, PrefixHandle};

    #[test]
    fn prefix_block_manager_updates_owned_prefix_ref_counts() {
        let mut manager = PrefixBlockManager::new();
        let sequence_id = SequenceId::new(3);
        let full_prefix = PrefixHandle::new_for_tests(
            4,
            2,
            vec![vec![
                KvBlockTableEntry::new_for_tests(0, 0, 0, 2),
                KvBlockTableEntry::new_for_tests(1, 1, 2, 2),
            ]],
        );
        let shorter_prefix = PrefixHandle::new_for_tests(
            2,
            2,
            vec![vec![KvBlockTableEntry::new_for_tests(0, 0, 0, 2)]],
        );

        manager.update_owned_prefix(sequence_id, Some(full_prefix));
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(sequence_id, 0, KvBlockId::new(0))),
            1
        );
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(sequence_id, 0, KvBlockId::new(1))),
            1
        );

        manager.update_owned_prefix(sequence_id, Some(shorter_prefix));
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(sequence_id, 0, KvBlockId::new(0))),
            1
        );
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(sequence_id, 0, KvBlockId::new(1))),
            0
        );
    }

    #[test]
    fn prefix_block_manager_tracks_shared_prefix_references_against_source_blocks() {
        let mut manager = PrefixBlockManager::new();
        let source_sequence_id = SequenceId::new(1);
        let target_sequence_id = SequenceId::new(2);
        let shared_prefix = PrefixHandle::new_for_tests(
            3,
            2,
            vec![vec![
                KvBlockTableEntry::new_for_tests(0, 0, 0, 2),
                KvBlockTableEntry::new_for_tests(1, 1, 2, 1),
            ]],
        );

        manager.update_owned_prefix(source_sequence_id, Some(shared_prefix.clone()));
        manager.update_shared_prefix(
            target_sequence_id,
            Some(SharedPrefixAssignment::new(
                source_sequence_id,
                shared_prefix.clone(),
            )),
        );

        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(
                source_sequence_id,
                0,
                KvBlockId::new(0)
            )),
            2
        );
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(
                source_sequence_id,
                0,
                KvBlockId::new(1)
            )),
            2
        );

        manager.remove_sequence(target_sequence_id);
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(
                source_sequence_id,
                0,
                KvBlockId::new(0)
            )),
            1
        );
        assert_eq!(
            manager.block_ref_count(ManagedBlockId::new(
                source_sequence_id,
                0,
                KvBlockId::new(1)
            )),
            1
        );
    }
}
