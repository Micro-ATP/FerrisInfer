use std::collections::BTreeMap;

use ferrisinfer_core::Tensor;

#[derive(Debug, Default)]
pub struct WeightMap {
    tensors: BTreeMap<String, Tensor>,
}

impl WeightMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) -> Option<Tensor> {
        self.tensors.insert(name.into(), tensor)
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }
}
