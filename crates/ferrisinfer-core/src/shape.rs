use crate::error::{ErrorKind, FerrisError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn new(dims: Vec<usize>) -> Result<Self> {
        if dims.iter().any(|&dim| dim == 0) {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "shape dimensions must be greater than zero",
            ));
        }

        Ok(Self { dims })
    }

    pub fn from_slice(dims: &[usize]) -> Result<Self> {
        Self::new(dims.to_vec())
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn element_count(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }
}
