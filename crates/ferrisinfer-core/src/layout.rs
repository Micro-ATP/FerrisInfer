use crate::error::{ErrorKind, FerrisError, Result};
use crate::shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    offset_elements: usize,
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Self {
        let strides = contiguous_strides(shape.dims());

        Self {
            shape,
            strides,
            offset_elements: 0,
        }
    }

    pub fn with_strides(shape: Shape, strides: Vec<usize>, offset_elements: usize) -> Result<Self> {
        if shape.rank() != strides.len() {
            return Err(FerrisError::new(
                ErrorKind::InvalidLayout,
                "shape rank does not match stride rank",
            ));
        }

        Ok(Self {
            shape,
            strides,
            offset_elements,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset_elements(&self) -> usize {
        self.offset_elements
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(self.shape.dims())
    }
}

pub fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1; dims.len()];
    let mut running = 1usize;

    for index in (0..dims.len()).rev() {
        strides[index] = running;
        running = running.saturating_mul(dims[index]);
    }

    strides
}
