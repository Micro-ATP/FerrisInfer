use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{ErrorKind, FerrisError, Result};
use crate::layout::Layout;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub enum Storage {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    dtype: DType,
    layout: Layout,
    storage: Storage,
}

impl Tensor {
    pub fn zeros(dtype: DType, shape: Shape) -> Result<Self> {
        let element_count = shape.element_count();
        let byte_len = byte_len(dtype, element_count)?;
        let layout = Layout::contiguous(shape);

        Ok(Self {
            dtype,
            layout,
            storage: Storage::Owned(vec![0; byte_len]),
        })
    }

    pub fn from_owned_bytes(dtype: DType, shape: Shape, bytes: Vec<u8>) -> Result<Self> {
        let expected = byte_len(dtype, shape.element_count())?;
        if bytes.len() != expected {
            return Err(FerrisError::new(
                ErrorKind::InvalidType,
                format!(
                    "tensor byte size mismatch, expected {expected} bytes but got {}",
                    bytes.len()
                ),
            ));
        }

        Ok(Self {
            dtype,
            layout: Layout::contiguous(shape),
            storage: Storage::Owned(bytes),
        })
    }

    pub fn from_shared_bytes(dtype: DType, shape: Shape, bytes: Arc<[u8]>) -> Result<Self> {
        let expected = byte_len(dtype, shape.element_count())?;
        if bytes.len() != expected {
            return Err(FerrisError::new(
                ErrorKind::InvalidType,
                format!(
                    "tensor byte size mismatch, expected {expected} bytes but got {}",
                    bytes.len()
                ),
            ));
        }

        Ok(Self {
            dtype,
            layout: Layout::contiguous(shape),
            storage: Storage::Shared(bytes),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Owned(bytes) => bytes.as_slice(),
            Storage::Shared(bytes) => bytes.as_ref(),
        }
    }

    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        match &mut self.storage {
            Storage::Owned(bytes) => Ok(bytes.as_mut_slice()),
            Storage::Shared(_) => Err(FerrisError::unsupported(
                "shared tensor storage is immutable",
            )),
        }
    }
}

fn byte_len(dtype: DType, element_count: usize) -> Result<usize> {
    dtype
        .size_in_bytes()
        .checked_mul(element_count)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte size overflow"))
}
