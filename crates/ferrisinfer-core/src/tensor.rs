use std::ops::Range;
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

    pub fn from_f32_vec(shape: Shape, values: Vec<f32>) -> Result<Self> {
        let expected = shape.element_count();
        if values.len() != expected {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor element count mismatch, expected {expected} values but got {}",
                    values.len()
                ),
            ));
        }

        let mut bytes = Vec::with_capacity(values.len() * DType::F32.size_in_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        Self::from_owned_bytes(DType::F32, shape, bytes)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self> {
        self.ensure_contiguous()?;

        if self.element_count() != shape.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "reshape must preserve element count",
            ));
        }

        self.layout = Layout::contiguous(shape);
        Ok(self)
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

    pub fn ensure_dtype(&self, dtype: DType) -> Result<()> {
        if self.dtype != dtype {
            return Err(FerrisError::new(
                ErrorKind::InvalidType,
                format!(
                    "tensor dtype mismatch, expected {} but got {}",
                    dtype.name(),
                    self.dtype.name()
                ),
            ));
        }

        Ok(())
    }

    pub fn ensure_contiguous(&self) -> Result<()> {
        if !self.is_contiguous() {
            return Err(FerrisError::new(
                ErrorKind::InvalidLayout,
                "operation requires a contiguous tensor layout",
            ));
        }

        Ok(())
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        let mut values = Vec::with_capacity(self.element_count());
        for index in 0..self.element_count() {
            values.push(self.read_f32(index)?);
        }

        Ok(values)
    }

    pub fn copy_from_f32_slice(&mut self, values: &[f32]) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        if values.len() != self.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor element count mismatch, expected {} values but got {}",
                    self.element_count(),
                    values.len()
                ),
            ));
        }

        for (index, value) in values.iter().copied().enumerate() {
            self.write_f32(index, value)?;
        }

        Ok(())
    }

    pub fn read_f32(&self, index: usize) -> Result<f32> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        let byte_range = self.element_byte_range(index)?;
        let bytes = &self.as_bytes()[byte_range];

        let mut array = [0u8; 4];
        array.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(array))
    }

    pub fn write_f32(&mut self, index: usize, value: f32) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        let byte_range = self.element_byte_range(index)?;
        let bytes = self.as_bytes_mut()?;
        bytes[byte_range].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn fill_f32(&mut self, value: f32) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        for index in 0..self.element_count() {
            self.write_f32(index, value)?;
        }

        Ok(())
    }

    fn element_byte_range(&self, index: usize) -> Result<Range<usize>> {
        if index >= self.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor element index out of bounds, index {index} for {} elements",
                    self.element_count()
                ),
            ));
        }

        let start_element = self
            .layout
            .offset_elements()
            .checked_add(index)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor index overflow"))?;

        let element_size = self.dtype.size_in_bytes();
        let start = start_element
            .checked_mul(element_size)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte offset overflow"))?;
        let end = start
            .checked_add(element_size)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte range overflow"))?;

        Ok(start..end)
    }
}

fn byte_len(dtype: DType, element_count: usize) -> Result<usize> {
    dtype
        .size_in_bytes()
        .checked_mul(element_count)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_tensor_round_trip_preserves_values() {
        let shape = Shape::from_slice(&[2, 2]).unwrap();
        let tensor = Tensor::from_f32_vec(shape, vec![1.0, 2.5, -3.0, 4.25]).unwrap();

        assert_eq!(tensor.to_vec_f32().unwrap(), vec![1.0, 2.5, -3.0, 4.25]);
    }

    #[test]
    fn reshape_preserves_storage_and_updates_shape() {
        let shape = Shape::from_slice(&[2, 2]).unwrap();
        let reshaped = Tensor::from_f32_vec(shape, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(Shape::from_slice(&[4]).unwrap())
            .unwrap();

        assert_eq!(reshaped.shape().dims(), &[4]);
        assert_eq!(reshaped.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reshape_rejects_mismatched_element_count() {
        let shape = Shape::from_slice(&[2, 2]).unwrap();
        let error = Tensor::from_f32_vec(shape, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape(Shape::from_slice(&[3]).unwrap())
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidShape);
    }

    #[test]
    fn copy_from_f32_slice_updates_tensor_contents() {
        let shape = Shape::from_slice(&[4]).unwrap();
        let mut tensor = Tensor::zeros(DType::F32, shape).unwrap();
        tensor.copy_from_f32_slice(&[0.5, 1.5, 2.5, 3.5]).unwrap();

        assert_eq!(tensor.to_vec_f32().unwrap(), vec![0.5, 1.5, 2.5, 3.5]);
    }
}
