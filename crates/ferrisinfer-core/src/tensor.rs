use std::ops::Range;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{ErrorKind, FerrisError, Result};
use crate::layout::Layout;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub enum Storage {
    OwnedBytes(Vec<u8>),
    OwnedF32(Vec<f32>),
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
            storage: if dtype == DType::F32 {
                #[cfg(target_endian = "little")]
                {
                    Storage::OwnedF32(vec![0.0; element_count])
                }
                #[cfg(not(target_endian = "little"))]
                {
                    Storage::OwnedBytes(vec![0; byte_len])
                }
            } else {
                Storage::OwnedBytes(vec![0; byte_len])
            },
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
            storage: storage_from_owned_bytes(dtype, bytes)?,
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
            storage: storage_from_shared_bytes(dtype, bytes)?,
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

        Ok(Self {
            dtype: DType::F32,
            layout: Layout::contiguous(shape),
            storage: {
                #[cfg(target_endian = "little")]
                {
                    Storage::OwnedF32(values)
                }
                #[cfg(not(target_endian = "little"))]
                {
                    let mut bytes = Vec::with_capacity(values.len() * DType::F32.size_in_bytes());
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    Storage::OwnedBytes(bytes)
                }
            },
        })
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

    pub fn reshape_in_place(&mut self, shape: Shape) -> Result<()> {
        self.ensure_contiguous()?;

        if self.element_count() != shape.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "reshape must preserve element count",
            ));
        }

        self.layout = Layout::contiguous(shape);
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::OwnedBytes(bytes) => bytes.as_slice(),
            Storage::OwnedF32(values) => unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    values.len() * DType::F32.size_in_bytes(),
                )
            },
            Storage::Shared(bytes) => bytes.as_ref(),
        }
    }

    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        match &mut self.storage {
            Storage::OwnedBytes(bytes) => Ok(bytes.as_mut_slice()),
            Storage::OwnedF32(values) => unsafe {
                Ok(std::slice::from_raw_parts_mut(
                    values.as_mut_ptr() as *mut u8,
                    values.len() * DType::F32.size_in_bytes(),
                ))
            },
            Storage::Shared(_) => Err(FerrisError::unsupported(
                "shared tensor storage is immutable",
            )),
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        match &self.storage {
            Storage::OwnedF32(values) => Ok(values.as_slice()),
            _ => Err(FerrisError::unsupported(
                "direct f32 slice access requires owned f32 tensor storage",
            )),
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32]> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        match &mut self.storage {
            Storage::OwnedF32(values) => Ok(values.as_mut_slice()),
            _ => Err(FerrisError::unsupported(
                "direct mutable f32 slice access requires owned f32 tensor storage",
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

        if let Storage::OwnedF32(values) = &self.storage {
            return Ok(values.clone());
        }

        let byte_range = self.element_byte_range_span(0, self.element_count())?;
        let bytes = &self.as_bytes()[byte_range];

        #[cfg(target_endian = "little")]
        {
            let mut values = Vec::<f32>::with_capacity(self.element_count());
            unsafe {
                values.set_len(self.element_count());
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    values.as_mut_ptr() as *mut u8,
                    bytes.len(),
                );
            }
            Ok(values)
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut values = Vec::with_capacity(self.element_count());
            for chunk in bytes.chunks_exact(DType::F32.size_in_bytes()) {
                let mut array = [0u8; 4];
                array.copy_from_slice(chunk);
                values.push(f32::from_le_bytes(array));
            }

            Ok(values)
        }
    }

    pub fn copy_from_f32_slice(&mut self, values: &[f32]) -> Result<()> {
        self.copy_from_f32_slice_at(0, values)
    }

    pub fn copy_from_f32_slice_at(&mut self, element_offset: usize, values: &[f32]) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        if let Storage::OwnedF32(storage) = &mut self.storage {
            let end = element_offset.checked_add(values.len()).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "tensor element range overflow")
            })?;
            if end > storage.len() {
                return Err(FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "tensor element range out of bounds, end {end} for {} elements",
                        storage.len()
                    ),
                ));
            }

            storage[element_offset..end].copy_from_slice(values);
            return Ok(());
        }

        let byte_range = self.element_byte_range_span(element_offset, values.len())?;
        let bytes = &mut self.as_bytes_mut()?[byte_range];

        #[cfg(target_endian = "little")]
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }

        #[cfg(not(target_endian = "little"))]
        for (chunk, value) in bytes
            .chunks_exact_mut(DType::F32.size_in_bytes())
            .zip(values.iter().copied())
        {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        Ok(())
    }

    pub fn copy_from_tensor_f32_at(
        &mut self,
        element_offset: usize,
        source: &Tensor,
    ) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;
        source.ensure_dtype(DType::F32)?;
        source.ensure_contiguous()?;

        if let (Storage::OwnedF32(destination), Ok(source_values)) =
            (&mut self.storage, source.as_f32_slice())
        {
            let end = element_offset
                .checked_add(source_values.len())
                .ok_or_else(|| {
                    FerrisError::new(ErrorKind::Runtime, "tensor element range overflow")
                })?;
            if end > destination.len() {
                return Err(FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "tensor element range out of bounds, end {end} for {} elements",
                        destination.len()
                    ),
                ));
            }

            destination[element_offset..end].copy_from_slice(source_values);
            return Ok(());
        }

        let source_byte_range = source.element_byte_range_span(0, source.element_count())?;
        let byte_range = self.element_byte_range_span(element_offset, source.element_count())?;
        let destination = &mut self.as_bytes_mut()?[byte_range];
        destination.copy_from_slice(&source.as_bytes()[source_byte_range]);
        Ok(())
    }

    pub fn read_f32(&self, index: usize) -> Result<f32> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        if let Storage::OwnedF32(values) = &self.storage {
            return values.get(index).copied().ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "tensor element index out of bounds, index {index} for {} elements",
                        values.len()
                    ),
                )
            });
        }

        let byte_range = self.element_byte_range(index)?;
        let bytes = &self.as_bytes()[byte_range];

        let mut array = [0u8; 4];
        array.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(array))
    }

    pub fn write_f32(&mut self, index: usize, value: f32) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        if let Storage::OwnedF32(values) = &mut self.storage {
            let len = values.len();
            let slot = values.get_mut(index).ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "tensor element index out of bounds, index {index} for {} elements",
                        len
                    ),
                )
            })?;
            *slot = value;
            return Ok(());
        }

        let byte_range = self.element_byte_range(index)?;
        let bytes = self.as_bytes_mut()?;
        bytes[byte_range].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn fill_f32(&mut self, value: f32) -> Result<()> {
        self.ensure_dtype(DType::F32)?;
        self.ensure_contiguous()?;

        if let Storage::OwnedF32(values) = &mut self.storage {
            values.fill(value);
            return Ok(());
        }

        let byte_range = self.element_byte_range_span(0, self.element_count())?;
        let bytes = &mut self.as_bytes_mut()?[byte_range];
        let pattern = value.to_le_bytes();

        for chunk in bytes.chunks_exact_mut(DType::F32.size_in_bytes()) {
            chunk.copy_from_slice(&pattern);
        }

        Ok(())
    }

    fn element_byte_range(&self, index: usize) -> Result<Range<usize>> {
        self.element_byte_range_span(index, 1)
    }

    fn element_byte_range_span(&self, index: usize, element_count: usize) -> Result<Range<usize>> {
        if index > self.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor element index out of bounds, index {index} for {} elements",
                    self.element_count()
                ),
            ));
        }

        let end_index = index
            .checked_add(element_count)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor element range overflow"))?;
        if end_index > self.element_count() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "tensor element range out of bounds, end {end_index} for {} elements",
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
        let byte_len = element_count
            .checked_mul(element_size)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte length overflow"))?;
        let end = start
            .checked_add(byte_len)
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

fn storage_from_owned_bytes(dtype: DType, bytes: Vec<u8>) -> Result<Storage> {
    if dtype == DType::F32 {
        #[cfg(target_endian = "little")]
        {
            return Ok(Storage::OwnedF32(bytes_to_f32_vec(&bytes)?));
        }
    }

    Ok(Storage::OwnedBytes(bytes))
}

fn storage_from_shared_bytes(dtype: DType, bytes: Arc<[u8]>) -> Result<Storage> {
    if dtype == DType::F32 {
        #[cfg(target_endian = "little")]
        {
            return Ok(Storage::OwnedF32(bytes_to_f32_vec(bytes.as_ref())?));
        }
    }

    Ok(Storage::Shared(bytes))
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % DType::F32.size_in_bytes() != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidType,
            "f32 tensor bytes must be a multiple of 4",
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let len = bytes.len() / DType::F32.size_in_bytes();
        let mut values = Vec::<f32>::with_capacity(len);
        unsafe {
            values.set_len(len);
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                values.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Ok(values)
    }

    #[cfg(not(target_endian = "little"))]
    {
        let mut values = Vec::with_capacity(bytes.len() / DType::F32.size_in_bytes());
        for chunk in bytes.chunks_exact(DType::F32.size_in_bytes()) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }
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

    #[test]
    fn copy_from_tensor_f32_at_updates_subrange() {
        let mut destination =
            Tensor::from_f32_vec(Shape::from_slice(&[6]).unwrap(), vec![0.0; 6]).unwrap();
        let source =
            Tensor::from_f32_vec(Shape::from_slice(&[2]).unwrap(), vec![3.0, 4.0]).unwrap();

        destination.copy_from_tensor_f32_at(2, &source).unwrap();

        assert_eq!(
            destination.to_vec_f32().unwrap(),
            vec![0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }
}
