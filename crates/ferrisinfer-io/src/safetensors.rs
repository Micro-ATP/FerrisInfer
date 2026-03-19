use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Shape, Tensor};

use crate::json::{parse_json, JsonValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    F32,
    F16,
    BF16,
}

impl SafeTensorDType {
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            other => Err(FerrisError::unsupported(format!(
                "safetensors dtype '{other}' is not supported yet"
            ))),
        }
    }

    fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    #[cfg(test)]
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorEntry {
    path: PathBuf,
    dtype: SafeTensorDType,
    shape: Vec<usize>,
    data_start_offset: u64,
    data_offsets: (u64, u64),
}

impl SafeTensorEntry {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }

    fn absolute_offset(&self) -> u64 {
        self.data_start_offset + self.data_offsets.0
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn read_raw_bytes(&self) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        self.read_raw_bytes_from(&mut file)
    }

    fn read_raw_bytes_from(&self, file: &mut File) -> Result<Vec<u8>> {
        let len = usize::try_from(self.byte_len()).map_err(|_| {
            FerrisError::new(
                ErrorKind::Runtime,
                "tensor byte length does not fit into usize",
            )
        })?;

        file.seek(SeekFrom::Start(self.absolute_offset()))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorsRepository {
    tensors: BTreeMap<String, SafeTensorEntry>,
    shard_count: usize,
}

impl SafeTensorsRepository {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let sharded_index_path = model_dir.join("model.safetensors.index.json");
        if sharded_index_path.is_file() {
            return Self::open_sharded(model_dir, &sharded_index_path);
        }

        let single_file_path = model_dir.join("model.safetensors");
        if single_file_path.is_file() {
            let tensors = load_entries_from_file(&single_file_path)?;
            return Ok(Self {
                tensors,
                shard_count: 1,
            });
        }

        Err(FerrisError::new(
            ErrorKind::Io,
            format!(
                "no model.safetensors or model.safetensors.index.json found under {}",
                model_dir.display()
            ),
        ))
    }

    fn open_sharded(model_dir: &Path, index_path: &Path) -> Result<Self> {
        let index_text = fs::read_to_string(index_path)?;
        let index_root = parse_json(&index_text)?;
        let weight_map = index_root.get("weight_map")?.as_object()?;

        let mut shard_names = BTreeSet::new();
        for value in weight_map.values() {
            shard_names.insert(value.as_str()?.to_string());
        }

        let mut shard_entries = BTreeMap::new();
        for shard_name in shard_names {
            let path = model_dir.join(&shard_name);
            shard_entries.insert(shard_name, load_entries_from_file(&path)?);
        }

        let mut tensors = BTreeMap::new();
        for (tensor_name, shard_name_value) in weight_map {
            let shard_name = shard_name_value.as_str()?;
            let shard = shard_entries.get(shard_name).ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::MissingWeight,
                    format!("missing shard '{shard_name}' referenced by index"),
                )
            })?;

            let entry = shard.get(tensor_name).ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::MissingWeight,
                    format!(
                        "tensor '{tensor_name}' referenced by index is absent from shard '{shard_name}'"
                    ),
                )
            })?;

            tensors.insert(tensor_name.clone(), entry.clone());
        }

        Ok(Self {
            tensors,
            shard_count: shard_entries.len(),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shard_count
    }

    pub fn total_data_bytes(&self) -> u64 {
        self.tensors.values().map(SafeTensorEntry::byte_len).sum()
    }

    pub fn get(&self, name: &str) -> Option<&SafeTensorEntry> {
        self.tensors.get(name)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn load_tensor_f32(&self, name: &str, transpose_2d: bool) -> Result<Tensor> {
        let entry = self.tensor_entry(name)?;
        let source_bytes = entry.read_raw_bytes()?;
        self.load_tensor_f32_from_bytes(name, entry, source_bytes, transpose_2d)
    }

    pub fn load_tensor_f32_cached(
        &self,
        name: &str,
        transpose_2d: bool,
        file_cache: &mut BTreeMap<PathBuf, File>,
    ) -> Result<Tensor> {
        let entry = self.tensor_entry(name)?;
        let source_bytes = if let Some(file) = file_cache.get_mut(&entry.path) {
            entry.read_raw_bytes_from(file)?
        } else {
            let mut file = File::open(&entry.path)?;
            let bytes = entry.read_raw_bytes_from(&mut file)?;
            file_cache.insert(entry.path.clone(), file);
            bytes
        };

        self.load_tensor_f32_from_bytes(name, entry, source_bytes, transpose_2d)
    }

    fn tensor_entry(&self, name: &str) -> Result<&SafeTensorEntry> {
        self.tensors.get(name).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::MissingWeight,
                format!("tensor '{name}' was not found in safetensors repository"),
            )
        })
    }

    fn load_tensor_f32_from_bytes(
        &self,
        name: &str,
        entry: &SafeTensorEntry,
        source_bytes: Vec<u8>,
        transpose_2d: bool,
    ) -> Result<Tensor> {
        let expected_len = checked_tensor_byte_len(entry.dtype, &entry.shape)?;
        if source_bytes.len() != expected_len {
            return Err(FerrisError::new(
                ErrorKind::InvalidType,
                format!(
                    "tensor '{name}' byte length mismatch, expected {expected_len} bytes but got {}",
                    source_bytes.len()
                ),
            ));
        }

        let output_shape = output_shape(&entry.shape, transpose_2d)?;
        let output_values =
            convert_to_f32_vec(entry.dtype, &entry.shape, &source_bytes, transpose_2d)?;
        Tensor::from_f32_vec(shape_from_dims(output_shape)?, output_values)
    }
}

fn load_entries_from_file(path: &Path) -> Result<BTreeMap<String, SafeTensorEntry>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);

    let data_start_offset = 8u64.checked_add(header_len).ok_or_else(|| {
        FerrisError::new(ErrorKind::Runtime, "safetensors header offset overflow")
    })?;

    if data_start_offset > file_len {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            format!("safetensors file '{}' is truncated", path.display()),
        ));
    }

    let header_len_usize = usize::try_from(header_len).map_err(|_| {
        FerrisError::new(
            ErrorKind::Runtime,
            "safetensors header length does not fit into usize",
        )
    })?;
    let mut header_bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut header_bytes)?;
    let header_text = String::from_utf8(header_bytes).map_err(|_| {
        FerrisError::new(
            ErrorKind::Parse,
            format!(
                "safetensors header in '{}' is not valid UTF-8",
                path.display()
            ),
        )
    })?;

    let root = parse_json(&header_text)?;
    let object = root.as_object()?;
    let mut entries = BTreeMap::new();

    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }

        let tensor = parse_tensor_entry(path, data_start_offset, file_len, value)?;
        entries.insert(name.clone(), tensor);
    }

    Ok(entries)
}

fn parse_tensor_entry(
    path: &Path,
    data_start_offset: u64,
    file_len: u64,
    value: &JsonValue,
) -> Result<SafeTensorEntry> {
    let object = value.as_object()?;
    let dtype = SafeTensorDType::from_str(
        object
            .get("dtype")
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Parse,
                    "safetensors tensor entry is missing dtype",
                )
            })?
            .as_str()?,
    )?;

    let shape = object
        .get("shape")
        .ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Parse,
                "safetensors tensor entry is missing shape",
            )
        })?
        .as_array()?
        .iter()
        .map(|value| {
            let dimension = value.as_number()?.as_u64()?;
            usize::try_from(dimension).map_err(|_| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    "safetensors tensor dimension does not fit into usize",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let offsets = object
        .get("data_offsets")
        .ok_or_else(|| {
            FerrisError::new(
                ErrorKind::Parse,
                "safetensors tensor entry is missing data_offsets",
            )
        })?
        .as_array()?;

    if offsets.len() != 2 {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            "safetensors tensor data_offsets must contain exactly two integers",
        ));
    }

    let start = offsets[0].as_number()?.as_u64()?;
    let end = offsets[1].as_number()?.as_u64()?;
    if end < start {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            "safetensors tensor data_offsets end must be >= start",
        ));
    }

    let tensor_byte_len = end - start;
    let expected_byte_len =
        u64::try_from(checked_tensor_byte_len(dtype, &shape)?).map_err(|_| {
            FerrisError::new(
                ErrorKind::Runtime,
                "safetensors tensor byte length does not fit into u64",
            )
        })?;
    if tensor_byte_len != expected_byte_len {
        return Err(FerrisError::new(
            ErrorKind::InvalidType,
            format!(
                "safetensors tensor byte length mismatch, expected {expected_byte_len} bytes but got {tensor_byte_len}"
            ),
        ));
    }

    let absolute_end = data_start_offset.checked_add(end).ok_or_else(|| {
        FerrisError::new(ErrorKind::Runtime, "safetensors tensor end offset overflow")
    })?;
    if absolute_end > file_len {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            format!(
                "tensor data in '{}' extends beyond file length",
                path.display()
            ),
        ));
    }

    Ok(SafeTensorEntry {
        path: path.to_path_buf(),
        dtype,
        shape,
        data_start_offset,
        data_offsets: (start, end),
    })
}

fn output_shape(shape: &[usize], transpose_2d: bool) -> Result<Vec<usize>> {
    if !transpose_2d {
        return Ok(shape.to_vec());
    }

    if shape.len() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "only rank-2 tensors can be transposed during safetensors loading",
        ));
    }

    Ok(vec![shape[1], shape[0]])
}

fn checked_tensor_byte_len(dtype: SafeTensorDType, shape: &[usize]) -> Result<usize> {
    let elements = element_count(shape)?;
    dtype
        .size_in_bytes()
        .checked_mul(elements)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor byte length overflow"))
}

fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }

    let mut count = 1usize;
    for &dimension in shape {
        count = count
            .checked_mul(dimension)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "tensor element count overflow"))?;
    }
    Ok(count)
}

fn convert_to_f32_vec(
    dtype: SafeTensorDType,
    shape: &[usize],
    source: &[u8],
    transpose_2d: bool,
) -> Result<Vec<f32>> {
    if dtype == SafeTensorDType::F32 && !transpose_2d {
        return raw_f32_bytes_to_vec(source);
    }

    let elements = element_count(shape)?;
    let mut output = vec![0.0f32; elements];
    let rows = if transpose_2d { shape[0] } else { 0 };
    let cols = if transpose_2d { shape[1] } else { 0 };

    for index in 0..elements {
        let value = match dtype {
            SafeTensorDType::F32 => read_f32(source, index)?,
            SafeTensorDType::BF16 => bf16_to_f32(read_u16(source, index)?),
            SafeTensorDType::F16 => f16_to_f32(read_u16(source, index)?),
        };

        let output_index = if transpose_2d {
            let row = index / cols;
            let col = index % cols;
            col * rows + row
        } else {
            index
        };

        output[output_index] = value;
    }

    Ok(output)
}

fn raw_f32_bytes_to_vec(source: &[u8]) -> Result<Vec<f32>> {
    if source.len() % DType::F32.size_in_bytes() != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidType,
            "source tensor bytes are truncated for f32 read",
        ));
    }

    #[cfg(target_endian = "little")]
    {
        let len = source.len() / DType::F32.size_in_bytes();
        let mut values = Vec::<f32>::with_capacity(len);
        unsafe {
            values.set_len(len);
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                values.as_mut_ptr() as *mut u8,
                source.len(),
            );
        }
        Ok(values)
    }

    #[cfg(not(target_endian = "little"))]
    {
        let mut values = Vec::with_capacity(source.len() / DType::F32.size_in_bytes());
        for chunk in source.chunks_exact(DType::F32.size_in_bytes()) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }
}

fn read_u16(source: &[u8], index: usize) -> Result<u16> {
    let start = index
        .checked_mul(2)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "u16 tensor byte offset overflow"))?;
    let end = start
        .checked_add(2)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "u16 tensor byte range overflow"))?;
    let bytes = source.get(start..end).ok_or_else(|| {
        FerrisError::new(
            ErrorKind::InvalidType,
            "source tensor bytes are truncated for u16 read",
        )
    })?;

    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_f32(source: &[u8], index: usize) -> Result<f32> {
    let start = index
        .checked_mul(4)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "f32 tensor byte offset overflow"))?;
    let end = start
        .checked_add(4)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "f32 tensor byte range overflow"))?;
    let bytes = source.get(start..end).ok_or_else(|| {
        FerrisError::new(
            ErrorKind::InvalidType,
            "source tensor bytes are truncated for f32 read",
        )
    })?;

    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as u32;

    let f32_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa_norm = mantissa;
                let mut exponent_norm = -14i32;
                while (mantissa_norm & 0x0400) == 0 {
                    mantissa_norm <<= 1;
                    exponent_norm -= 1;
                }
                mantissa_norm &= 0x03FF;
                sign | (((exponent_norm + 127) as u32) << 23) | (mantissa_norm << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (mantissa << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13),
    };

    f32::from_bits(f32_bits)
}

fn shape_from_dims(dims: Vec<usize>) -> Result<Shape> {
    if dims.is_empty() {
        Ok(Shape::scalar())
    } else {
        Shape::new(dims)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone)]
    struct TestTensorSpec {
        name: &'static str,
        dtype: SafeTensorDType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    #[test]
    fn safetensors_repository_loads_single_file_and_transposes_to_f32() {
        let temp = TestDir::new();
        write_safetensors_file(
            &temp.path().join("model.safetensors"),
            &[TestTensorSpec {
                name: "tensor",
                dtype: SafeTensorDType::BF16,
                shape: vec![3, 2],
                bytes: bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }],
        )
        .unwrap();

        let repository = SafeTensorsRepository::open(temp.path()).unwrap();
        let tensor = repository.load_tensor_f32("tensor", true).unwrap();

        assert_eq!(repository.tensor_count(), 1);
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(
            tensor.to_vec_f32().unwrap(),
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn safetensors_repository_loads_sharded_index() {
        let temp = TestDir::new();
        write_safetensors_file(
            &temp.path().join("model-00001-of-00002.safetensors"),
            &[TestTensorSpec {
                name: "first",
                dtype: SafeTensorDType::F32,
                shape: vec![2],
                bytes: f32_bytes(&[10.0, 20.0]),
            }],
        )
        .unwrap();
        write_safetensors_file(
            &temp.path().join("model-00002-of-00002.safetensors"),
            &[TestTensorSpec {
                name: "second",
                dtype: SafeTensorDType::F16,
                shape: vec![2],
                bytes: f16_bytes(&[1.5, -2.0]),
            }],
        )
        .unwrap();

        let index = r#"{
            "metadata": {"total_size": 12},
            "weight_map": {
                "first": "model-00001-of-00002.safetensors",
                "second": "model-00002-of-00002.safetensors"
            }
        }"#;
        fs::write(temp.path().join("model.safetensors.index.json"), index).unwrap();

        let repository = SafeTensorsRepository::open(temp.path()).unwrap();
        let first = repository.load_tensor_f32("first", false).unwrap();
        let second = repository.load_tensor_f32("second", false).unwrap();

        assert_eq!(repository.shard_count(), 2);
        assert_eq!(first.to_vec_f32().unwrap(), vec![10.0, 20.0]);
        assert_eq!(second.to_vec_f32().unwrap(), vec![1.5, -2.0]);
    }

    fn write_safetensors_file(path: &Path, tensors: &[TestTensorSpec]) -> Result<()> {
        let mut data_offset = 0u64;
        let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut data = Vec::new();

        for tensor in tensors {
            let start = data_offset;
            let end = start
                .checked_add(u64::try_from(tensor.bytes.len()).map_err(|_| {
                    FerrisError::new(
                        ErrorKind::Runtime,
                        "test tensor byte length does not fit into u64",
                    )
                })?)
                .ok_or_else(|| {
                    FerrisError::new(ErrorKind::Runtime, "test tensor offset overflow")
                })?;
            data_offset = end;

            let shape = tensor
                .shape
                .iter()
                .map(|dimension| dimension.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                ",\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                tensor.name,
                tensor.dtype.name(),
                shape,
                start,
                end
            ));
            data.extend_from_slice(&tensor.bytes);
        }

        header.push('}');
        let header_bytes = header.into_bytes();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&data);
        fs::write(path, file_bytes)?;
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bits = value.to_bits();
            bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        bytes
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&f32_to_f16(*value).to_le_bytes());
        }
        bytes
    }

    fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7F_FFFF;

        if exponent == 0xFF {
            if mantissa == 0 {
                return sign | 0x7C00;
            }
            return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1);
        }

        let half_exponent = exponent - 127 + 15;
        if half_exponent >= 0x1F {
            return sign | 0x7C00;
        }

        if half_exponent <= 0 {
            if half_exponent < -10 {
                return sign;
            }

            let mantissa_with_hidden = mantissa | 0x80_0000;
            let shift = u32::try_from(14 - half_exponent).unwrap();
            let mut half_mantissa = (mantissa_with_hidden >> shift) as u16;
            let round_bit = (mantissa_with_hidden >> (shift - 1)) & 1;
            if round_bit == 1 {
                half_mantissa = half_mantissa.wrapping_add(1);
            }
            return sign | half_mantissa;
        }

        let mut half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
        if ((mantissa >> 12) & 1) == 1 {
            half = half.wrapping_add(1);
        }
        half
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrisinfer-safetensors-test-{}-{}-{}",
                std::process::id(),
                timestamp,
                unique
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
