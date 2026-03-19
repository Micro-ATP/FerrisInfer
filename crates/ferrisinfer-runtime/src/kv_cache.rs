use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Shape, Tensor};

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_sequence_length: usize,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct KvCacheLayer {
    key: Tensor,
    value: Tensor,
}

impl KvCacheLayer {
    pub fn key(&self) -> &Tensor {
        &self.key
    }

    pub fn value(&self) -> &Tensor {
        &self.value
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    config: KvCacheConfig,
    layers: Vec<KvCacheLayer>,
    used_tokens: usize,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        if config.num_layers == 0
            || config.num_kv_heads == 0
            || config.head_dim == 0
            || config.max_sequence_length == 0
        {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "KV cache dimensions must be greater than zero",
            ));
        }

        if config.dtype != DType::F32 {
            return Err(FerrisError::unsupported(
                "KV cache currently supports f32 storage only",
            ));
        }

        let layer_shape = Shape::from_slice(&[
            config.max_sequence_length,
            config.num_kv_heads,
            config.head_dim,
        ])?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(KvCacheLayer {
                key: Tensor::zeros(config.dtype, layer_shape.clone())?,
                value: Tensor::zeros(config.dtype, layer_shape.clone())?,
            });
        }

        Ok(Self {
            config,
            layers,
            used_tokens: 0,
        })
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer_index: usize) -> Result<&KvCacheLayer> {
        self.layers.get(layer_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!("KV cache layer index out of bounds: {layer_index}"),
            )
        })
    }

    pub fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.config
            .max_sequence_length
            .saturating_sub(self.used_tokens)
    }

    pub fn reset(&mut self) {
        self.used_tokens = 0;
    }

    pub fn reserve_slot(&mut self) -> Result<usize> {
        let position = self.used_tokens;
        self.advance(1)?;
        Ok(position)
    }

    pub fn advance(&mut self, tokens: usize) -> Result<()> {
        let next = self
            .used_tokens
            .checked_add(tokens)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache position overflow"))?;

        if next > self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "KV cache capacity exceeded",
            ));
        }

        self.used_tokens = next;
        Ok(())
    }

    pub fn write_f32(
        &mut self,
        layer_index: usize,
        position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        self.ensure_position_committed(position)?;
        self.write_slot_f32(layer_index, position, key, value)
    }

    pub fn write_uncommitted_f32(
        &mut self,
        layer_index: usize,
        position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        if position > self.used_tokens {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "KV cache position {position} is beyond the next writable slot {}; committed tokens {}",
                    self.used_tokens,
                    self.used_tokens
                ),
            ));
        }

        self.write_slot_f32(layer_index, position, key, value)
    }

    pub fn write_sequence_uncommitted_f32(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        validate_sequence_tensor(key, self.config.num_kv_heads, self.config.head_dim, "key")?;
        validate_sequence_tensor(
            value,
            self.config.num_kv_heads,
            self.config.head_dim,
            "value",
        )?;

        if key.shape().dims() != value.shape().dims() {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "KV cache key/value sequence tensors must have matching shapes",
            ));
        }

        if start_position > self.used_tokens {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "KV cache sequence start {start_position} is beyond committed tokens {}",
                    self.used_tokens
                ),
            ));
        }

        let seq_len = key.shape().dims()[0];
        let end_position = start_position.checked_add(seq_len).ok_or_else(|| {
            FerrisError::new(ErrorKind::Runtime, "KV cache sequence end overflow")
        })?;
        if end_position > self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "KV cache capacity exceeded",
            ));
        }

        let slot_width = self.config.num_kv_heads * self.config.head_dim;
        let element_offset = start_position
            .checked_mul(slot_width)
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache write offset overflow"))?;
        let layer = self.layers.get_mut(layer_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!("KV cache layer index out of bounds: {layer_index}"),
            )
        })?;

        layer.key.copy_from_tensor_f32_at(element_offset, key)?;
        layer.value.copy_from_tensor_f32_at(element_offset, value)?;
        Ok(())
    }

    fn write_slot_f32(
        &mut self,
        layer_index: usize,
        position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        validate_slot_tensor(key, self.config.num_kv_heads, self.config.head_dim, "key")?;
        validate_slot_tensor(
            value,
            self.config.num_kv_heads,
            self.config.head_dim,
            "value",
        )?;

        let slot_width = self.config.num_kv_heads * self.config.head_dim;
        let layer = self.layers.get_mut(layer_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!("KV cache layer index out of bounds: {layer_index}"),
            )
        })?;
        let start = position.checked_mul(slot_width).ok_or_else(|| {
            FerrisError::new(ErrorKind::Runtime, "KV cache write offset overflow")
        })?;

        layer.key.copy_from_tensor_f32_at(start, key)?;
        layer.value.copy_from_tensor_f32_at(start, value)?;
        Ok(())
    }

    pub fn read_prefix_f32(&self, layer_index: usize, length: usize) -> Result<(Tensor, Tensor)> {
        if length == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                "KV cache prefix length must be greater than zero",
            ));
        }

        if length > self.used_tokens {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "KV cache prefix length {length} exceeds committed tokens {}",
                    self.used_tokens
                ),
            ));
        }

        let layer = self.layer(layer_index)?;
        let prefix_elements = length
            .checked_mul(self.config.num_kv_heads)
            .and_then(|count| count.checked_mul(self.config.head_dim))
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache prefix size overflow"))?;
        let shape = Shape::from_slice(&[length, self.config.num_kv_heads, self.config.head_dim])?;
        let prefix_bytes = prefix_elements
            .checked_mul(DType::F32.size_in_bytes())
            .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache prefix byte size overflow"))?;

        Ok((
            Tensor::from_owned_bytes(DType::F32, shape.clone(), layer.key.as_bytes()[..prefix_bytes].to_vec())?,
            Tensor::from_owned_bytes(DType::F32, shape, layer.value.as_bytes()[..prefix_bytes].to_vec())?,
        ))
    }

    fn ensure_position_committed(&self, position: usize) -> Result<()> {
        if position >= self.used_tokens {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "KV cache position {position} has not been reserved yet; committed tokens {}",
                    self.used_tokens
                ),
            ));
        }

        Ok(())
    }
}

fn validate_slot_tensor(
    tensor: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    label: &str,
) -> Result<()> {
    tensor.ensure_dtype(DType::F32)?;
    tensor.ensure_contiguous()?;

    if tensor.shape().dims() != [num_kv_heads, head_dim] {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!("KV cache {label} tensor must have shape [{num_kv_heads}, {head_dim}]",),
        ));
    }

    Ok(())
}

fn validate_sequence_tensor(
    tensor: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    label: &str,
) -> Result<()> {
    tensor.ensure_dtype(DType::F32)?;
    tensor.ensure_contiguous()?;

    if tensor.shape().rank() != 3 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!("KV cache {label} sequence tensor must have shape [seq_len, {num_kv_heads}, {head_dim}]"),
        ));
    }

    let dims = tensor.shape().dims();
    if dims[1] != num_kv_heads || dims[2] != head_dim {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!("KV cache {label} sequence tensor must have shape [seq_len, {num_kv_heads}, {head_dim}]"),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_allocates_layer_storage() {
        let cache = KvCache::new(KvCacheConfig {
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        assert_eq!(cache.layer_count(), 2);
        assert_eq!(cache.layer(0).unwrap().key().shape().dims(), &[8, 1, 2]);
        assert_eq!(cache.layer(0).unwrap().value().shape().dims(), &[8, 1, 2]);
    }

    #[test]
    fn kv_cache_write_and_read_prefix_round_trip() {
        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        let first = cache.reserve_slot().unwrap();
        let second = cache.reserve_slot().unwrap();
        assert_eq!(first, 0);
        assert_eq!(second, 1);

        let key0 =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![1.0, 2.0]).unwrap();
        let value0 =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![3.0, 4.0]).unwrap();
        let key1 =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![5.0, 6.0]).unwrap();
        let value1 =
            Tensor::from_f32_vec(Shape::from_slice(&[1, 2]).unwrap(), vec![7.0, 8.0]).unwrap();

        cache.write_f32(0, first, &key0, &value0).unwrap();
        cache.write_f32(0, second, &key1, &value1).unwrap();

        let (keys, values) = cache.read_prefix_f32(0, 2).unwrap();
        assert_eq!(keys.shape().dims(), &[2, 1, 2]);
        assert_eq!(values.shape().dims(), &[2, 1, 2]);
        assert_eq!(keys.to_vec_f32().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(values.to_vec_f32().unwrap(), vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn kv_cache_reset_restores_capacity() {
        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        cache.advance(3).unwrap();
        assert_eq!(cache.used_tokens(), 3);
        assert_eq!(cache.remaining_tokens(), 5);

        cache.reset();
        assert_eq!(cache.used_tokens(), 0);
        assert_eq!(cache.remaining_tokens(), 8);
    }

    #[test]
    fn kv_cache_write_sequence_uncommitted_round_trip() {
        let mut cache = KvCache::new(KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
            dtype: DType::F32,
        })
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 2.0, 5.0, 6.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![3.0, 4.0, 7.0, 8.0],
        )
        .unwrap();

        cache
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();
        cache.advance(2).unwrap();

        let (read_keys, read_values) = cache.read_prefix_f32(0, 2).unwrap();
        assert_eq!(read_keys.to_vec_f32().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(read_values.to_vec_f32().unwrap(), vec![3.0, 4.0, 7.0, 8.0]);
    }
}
