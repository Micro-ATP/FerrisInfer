use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Shape, Tensor};

use crate::kv_cache::{KvCacheConfig, KvCacheLayer, KvCacheStorage, KvCacheStorageKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvCachePageId(usize);

impl KvCachePageId {
    pub fn new(raw: usize) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCachePageInfo {
    page_id: KvCachePageId,
    start_position: usize,
    token_count: usize,
}

impl KvCachePageInfo {
    pub fn page_id(&self) -> KvCachePageId {
        self.page_id
    }

    pub fn start_position(&self) -> usize {
        self.start_position
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvBlockId(usize);

impl KvBlockId {
    pub fn new(raw: usize) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBlockTableEntry {
    logical_block_index: usize,
    block_id: KvBlockId,
    start_position: usize,
    token_count: usize,
}

impl KvBlockTableEntry {
    pub fn logical_block_index(&self) -> usize {
        self.logical_block_index
    }

    pub fn block_id(&self) -> KvBlockId {
        self.block_id
    }

    pub fn start_position(&self) -> usize {
        self.start_position
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(
        logical_block_index: usize,
        block_id: usize,
        start_position: usize,
        token_count: usize,
    ) -> Self {
        Self {
            logical_block_index,
            block_id: KvBlockId::new(block_id),
            start_position,
            token_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixHandle {
    token_count: usize,
    page_size: usize,
    layer_block_tables: Vec<Vec<KvBlockTableEntry>>,
}

impl PrefixHandle {
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn layer_count(&self) -> usize {
        self.layer_block_tables.len()
    }

    pub fn layer_block_table(&self, layer_index: usize) -> Option<&[KvBlockTableEntry]> {
        self.layer_block_tables.get(layer_index).map(Vec::as_slice)
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(
        token_count: usize,
        page_size: usize,
        layer_block_tables: Vec<Vec<KvBlockTableEntry>>,
    ) -> Self {
        Self {
            token_count,
            page_size,
            layer_block_tables,
        }
    }
}

#[derive(Debug, Clone)]
struct PagedKvCachePage {
    id: KvCachePageId,
    assigned_logical_block_index: Option<usize>,
    committed_tokens: usize,
    key: Tensor,
    value: Tensor,
}

impl PagedKvCachePage {
    fn new(page_id: KvCachePageId, page_size: usize, config: &KvCacheConfig) -> Result<Self> {
        let shape = Shape::from_slice(&[page_size, config.num_kv_heads, config.head_dim])?;
        Ok(Self {
            id: page_id,
            assigned_logical_block_index: None,
            committed_tokens: 0,
            key: Tensor::zeros(config.dtype, shape.clone())?,
            value: Tensor::zeros(config.dtype, shape)?,
        })
    }

    fn info(&self, page_size: usize) -> KvCachePageInfo {
        KvCachePageInfo {
            page_id: self.id,
            start_position: self
                .assigned_logical_block_index
                .unwrap_or(0)
                .saturating_mul(page_size),
            token_count: self.committed_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PagedKvCacheStorage {
    config: KvCacheConfig,
    page_size: usize,
    layer_views: Vec<KvCacheLayer>,
    layer_pages: Vec<Vec<PagedKvCachePage>>,
    layer_block_tables: Vec<Vec<Option<KvCachePageId>>>,
    layer_free_page_ids: Vec<Vec<KvCachePageId>>,
}

impl PagedKvCacheStorage {
    pub fn new(config: KvCacheConfig, page_size: usize) -> Result<Self> {
        validate_config(&config)?;
        if page_size == 0 {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "paged KV cache page size must be greater than zero",
            ));
        }

        let layer_shape = Shape::from_slice(&[
            config.max_sequence_length,
            config.num_kv_heads,
            config.head_dim,
        ])?;
        let page_count = config.max_sequence_length.div_ceil(page_size);
        let mut layer_views = Vec::with_capacity(config.num_layers);
        let mut layer_pages = Vec::with_capacity(config.num_layers);
        let mut layer_block_tables = Vec::with_capacity(config.num_layers);
        let mut layer_free_page_ids = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            layer_views.push(KvCacheLayer {
                key: Tensor::zeros(config.dtype, layer_shape.clone())?,
                value: Tensor::zeros(config.dtype, layer_shape.clone())?,
            });

            let mut pages = Vec::with_capacity(page_count);
            for page_index in 0..page_count {
                pages.push(PagedKvCachePage::new(
                    KvCachePageId::new(page_index),
                    page_size,
                    &config,
                )?);
            }
            layer_pages.push(pages);
            layer_block_tables.push(vec![None; page_count]);
            layer_free_page_ids.push(initial_free_page_ids(page_count));
        }

        Ok(Self {
            config,
            page_size,
            layer_views,
            layer_pages,
            layer_block_tables,
            layer_free_page_ids,
        })
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn page_count_per_layer(&self) -> usize {
        self.layer_pages.first().map(Vec::len).unwrap_or(0)
    }

    pub fn allocated_page_count(&self, layer_index: usize) -> Result<usize> {
        Ok(self
            .layer_block_assignments(layer_index)?
            .iter()
            .filter(|page_id| page_id.is_some())
            .count())
    }

    pub fn free_page_count(&self, layer_index: usize) -> Result<usize> {
        self.layer_free_page_ids
            .get(layer_index)
            .map(Vec::len)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!("KV cache layer index out of bounds: {layer_index}"),
                )
            })
    }

    pub fn page_infos(&self, layer_index: usize) -> Result<Vec<KvCachePageInfo>> {
        Ok(self
            .layer_pages(layer_index)?
            .iter()
            .map(|page| page.info(self.page_size))
            .collect())
    }

    pub fn block_table(
        &self,
        layer_index: usize,
        token_count: usize,
    ) -> Result<Vec<KvBlockTableEntry>> {
        validate_prefix_length(&self.config, token_count)?;

        let mut remaining_tokens = token_count;
        let logical_block_count = token_count.div_ceil(self.page_size);
        let block_assignments = self.layer_block_assignments(layer_index)?;
        let mut block_table = Vec::with_capacity(logical_block_count);

        for logical_block_index in 0..logical_block_count {
            let page_id = block_assignments
                .get(logical_block_index)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    FerrisError::new(
                        ErrorKind::Runtime,
                        format!(
                            "KV cache block table missing physical page for logical block {logical_block_index}"
                        ),
                    )
                })?;
            let page = self.layer_page(layer_index, page_id)?;
            let tokens_from_page = remaining_tokens.min(self.page_size);
            if page.committed_tokens < tokens_from_page {
                return Err(FerrisError::new(
                    ErrorKind::Runtime,
                    format!(
                        "KV cache block table touched uncommitted tokens in page {}",
                        page.id.raw()
                    ),
                ));
            }

            block_table.push(KvBlockTableEntry {
                logical_block_index,
                block_id: KvBlockId::new(page_id.raw()),
                start_position: logical_block_index * self.page_size,
                token_count: tokens_from_page,
            });
            remaining_tokens -= tokens_from_page;
        }

        if remaining_tokens != 0 {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "KV cache block table ran out of committed pages",
            ));
        }

        Ok(block_table)
    }

    pub fn capture_prefix_handle(&self, token_count: usize) -> Result<PrefixHandle> {
        let mut layer_block_tables = Vec::with_capacity(self.layer_count());
        for layer_index in 0..self.layer_count() {
            layer_block_tables.push(self.block_table(layer_index, token_count)?);
        }

        Ok(PrefixHandle {
            token_count,
            page_size: self.page_size,
            layer_block_tables,
        })
    }

    pub fn import_prefix_from(&mut self, source: &Self, handle: &PrefixHandle) -> Result<()> {
        self.validate_prefix_compatibility(source, handle)?;
        self.reset();

        let slot_width = self.slot_width();
        for (layer_index, entries) in handle.layer_block_tables.iter().enumerate() {
            for entry in entries {
                let source_page = source.layer_page_by_block_id(layer_index, entry.block_id)?;
                if source_page.committed_tokens < entry.token_count {
                    return Err(FerrisError::new(
                        ErrorKind::Runtime,
                        format!(
                            "KV cache prefix import touched uncommitted tokens in source block {}",
                            entry.block_id.raw()
                        ),
                    ));
                }

                let token_elements =
                    entry.token_count.checked_mul(slot_width).ok_or_else(|| {
                        FerrisError::new(ErrorKind::Runtime, "KV cache prefix import size overflow")
                    })?;
                let layer_element_offset = entry
                    .start_position
                    .checked_mul(slot_width)
                    .ok_or_else(|| {
                        FerrisError::new(
                            ErrorKind::Runtime,
                            "KV cache prefix import offset overflow",
                        )
                    })?;
                let key_values = source_page.key.as_f32_slice()?;
                let value_values = source_page.value.as_f32_slice()?;
                let dest_page_id =
                    self.allocate_page_for_logical_block(layer_index, entry.logical_block_index)?;

                {
                    let layer_view = self.layer_view_mut(layer_index)?;
                    layer_view.key.copy_from_f32_slice_at(
                        layer_element_offset,
                        &key_values[..token_elements],
                    )?;
                    layer_view.value.copy_from_f32_slice_at(
                        layer_element_offset,
                        &value_values[..token_elements],
                    )?;
                }

                let dest_page = self.layer_page_mut(layer_index, dest_page_id)?;
                dest_page
                    .key
                    .copy_from_f32_slice_at(0, &key_values[..token_elements])?;
                dest_page
                    .value
                    .copy_from_f32_slice_at(0, &value_values[..token_elements])?;
                dest_page.committed_tokens = entry.token_count;
            }
        }

        Ok(())
    }

    fn slot_width(&self) -> usize {
        self.config.num_kv_heads * self.config.head_dim
    }

    fn layer_view_mut(&mut self, layer_index: usize) -> Result<&mut KvCacheLayer> {
        self.layer_views.get_mut(layer_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!("KV cache layer index out of bounds: {layer_index}"),
            )
        })
    }

    fn layer_pages(&self, layer_index: usize) -> Result<&[PagedKvCachePage]> {
        self.layer_pages
            .get(layer_index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!("KV cache layer index out of bounds: {layer_index}"),
                )
            })
    }

    fn layer_block_assignments(&self, layer_index: usize) -> Result<&[Option<KvCachePageId>]> {
        self.layer_block_tables
            .get(layer_index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!("KV cache layer index out of bounds: {layer_index}"),
                )
            })
    }

    fn layer_page(&self, layer_index: usize, page_id: KvCachePageId) -> Result<&PagedKvCachePage> {
        self.layer_pages(layer_index)?
            .get(page_id.raw())
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "KV cache page id out of bounds for layer {layer_index}: {}",
                        page_id.raw()
                    ),
                )
            })
    }

    fn layer_page_by_block_id(
        &self,
        layer_index: usize,
        block_id: KvBlockId,
    ) -> Result<&PagedKvCachePage> {
        self.layer_page(layer_index, KvCachePageId::new(block_id.raw()))
    }

    fn layer_page_mut(
        &mut self,
        layer_index: usize,
        page_id: KvCachePageId,
    ) -> Result<&mut PagedKvCachePage> {
        self.layer_pages
            .get_mut(layer_index)
            .and_then(|pages| pages.get_mut(page_id.raw()))
            .ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "KV cache page id out of bounds for layer {layer_index}: {}",
                        page_id.raw()
                    ),
                )
            })
    }

    fn allocate_page_for_logical_block(
        &mut self,
        layer_index: usize,
        logical_block_index: usize,
    ) -> Result<KvCachePageId> {
        let existing_page_id = self
            .layer_block_tables
            .get(layer_index)
            .and_then(|block_table| block_table.get(logical_block_index))
            .copied()
            .flatten();
        if let Some(page_id) = existing_page_id {
            return Ok(page_id);
        }

        let page_id = {
            let free_page_ids = self
                .layer_free_page_ids
                .get_mut(layer_index)
                .ok_or_else(|| {
                    FerrisError::new(
                        ErrorKind::InvalidShape,
                        format!("KV cache layer index out of bounds: {layer_index}"),
                    )
                })?;
            free_page_ids.pop().ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("KV cache layer {layer_index} ran out of free pages"),
                )
            })?
        };

        {
            let block_table = self
                .layer_block_tables
                .get_mut(layer_index)
                .ok_or_else(|| {
                    FerrisError::new(
                        ErrorKind::InvalidShape,
                        format!("KV cache layer index out of bounds: {layer_index}"),
                    )
                })?;
            let slot = block_table.get_mut(logical_block_index).ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "KV cache logical block index out of bounds for layer {layer_index}: {logical_block_index}"
                    ),
                )
            })?;
            *slot = Some(page_id);
        }

        let page = self.layer_page_mut(layer_index, page_id)?;
        page.assigned_logical_block_index = Some(logical_block_index);
        page.committed_tokens = 0;
        Ok(page_id)
    }

    fn reset_page_assignments(&mut self) {
        let page_count = self.page_count_per_layer();

        for pages in &mut self.layer_pages {
            for page in pages {
                page.assigned_logical_block_index = None;
                page.committed_tokens = 0;
            }
        }

        for block_table in &mut self.layer_block_tables {
            for slot in block_table.iter_mut() {
                *slot = None;
            }
        }

        for free_page_ids in &mut self.layer_free_page_ids {
            free_page_ids.clear();
            free_page_ids.extend(initial_free_page_ids(page_count));
        }
    }

    fn validate_prefix_compatibility(&self, source: &Self, handle: &PrefixHandle) -> Result<()> {
        if self.page_size != source.page_size || self.page_size != handle.page_size {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "paged KV prefix import requires matching page sizes",
            ));
        }
        if self.config.num_layers != source.config.num_layers
            || self.config.num_layers != handle.layer_count()
            || self.config.num_kv_heads != source.config.num_kv_heads
            || self.config.head_dim != source.config.head_dim
            || self.config.max_sequence_length != source.config.max_sequence_length
            || self.config.dtype != source.config.dtype
        {
            return Err(FerrisError::new(
                ErrorKind::InvalidConfig,
                "paged KV prefix import requires matching cache configurations",
            ));
        }
        if handle.token_count > self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "paged KV prefix handle length {} exceeds capacity {}",
                    handle.token_count, self.config.max_sequence_length
                ),
            ));
        }

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
        if position >= self.config.max_sequence_length {
            return Err(FerrisError::new(
                ErrorKind::Runtime,
                "KV cache capacity exceeded",
            ));
        }

        let slot_width = self.slot_width();
        let logical_block_index = position / self.page_size;
        let page_offset = position % self.page_size;
        let page_element_offset = page_offset.checked_mul(slot_width).ok_or_else(|| {
            FerrisError::new(ErrorKind::Runtime, "KV cache page write offset overflow")
        })?;
        let layer_element_offset = position.checked_mul(slot_width).ok_or_else(|| {
            FerrisError::new(ErrorKind::Runtime, "KV cache write offset overflow")
        })?;
        let page_id = self.allocate_page_for_logical_block(layer_index, logical_block_index)?;

        {
            let layer_view = self.layer_view_mut(layer_index)?;
            layer_view
                .key
                .copy_from_tensor_f32_at(layer_element_offset, key)?;
            layer_view
                .value
                .copy_from_tensor_f32_at(layer_element_offset, value)?;
        }

        let page = self.layer_page_mut(layer_index, page_id)?;
        page.key.copy_from_tensor_f32_at(page_element_offset, key)?;
        page.value
            .copy_from_tensor_f32_at(page_element_offset, value)?;
        page.committed_tokens = page.committed_tokens.max(page_offset + 1);
        Ok(())
    }
}
impl KvCacheStorage for PagedKvCacheStorage {
    fn kind(&self) -> KvCacheStorageKind {
        KvCacheStorageKind::Paged
    }

    fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    fn layer_count(&self) -> usize {
        self.layer_views.len()
    }

    fn layer(&self, layer_index: usize) -> Result<&KvCacheLayer> {
        self.layer_views.get(layer_index).ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!("KV cache layer index out of bounds: {layer_index}"),
            )
        })
    }

    fn reset(&mut self) {
        self.reset_page_assignments();
    }

    fn write_f32(
        &mut self,
        layer_index: usize,
        position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        self.write_slot_f32(layer_index, position, key, value)
    }

    fn write_uncommitted_f32(
        &mut self,
        layer_index: usize,
        position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        self.write_slot_f32(layer_index, position, key, value)
    }

    fn write_sequence_uncommitted_f32(
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

        let slot_width = self.slot_width();
        let key_values = key.as_f32_slice()?;
        let value_values = value.as_f32_slice()?;

        let mut token_offset = 0usize;
        while token_offset < seq_len {
            let position = start_position + token_offset;
            let logical_block_index = position / self.page_size;
            let page_offset = position % self.page_size;
            let tokens_in_page = (self.page_size - page_offset).min(seq_len - token_offset);
            let source_start = token_offset.checked_mul(slot_width).ok_or_else(|| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    "KV cache sequence source offset overflow",
                )
            })?;
            let token_elements = tokens_in_page.checked_mul(slot_width).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "KV cache sequence source size overflow")
            })?;
            let source_end = source_start.checked_add(token_elements).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "KV cache sequence source end overflow")
            })?;
            let page_element_offset = page_offset.checked_mul(slot_width).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "KV cache page write offset overflow")
            })?;
            let layer_element_offset = position.checked_mul(slot_width).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "KV cache write offset overflow")
            })?;
            let page_id = self.allocate_page_for_logical_block(layer_index, logical_block_index)?;

            {
                let layer_view = self.layer_view_mut(layer_index)?;
                layer_view.key.copy_from_f32_slice_at(
                    layer_element_offset,
                    &key_values[source_start..source_end],
                )?;
                layer_view.value.copy_from_f32_slice_at(
                    layer_element_offset,
                    &value_values[source_start..source_end],
                )?;
            }

            let page = self.layer_page_mut(layer_index, page_id)?;
            page.key.copy_from_f32_slice_at(
                page_element_offset,
                &key_values[source_start..source_end],
            )?;
            page.value.copy_from_f32_slice_at(
                page_element_offset,
                &value_values[source_start..source_end],
            )?;
            page.committed_tokens = page.committed_tokens.max(page_offset + tokens_in_page);
            token_offset += tokens_in_page;
        }

        Ok(())
    }

    fn read_prefix_f32(&self, layer_index: usize, length: usize) -> Result<(Tensor, Tensor)> {
        validate_prefix_length(&self.config, length)?;

        let slot_width = self.slot_width();
        let prefix_elements = prefix_element_count(&self.config, length)?;
        let mut key_values = Vec::with_capacity(prefix_elements);
        let mut value_values = Vec::with_capacity(prefix_elements);

        for entry in self.block_table(layer_index, length)? {
            let page = self.layer_page_by_block_id(layer_index, entry.block_id())?;
            let page_elements = entry.token_count().checked_mul(slot_width).ok_or_else(|| {
                FerrisError::new(ErrorKind::Runtime, "KV cache paged prefix size overflow")
            })?;
            key_values.extend_from_slice(&page.key.as_f32_slice()?[..page_elements]);
            value_values.extend_from_slice(&page.value.as_f32_slice()?[..page_elements]);
        }

        let shape = Shape::from_slice(&[length, self.config.num_kv_heads, self.config.head_dim])?;
        Ok((
            Tensor::from_f32_vec(shape.clone(), key_values)?,
            Tensor::from_f32_vec(shape, value_values)?,
        ))
    }
}

fn initial_free_page_ids(page_count: usize) -> Vec<KvCachePageId> {
    (0..page_count).rev().map(KvCachePageId::new).collect()
}

fn validate_config(config: &KvCacheConfig) -> Result<()> {
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

    Ok(())
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
            format!("KV cache {label} tensor must have shape [{num_kv_heads}, {head_dim}]"),
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
            format!(
                "KV cache {label} sequence tensor must have shape [seq_len, {num_kv_heads}, {head_dim}]"
            ),
        ));
    }

    let dims = tensor.shape().dims();
    if dims[1] != num_kv_heads || dims[2] != head_dim {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!(
                "KV cache {label} sequence tensor must have shape [seq_len, {num_kv_heads}, {head_dim}]"
            ),
        ));
    }

    Ok(())
}

fn validate_prefix_length(config: &KvCacheConfig, length: usize) -> Result<()> {
    if length == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "KV cache prefix length must be greater than zero",
        ));
    }
    if length > config.max_sequence_length {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!(
                "KV cache prefix length {length} exceeds capacity {}",
                config.max_sequence_length
            ),
        ));
    }
    Ok(())
}

fn prefix_element_count(config: &KvCacheConfig, length: usize) -> Result<usize> {
    length
        .checked_mul(config.num_kv_heads)
        .and_then(|count| count.checked_mul(config.head_dim))
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "KV cache prefix size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvCache;

    #[test]
    fn paged_storage_implements_storage_trait() {
        let storage = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let storage_trait: &dyn KvCacheStorage = &storage;
        assert_eq!(storage_trait.kind(), KvCacheStorageKind::Paged);
        assert_eq!(storage.page_size(), 2);
        assert_eq!(storage.page_count_per_layer(), 4);
        assert_eq!(
            storage_trait.layer(0).unwrap().key().shape().dims(),
            &[8, 1, 2]
        );
    }

    #[test]
    fn paged_storage_write_sequence_round_trip_across_pages() {
        let mut storage = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 17.0, 18.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0, 19.0, 20.0],
        )
        .unwrap();

        storage
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();

        let (read_keys, read_values) = storage.read_prefix_f32(0, 5).unwrap();
        assert_eq!(
            read_keys.to_vec_f32().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 17.0, 18.0]
        );
        assert_eq!(
            read_values.to_vec_f32().unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0, 19.0, 20.0]
        );

        let page_infos = storage.page_infos(0).unwrap();
        assert_eq!(page_infos[0].token_count(), 2);
        assert_eq!(page_infos[1].token_count(), 2);
        assert_eq!(page_infos[2].token_count(), 1);
        assert_eq!(page_infos[3].token_count(), 0);
    }

    #[test]
    fn paged_storage_captures_prefix_handle_block_table() {
        let mut storage = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 17.0, 18.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0, 19.0, 20.0],
        )
        .unwrap();

        storage
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();

        let handle = storage.capture_prefix_handle(5).unwrap();
        let layer_block_table = handle.layer_block_table(0).unwrap();

        assert_eq!(handle.token_count(), 5);
        assert_eq!(handle.page_size(), 2);
        assert_eq!(layer_block_table.len(), 3);
        assert_eq!(layer_block_table[0].logical_block_index(), 0);
        assert_eq!(layer_block_table[0].block_id().raw(), 0);
        assert_eq!(layer_block_table[0].token_count(), 2);
        assert_eq!(layer_block_table[1].logical_block_index(), 1);
        assert_eq!(layer_block_table[1].block_id().raw(), 1);
        assert_eq!(layer_block_table[1].token_count(), 2);
        assert_eq!(layer_block_table[2].logical_block_index(), 2);
        assert_eq!(layer_block_table[2].block_id().raw(), 2);
        assert_eq!(layer_block_table[2].token_count(), 1);
    }

    #[test]
    fn paged_storage_imports_prefix_from_handle() {
        let mut source = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();
        let mut dest = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 17.0, 18.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[5, 1, 2]).unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0, 19.0, 20.0],
        )
        .unwrap();

        source
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();
        let handle = source.capture_prefix_handle(5).unwrap();
        dest.import_prefix_from(&source, &handle).unwrap();

        let (read_keys, read_values) = dest.read_prefix_f32(0, 5).unwrap();
        assert_eq!(
            read_keys.to_vec_f32().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 17.0, 18.0]
        );
        assert_eq!(
            read_values.to_vec_f32().unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 15.0, 16.0, 19.0, 20.0]
        );
        let page_infos = dest.page_infos(0).unwrap();
        assert_eq!(page_infos[0].token_count(), 2);
        assert_eq!(page_infos[1].token_count(), 2);
        assert_eq!(page_infos[2].token_count(), 1);
    }

    #[test]
    fn paged_storage_reuses_physical_page_ids_after_reset() {
        let mut storage = PagedKvCacheStorage::new(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 1, 2]).unwrap(),
            vec![5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();

        storage
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();
        assert_eq!(storage.allocated_page_count(0).unwrap(), 1);
        assert_eq!(storage.free_page_count(0).unwrap(), 3);

        storage.reset();
        storage
            .write_sequence_uncommitted_f32(0, 2, &keys, &values)
            .unwrap();

        let page_infos = storage.page_infos(0).unwrap();
        assert_eq!(storage.allocated_page_count(0).unwrap(), 1);
        assert_eq!(storage.free_page_count(0).unwrap(), 3);
        assert_eq!(page_infos[0].page_id().raw(), 0);
        assert_eq!(page_infos[0].start_position(), 2);
        assert_eq!(page_infos[0].token_count(), 2);
        assert_eq!(page_infos[1].token_count(), 0);
    }

    #[test]
    fn kv_cache_switches_to_paged_storage() {
        let mut cache = KvCache::new_paged(
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 2,
                max_sequence_length: 8,
                dtype: DType::F32,
            },
            2,
        )
        .unwrap();

        let keys = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 1, 2]).unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
        )
        .unwrap();
        let values = Tensor::from_f32_vec(
            Shape::from_slice(&[3, 1, 2]).unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0],
        )
        .unwrap();

        cache
            .write_sequence_uncommitted_f32(0, 0, &keys, &values)
            .unwrap();
        cache.advance(3).unwrap();

        let (read_keys, read_values) = cache.read_prefix_f32(0, 3).unwrap();
        assert_eq!(
            read_keys.to_vec_f32().unwrap(),
            vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
        );
        assert_eq!(
            read_values.to_vec_f32().unwrap(),
            vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0]
        );
        assert_eq!(cache.storage_kind(), KvCacheStorageKind::Paged);
    }
}
