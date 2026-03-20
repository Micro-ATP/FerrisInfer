use ferrisinfer_core::{DType, ErrorKind, FerrisError, Result, Tensor};

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn chat_default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1 || self.top_p <= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSample {
    pub token_id: u32,
    pub probability: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerState {
    rng_state: u64,
}

impl SamplerState {
    pub fn new(seed: u64) -> Self {
        let rng_state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { rng_state }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let value = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let upper = (value >> 40) as u32;
        upper as f32 / (1u32 << 24) as f32
    }
}

pub fn argmax_last_token(logits: &Tensor) -> Result<TokenSample> {
    logits.ensure_dtype(DType::F32)?;
    logits.ensure_contiguous()?;

    if logits.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "argmax_last_token requires logits with shape [seq_len, vocab_size]",
        ));
    }

    let dims = logits.shape().dims();
    let seq_len = dims[0];
    let vocab_size = dims[1];
    if seq_len == 0 || vocab_size == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "argmax_last_token requires non-empty logits",
        ));
    }

    let values = logits.as_f32_slice()?;
    let row = &values[(seq_len - 1) * vocab_size..seq_len * vocab_size];

    let (best_index, &best_logit) = row
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "failed to scan logits row"))?;

    let max_logit = best_logit;
    let mut denominator = 0.0f32;
    for &logit in row {
        denominator += (logit - max_logit).exp();
    }

    Ok(TokenSample {
        token_id: u32::try_from(best_index).map_err(|_| {
            FerrisError::new(ErrorKind::Runtime, "best token index does not fit into u32")
        })?,
        probability: 1.0 / denominator,
    })
}

pub fn sample_last_token(
    logits: &Tensor,
    config: &SamplerConfig,
    state: &mut SamplerState,
) -> Result<TokenSample> {
    if config.is_greedy() {
        return argmax_last_token(logits);
    }

    logits.ensure_dtype(DType::F32)?;
    logits.ensure_contiguous()?;

    if logits.shape().rank() != 2 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "sample_last_token requires logits with shape [seq_len, vocab_size]",
        ));
    }

    let dims = logits.shape().dims();
    let seq_len = dims[0];
    let vocab_size = dims[1];
    if seq_len == 0 || vocab_size == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "sample_last_token requires non-empty logits",
        ));
    }

    let values = logits.as_f32_slice()?;
    let row = &values[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    let top_k = config.top_k.max(1).min(vocab_size);

    let mut candidates = Vec::with_capacity(top_k);
    if top_k == vocab_size {
        for (index, &logit) in row.iter().enumerate() {
            candidates.push((index, logit));
        }
    } else {
        for (index, &logit) in row.iter().enumerate() {
            if candidates.len() < top_k {
                candidates.push((index, logit));
                continue;
            }

            let (min_pos, min_logit) = candidates
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.1.total_cmp(&right.1))
                .map(|(pos, candidate)| (pos, candidate.1))
                .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "failed to scan top-k"))?;

            if logit > min_logit {
                candidates[min_pos] = (index, logit);
            }
        }
    }

    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));

    let temperature = config.temperature.max(f32::EPSILON);
    let scaled_max_logit = candidates
        .first()
        .map(|candidate| candidate.1 / temperature)
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "sampling candidates are empty"))?;

    let mut weighted = Vec::with_capacity(candidates.len());
    let mut total_weight = 0.0f32;
    for (token_index, logit) in candidates {
        let weight = ((logit / temperature) - scaled_max_logit).exp();
        total_weight += weight;
        weighted.push((token_index, weight));
    }

    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(FerrisError::new(
            ErrorKind::Runtime,
            "failed to build a finite sampling distribution",
        ));
    }

    let top_p = config.top_p.clamp(0.0, 1.0);
    let cutoff_weight = if top_p >= 1.0 {
        total_weight
    } else {
        total_weight * top_p
    };

    let mut retained_weight = 0.0f32;
    let mut retained_len = 0usize;
    for (_, weight) in &weighted {
        retained_weight += *weight;
        retained_len += 1;
        if retained_weight >= cutoff_weight {
            break;
        }
    }
    retained_len = retained_len.max(1);
    weighted.truncate(retained_len);

    let retained_total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    let mut threshold = state.next_f32() * retained_total;
    let mut selected = weighted[0];
    for candidate in &weighted {
        threshold -= candidate.1;
        if threshold <= 0.0 {
            selected = *candidate;
            break;
        }
    }

    Ok(TokenSample {
        token_id: u32::try_from(selected.0).map_err(|_| {
            FerrisError::new(
                ErrorKind::Runtime,
                "sampled token index does not fit into u32",
            )
        })?,
        probability: selected.1 / retained_total,
    })
}

#[cfg(test)]
mod tests {
    use ferrisinfer_core::Shape;

    use super::*;

    #[test]
    fn argmax_last_token_returns_best_last_row_token() {
        let logits = Tensor::from_f32_vec(
            Shape::from_slice(&[2, 3]).unwrap(),
            vec![0.0, 1.0, 2.0, 1.0, 3.0, 2.0],
        )
        .unwrap();

        let sample = argmax_last_token(&logits).unwrap();

        assert_eq!(sample.token_id, 1);
        assert!(sample.probability > 0.0 && sample.probability <= 1.0);
    }

    #[test]
    fn sample_last_token_respects_top_k_sampling() {
        let logits = Tensor::from_f32_vec(
            Shape::from_slice(&[1, 4]).unwrap(),
            vec![1.0, 3.0, 2.0, 0.5],
        )
        .unwrap();
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 123,
        };
        let mut state = SamplerState::new(config.seed);

        let sample = sample_last_token(&logits, &config, &mut state).unwrap();

        assert!(sample.token_id == 1 || sample.token_id == 2);
        assert!(sample.probability > 0.0 && sample.probability <= 1.0);
    }
}
