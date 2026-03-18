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
        Self {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSample {
    pub token_id: u32,
    pub probability: f32,
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

    let values = logits.to_vec_f32()?;
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
}
