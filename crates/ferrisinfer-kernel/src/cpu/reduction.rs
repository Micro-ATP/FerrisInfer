use ferrisinfer_core::{FerrisError, Result, Tensor};

pub fn rms_norm_f32(_input: &Tensor, _weight: &Tensor, _out: &mut Tensor, _eps: f32) -> Result<()> {
    Err(FerrisError::unsupported(
        "reference rms_norm_f32 kernel is not implemented yet",
    ))
}

pub fn softmax_f32(_input: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(FerrisError::unsupported(
        "reference softmax_f32 kernel is not implemented yet",
    ))
}
