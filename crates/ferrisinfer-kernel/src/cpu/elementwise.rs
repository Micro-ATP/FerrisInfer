use ferrisinfer_core::{FerrisError, Result, Tensor};

pub fn zero_tensor(tensor: &mut Tensor) -> Result<()> {
    tensor.as_bytes_mut()?.fill(0);
    Ok(())
}

pub fn add_f32(_lhs: &Tensor, _rhs: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(FerrisError::unsupported(
        "reference add_f32 kernel is not implemented yet",
    ))
}
