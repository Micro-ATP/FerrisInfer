use ferrisinfer_core::{FerrisError, Result, Tensor};

pub fn matmul_f32(_lhs: &Tensor, _rhs: &Tensor, _out: &mut Tensor) -> Result<()> {
    Err(FerrisError::unsupported(
        "reference matmul_f32 kernel is not implemented yet",
    ))
}
