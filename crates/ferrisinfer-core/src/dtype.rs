#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    U8,
    Q4_0,
    Q8_0,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::U8 => "u8",
            Self::Q4_0 => "q4_0",
            Self::Q8_0 => "q8_0",
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::Q4_0 | Self::Q8_0 => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0)
    }
}
