#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureKind {
    Llama,
    Mistral,
    Qwen2,
    Gemma,
    CustomDecoder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    Gelu,
    Silu,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayout {
    SeparateQkv,
    PackedQkv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingKind {
    None,
    Linear,
    DynamicNtK,
    Yarn,
}

#[derive(Debug, Clone)]
pub struct NormSpec {
    pub kind: NormKind,
    pub epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub theta: f32,
    pub scaling: RopeScalingKind,
    pub scaling_factor: f32,
    pub rotary_dims: usize,
}

#[derive(Debug, Clone)]
pub struct AttentionSpec {
    pub layout: AttentionLayout,
    pub causal: bool,
    pub use_qk_norm: bool,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct MlpSpec {
    pub hidden_act: ActivationKind,
    pub gated: bool,
}
