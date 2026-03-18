use ferrisinfer_core::DeviceKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStep {
    PrepareWeights,
    Embedding,
    Norm,
    Rope,
    Matmul,
    Attention,
    Mlp,
    KvCacheRead,
    KvCacheWrite,
    Sample,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub mode: ExecutionMode,
    pub target_device: DeviceKind,
    pub steps: Vec<ExecutionStep>,
}

impl ExecutionPlan {
    pub fn prefill(target_device: DeviceKind) -> Self {
        Self {
            mode: ExecutionMode::Prefill,
            target_device,
            steps: vec![
                ExecutionStep::PrepareWeights,
                ExecutionStep::Embedding,
                ExecutionStep::Norm,
                ExecutionStep::Rope,
                ExecutionStep::Matmul,
                ExecutionStep::Attention,
                ExecutionStep::KvCacheWrite,
                ExecutionStep::Mlp,
                ExecutionStep::Sample,
            ],
        }
    }

    pub fn decode(target_device: DeviceKind) -> Self {
        Self {
            mode: ExecutionMode::Decode,
            target_device,
            steps: vec![
                ExecutionStep::Embedding,
                ExecutionStep::KvCacheRead,
                ExecutionStep::Rope,
                ExecutionStep::Matmul,
                ExecutionStep::Attention,
                ExecutionStep::KvCacheWrite,
                ExecutionStep::Mlp,
                ExecutionStep::Sample,
            ],
        }
    }
}
