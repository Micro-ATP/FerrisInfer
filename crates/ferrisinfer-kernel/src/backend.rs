use ferrisinfer_core::{DeviceKind, ExecutionConfig, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub simd: bool,
    pub multithreaded: bool,
    pub quantized_kernels: bool,
    pub device_memory: bool,
    pub graph_capture: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendAvailability {
    pub available: bool,
    pub reason: Option<&'static str>,
}

impl BackendAvailability {
    pub const fn available() -> Self {
        Self {
            available: true,
            reason: None,
        }
    }

    pub const fn unavailable(reason: &'static str) -> Self {
        Self {
            available: false,
            reason: Some(reason),
        }
    }
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn device(&self) -> DeviceKind;
    fn config(&self) -> &ExecutionConfig;
    fn capabilities(&self) -> BackendCapabilities;
    fn availability(&self) -> BackendAvailability;
    fn fill_zero(&self, tensor: &mut Tensor) -> Result<()>;
}
