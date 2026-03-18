use ferrisinfer_core::{DeviceKind, ExecutionConfig, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub simd: bool,
    pub multithreaded: bool,
    pub quantized_kernels: bool,
    pub device_memory: bool,
    pub graph_capture: bool,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn device(&self) -> DeviceKind;
    fn config(&self) -> &ExecutionConfig;
    fn capabilities(&self) -> BackendCapabilities;
    fn fill_zero(&self, tensor: &mut Tensor) -> Result<()>;
}
