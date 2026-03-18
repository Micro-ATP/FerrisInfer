pub mod elementwise;
pub mod matmul;
pub mod reduction;

use ferrisinfer_core::{DeviceKind, ExecutionConfig, Result, Tensor};

use crate::backend::{Backend, BackendCapabilities};

#[derive(Debug, Clone)]
pub struct CpuBackend {
    config: ExecutionConfig,
}

impl CpuBackend {
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            simd: self.config.use_simd,
            multithreaded: self.config.threads > 1,
            quantized_kernels: false,
            device_memory: false,
            graph_capture: false,
        }
    }

    fn fill_zero(&self, tensor: &mut Tensor) -> Result<()> {
        elementwise::zero_tensor(tensor)
    }
}
