use ferrisinfer_core::{DeviceKind, ExecutionConfig, FerrisError, Result, Tensor};

use crate::backend::{Backend, BackendCapabilities};

#[derive(Debug, Clone)]
pub struct GpuBackend {
    config: ExecutionConfig,
    device: DeviceKind,
}

impl GpuBackend {
    pub fn new(device: DeviceKind, config: ExecutionConfig) -> Self {
        Self { config, device }
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        match self.device {
            DeviceKind::Cuda => "cuda",
            DeviceKind::Metal => "metal",
            DeviceKind::Vulkan => "vulkan",
            DeviceKind::WebGpu => "webgpu",
            DeviceKind::Cpu => "cpu",
        }
    }

    fn device(&self) -> DeviceKind {
        self.device
    }

    fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            simd: false,
            multithreaded: true,
            quantized_kernels: false,
            device_memory: true,
            graph_capture: false,
        }
    }

    fn fill_zero(&self, _tensor: &mut Tensor) -> Result<()> {
        Err(FerrisError::unsupported(
            "GPU tensor execution is not implemented yet",
        ))
    }
}
