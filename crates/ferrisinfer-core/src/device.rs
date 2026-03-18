#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    WebGpu,
}

#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub preferred_device: DeviceKind,
    pub threads: usize,
    pub use_simd: bool,
    pub scratch_bytes: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);

        Self {
            preferred_device: DeviceKind::Cpu,
            threads,
            use_simd: true,
            scratch_bytes: 0,
        }
    }
}
