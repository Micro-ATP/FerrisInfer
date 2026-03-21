#[cfg(all(unix, not(target_os = "macos")))]
use std::ffi::CString;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Arc;

use ferrisinfer_core::{
    DType, DeviceKind, ErrorKind, ExecutionConfig, FerrisError, Result, Shape, Tensor,
};

use crate::backend::{Backend, BackendAvailability, BackendCapabilities};

const CUDA_SUCCESS: i32 = 0;
const CUDA_DEVICE_NAME_BYTES: usize = 256;

type CuDevice = c_int;
type CuContextHandle = *mut c_void;
type CuDevicePtr = u64;
type CuInitFn = unsafe extern "C" fn(flags: u32) -> c_int;
type CuDriverGetVersionFn = unsafe extern "C" fn(driver_version: *mut c_int) -> c_int;
type CuDeviceGetCountFn = unsafe extern "C" fn(count: *mut c_int) -> c_int;
type CuDeviceGetFn = unsafe extern "C" fn(device: *mut CuDevice, ordinal: c_int) -> c_int;
type CuDeviceGetNameFn =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, device: CuDevice) -> c_int;
type CuDeviceComputeCapabilityFn =
    unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int, device: CuDevice) -> c_int;
type CuDeviceTotalMemFn = unsafe extern "C" fn(bytes: *mut usize, device: CuDevice) -> c_int;
type CuDevicePrimaryCtxRetainFn =
    unsafe extern "C" fn(context: *mut CuContextHandle, device: CuDevice) -> c_int;
type CuDevicePrimaryCtxReleaseFn = unsafe extern "C" fn(device: CuDevice) -> c_int;
type CuCtxPushCurrentFn = unsafe extern "C" fn(context: CuContextHandle) -> c_int;
type CuCtxPopCurrentFn = unsafe extern "C" fn(context: *mut CuContextHandle) -> c_int;
type CuCtxSynchronizeFn = unsafe extern "C" fn() -> c_int;
type CuMemAllocFn = unsafe extern "C" fn(device_ptr: *mut CuDevicePtr, bytes: usize) -> c_int;
type CuMemFreeFn = unsafe extern "C" fn(device_ptr: CuDevicePtr) -> c_int;
type CuMemcpyHtoDFn =
    unsafe extern "C" fn(dst_device: CuDevicePtr, src_host: *const c_void, bytes: usize) -> c_int;
type CuMemcpyDtoHFn =
    unsafe extern "C" fn(dst_host: *mut c_void, src_device: CuDevicePtr, bytes: usize) -> c_int;
type CuMemsetD8Fn = unsafe extern "C" fn(dst_device: CuDevicePtr, value: u8, count: usize) -> c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvidiaCudaDriverVersion {
    raw: i32,
    major: u32,
    minor: u32,
}

impl NvidiaCudaDriverVersion {
    pub fn from_raw(raw: i32) -> Self {
        let normalized = raw.max(0);
        Self {
            raw: normalized,
            major: (normalized / 1000) as u32,
            minor: ((normalized % 1000) / 10) as u32,
        }
    }

    pub fn raw(&self) -> i32 {
        self.raw
    }

    pub fn major(&self) -> u32 {
        self.major
    }

    pub fn minor(&self) -> u32 {
        self.minor
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaCudaDeviceInfo {
    ordinal: usize,
    name: String,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    total_memory_bytes: u64,
    raw_device: CuDevice,
}

impl NvidiaCudaDeviceInfo {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn compute_capability_major(&self) -> u32 {
        self.compute_capability_major
    }

    pub fn compute_capability_minor(&self) -> u32 {
        self.compute_capability_minor
    }

    pub fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    fn raw_device(&self) -> CuDevice {
        self.raw_device
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaCudaProbe {
    availability: BackendAvailability,
    detail: Option<String>,
    driver_library: Option<&'static str>,
    driver_version: Option<NvidiaCudaDriverVersion>,
    devices: Vec<NvidiaCudaDeviceInfo>,
}

impl NvidiaCudaProbe {
    fn available(
        driver_library: &'static str,
        driver_version: Option<NvidiaCudaDriverVersion>,
        devices: Vec<NvidiaCudaDeviceInfo>,
        detail: Option<String>,
    ) -> Self {
        Self {
            availability: BackendAvailability::available(),
            detail,
            driver_library: Some(driver_library),
            driver_version,
            devices,
        }
    }

    fn unavailable(
        reason: &'static str,
        detail: Option<String>,
        driver_library: Option<&'static str>,
        driver_version: Option<NvidiaCudaDriverVersion>,
    ) -> Self {
        Self {
            availability: BackendAvailability::unavailable(reason),
            detail,
            driver_library,
            driver_version,
            devices: Vec::new(),
        }
    }

    pub fn availability(&self) -> BackendAvailability {
        self.availability
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub fn driver_library(&self) -> Option<&'static str> {
        self.driver_library
    }

    pub fn driver_version(&self) -> Option<&NvidiaCudaDriverVersion> {
        self.driver_version.as_ref()
    }

    pub fn devices(&self) -> &[NvidiaCudaDeviceInfo] {
        &self.devices
    }
}

#[derive(Clone)]
pub struct NvidiaCudaRuntime {
    driver_library: &'static str,
    driver_version: Option<NvidiaCudaDriverVersion>,
    detail: Option<String>,
    devices: Vec<NvidiaCudaDeviceInfo>,
    driver: Arc<NvidiaCudaDriverApi>,
}

impl std::fmt::Debug for NvidiaCudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvidiaCudaRuntime")
            .field("driver_library", &self.driver_library)
            .field("driver_version", &self.driver_version)
            .field("detail", &self.detail)
            .field("devices", &self.devices)
            .finish()
    }
}

impl NvidiaCudaRuntime {
    pub fn probe(&self) -> NvidiaCudaProbe {
        NvidiaCudaProbe::available(
            self.driver_library,
            self.driver_version,
            self.devices.clone(),
            self.detail.clone(),
        )
    }

    pub fn driver_library(&self) -> &'static str {
        self.driver_library
    }

    pub fn driver_version(&self) -> Option<&NvidiaCudaDriverVersion> {
        self.driver_version.as_ref()
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    pub fn devices(&self) -> &[NvidiaCudaDeviceInfo] {
        &self.devices
    }

    pub fn create_context(&self, ordinal: usize) -> Result<NvidiaCudaContext> {
        let device = self.devices.get(ordinal).cloned().ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidShape,
                format!(
                    "NVIDIA CUDA device ordinal {ordinal} is out of range; discovered {} device(s)",
                    self.devices.len()
                ),
            )
        })?;
        let raw = self
            .driver
            .retain_primary_context(device.raw_device())
            .map_err(cuda_runtime_error)?;

        Ok(NvidiaCudaContext {
            inner: Arc::new(NvidiaCudaContextInner {
                driver: Arc::clone(&self.driver),
                device,
                raw,
            }),
        })
    }
}

#[derive(Debug, Clone)]
pub struct NvidiaCudaBackend {
    config: ExecutionConfig,
    probe: NvidiaCudaProbe,
    runtime: Option<NvidiaCudaRuntime>,
}

impl NvidiaCudaBackend {
    pub fn new(mut config: ExecutionConfig) -> Self {
        config.preferred_device = DeviceKind::Cuda;
        let (probe, runtime) = match try_load_nvidia_cuda_runtime() {
            Ok(runtime) => {
                let probe = runtime.probe();
                (probe, Some(runtime))
            }
            Err(probe) => (probe, None),
        };

        Self {
            config,
            probe,
            runtime,
        }
    }

    pub fn probe(&self) -> &NvidiaCudaProbe {
        &self.probe
    }

    pub fn runtime(&self) -> Option<&NvidiaCudaRuntime> {
        self.runtime.as_ref()
    }

    pub fn create_context(&self, ordinal: usize) -> Result<NvidiaCudaContext> {
        let Some(runtime) = self.runtime() else {
            let reason = self
                .probe
                .availability()
                .reason
                .unwrap_or("NVIDIA CUDA runtime is unavailable");
            return Err(FerrisError::unsupported(reason));
        };

        runtime.create_context(ordinal)
    }
}

impl Default for NvidiaCudaBackend {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl Backend for NvidiaCudaBackend {
    fn name(&self) -> &'static str {
        "nvidia-cuda"
    }

    fn device(&self) -> DeviceKind {
        DeviceKind::Cuda
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

    fn availability(&self) -> BackendAvailability {
        self.probe.availability()
    }

    fn fill_zero(&self, _tensor: &mut Tensor) -> Result<()> {
        Err(FerrisError::unsupported(
            "NVIDIA CUDA tensor execution is not implemented yet",
        ))
    }
}

#[derive(Clone)]
pub struct NvidiaCudaContext {
    inner: Arc<NvidiaCudaContextInner>,
}

impl std::fmt::Debug for NvidiaCudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvidiaCudaContext")
            .field("device", &self.inner.device)
            .field("raw", &self.inner.raw)
            .finish()
    }
}

impl NvidiaCudaContext {
    pub fn device(&self) -> &NvidiaCudaDeviceInfo {
        &self.inner.device
    }

    pub fn allocate(&self, len_bytes: usize) -> Result<NvidiaCudaDeviceBuffer> {
        validate_buffer_len(len_bytes)?;
        let raw = self
            .inner
            .with_current(|driver| driver.mem_alloc(len_bytes))?;

        Ok(NvidiaCudaDeviceBuffer {
            context: Arc::clone(&self.inner),
            raw,
            len_bytes,
        })
    }

    pub fn upload_bytes(&self, host_bytes: &[u8]) -> Result<NvidiaCudaDeviceBuffer> {
        let mut buffer = self.allocate(host_bytes.len())?;
        buffer.copy_from_host(host_bytes)?;
        Ok(buffer)
    }

    pub fn allocate_tensor(&self, dtype: DType, shape: Shape) -> Result<NvidiaCudaTensor> {
        let len_bytes = checked_tensor_byte_len(dtype, &shape)?;
        let buffer = self.allocate(len_bytes)?;
        Ok(NvidiaCudaTensor {
            dtype,
            shape,
            buffer,
        })
    }

    pub fn zeros_tensor(&self, dtype: DType, shape: Shape) -> Result<NvidiaCudaTensor> {
        let mut tensor = self.allocate_tensor(dtype, shape)?;
        tensor.fill_zero()?;
        Ok(tensor)
    }

    pub fn upload_tensor(&self, tensor: &Tensor) -> Result<NvidiaCudaTensor> {
        tensor.ensure_contiguous()?;
        let mut device_tensor = self.allocate_tensor(tensor.dtype(), tensor.shape().clone())?;
        device_tensor.copy_from_tensor(tensor)?;
        Ok(device_tensor)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.inner.with_current(|driver| driver.ctx_synchronize())
    }
}

struct NvidiaCudaContextInner {
    driver: Arc<NvidiaCudaDriverApi>,
    device: NvidiaCudaDeviceInfo,
    raw: CuContextHandle,
}

impl NvidiaCudaContextInner {
    fn with_current<T>(
        &self,
        f: impl FnOnce(&NvidiaCudaDriverApi) -> std::result::Result<T, String>,
    ) -> Result<T> {
        let _guard = NvidiaCudaCurrentGuard::push(self.driver.as_ref(), self.raw)
            .map_err(cuda_runtime_error)?;
        f(self.driver.as_ref()).map_err(cuda_runtime_error)
    }
}

impl Drop for NvidiaCudaContextInner {
    fn drop(&mut self) {
        let _ = self
            .driver
            .release_primary_context(self.device.raw_device());
    }
}

pub struct NvidiaCudaDeviceBuffer {
    context: Arc<NvidiaCudaContextInner>,
    raw: CuDevicePtr,
    len_bytes: usize,
}

impl std::fmt::Debug for NvidiaCudaDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvidiaCudaDeviceBuffer")
            .field("device", &self.context.device)
            .field("raw", &self.raw)
            .field("len_bytes", &self.len_bytes)
            .finish()
    }
}

impl NvidiaCudaDeviceBuffer {
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    pub fn copy_from_host(&mut self, host_bytes: &[u8]) -> Result<()> {
        validate_copy_len(self.len_bytes, host_bytes.len(), "host-to-device copy")?;
        self.context.with_current(|driver| {
            driver.memcpy_host_to_device(self.raw, host_bytes.as_ptr().cast(), self.len_bytes)
        })
    }

    pub fn copy_to_host(&self, host_bytes: &mut [u8]) -> Result<()> {
        validate_copy_len(self.len_bytes, host_bytes.len(), "device-to-host copy")?;
        self.context.with_current(|driver| {
            driver.memcpy_device_to_host(host_bytes.as_mut_ptr().cast(), self.raw, self.len_bytes)
        })
    }

    pub fn download_to_vec(&self) -> Result<Vec<u8>> {
        let mut host_bytes = vec![0u8; self.len_bytes];
        self.copy_to_host(&mut host_bytes)?;
        Ok(host_bytes)
    }

    pub fn fill_zero(&mut self) -> Result<()> {
        self.context
            .with_current(|driver| driver.memset_d8(self.raw, 0, self.len_bytes))
    }
}

impl Drop for NvidiaCudaDeviceBuffer {
    fn drop(&mut self) {
        let _ = self
            .context
            .with_current(|driver| driver.mem_free(self.raw));
    }
}

pub struct NvidiaCudaTensor {
    dtype: DType,
    shape: Shape,
    buffer: NvidiaCudaDeviceBuffer,
}

impl std::fmt::Debug for NvidiaCudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvidiaCudaTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl NvidiaCudaTensor {
    pub fn from_device_buffer(
        dtype: DType,
        shape: Shape,
        buffer: NvidiaCudaDeviceBuffer,
    ) -> Result<Self> {
        let expected_len = checked_tensor_byte_len(dtype, &shape)?;
        validate_copy_len(
            expected_len,
            buffer.len_bytes(),
            "tensor device buffer binding",
        )?;
        Ok(Self {
            dtype,
            shape,
            buffer,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn element_count(&self) -> usize {
        self.shape.element_count()
    }

    pub fn len_bytes(&self) -> usize {
        self.buffer.len_bytes()
    }

    pub fn device_buffer(&self) -> &NvidiaCudaDeviceBuffer {
        &self.buffer
    }

    pub fn device_buffer_mut(&mut self) -> &mut NvidiaCudaDeviceBuffer {
        &mut self.buffer
    }

    pub fn into_device_buffer(self) -> NvidiaCudaDeviceBuffer {
        self.buffer
    }

    pub fn copy_from_tensor(&mut self, tensor: &Tensor) -> Result<()> {
        tensor.ensure_contiguous()?;
        validate_tensor_metadata(
            self.dtype,
            &self.shape,
            tensor.dtype(),
            tensor.shape(),
            "host-to-device tensor upload",
        )?;
        self.buffer.copy_from_host(tensor.as_bytes())
    }

    pub fn copy_to_tensor(&self, tensor: &mut Tensor) -> Result<()> {
        tensor.ensure_contiguous()?;
        validate_tensor_metadata(
            self.dtype,
            &self.shape,
            tensor.dtype(),
            tensor.shape(),
            "device-to-host tensor download",
        )?;
        let host_bytes = tensor.as_bytes_mut()?;
        self.buffer.copy_to_host(host_bytes)
    }

    pub fn download_to_tensor(&self) -> Result<Tensor> {
        let bytes = self.buffer.download_to_vec()?;
        Tensor::from_owned_bytes(self.dtype, self.shape.clone(), bytes)
    }

    pub fn fill_zero(&mut self) -> Result<()> {
        self.buffer.fill_zero()
    }
}

pub fn probe_nvidia_cuda() -> NvidiaCudaProbe {
    match try_load_nvidia_cuda_runtime() {
        Ok(runtime) => runtime.probe(),
        Err(probe) => probe,
    }
}

fn try_load_nvidia_cuda_runtime() -> std::result::Result<NvidiaCudaRuntime, NvidiaCudaProbe> {
    if cfg!(target_os = "macos") {
        return Err(NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA is not supported on macOS",
            None,
            None,
            None,
        ));
    }

    let candidates = cuda_driver_library_candidates();
    if candidates.is_empty() {
        return Err(NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA probing is unsupported on this platform",
            None,
            None,
            None,
        ));
    }

    let (driver, library_name) = match NvidiaCudaDriverApi::load(candidates) {
        Ok(driver) => driver,
        Err(detail) => {
            return Err(NvidiaCudaProbe::unavailable(
                "NVIDIA CUDA driver library was not found",
                Some(detail),
                None,
                None,
            ));
        }
    };
    let driver = Arc::new(driver);

    if let Err(detail) = driver.initialize() {
        return Err(NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA driver failed to initialize",
            Some(detail),
            Some(library_name),
            None,
        ));
    }

    let mut probe_detail = None;
    let driver_version = match driver.driver_version() {
        Ok(version) => Some(version),
        Err(detail) => {
            probe_detail = Some(detail);
            None
        }
    };

    let devices = match driver.devices() {
        Ok(devices) => devices,
        Err(detail) => {
            return Err(NvidiaCudaProbe::unavailable(
                "NVIDIA CUDA device enumeration failed",
                Some(detail),
                Some(library_name),
                driver_version,
            ));
        }
    };

    if devices.is_empty() {
        return Err(NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA driver reported zero devices",
            probe_detail,
            Some(library_name),
            driver_version,
        ));
    }

    Ok(NvidiaCudaRuntime {
        driver_library: library_name,
        driver_version,
        detail: probe_detail,
        devices,
        driver,
    })
}

struct NvidiaCudaCurrentGuard<'a> {
    driver: &'a NvidiaCudaDriverApi,
}

impl<'a> NvidiaCudaCurrentGuard<'a> {
    fn push(
        driver: &'a NvidiaCudaDriverApi,
        context: CuContextHandle,
    ) -> std::result::Result<Self, String> {
        driver.push_current_context(context)?;
        Ok(Self { driver })
    }
}

impl Drop for NvidiaCudaCurrentGuard<'_> {
    fn drop(&mut self) {
        let _ = self.driver.pop_current_context();
    }
}

struct NvidiaCudaDriverApi {
    _library: DynamicLibrary,
    cu_init: CuInitFn,
    cu_driver_get_version: CuDriverGetVersionFn,
    cu_device_get_count: CuDeviceGetCountFn,
    cu_device_get: CuDeviceGetFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_compute_capability: CuDeviceComputeCapabilityFn,
    cu_device_total_mem: CuDeviceTotalMemFn,
    cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetainFn,
    cu_device_primary_ctx_release: CuDevicePrimaryCtxReleaseFn,
    cu_ctx_push_current: CuCtxPushCurrentFn,
    cu_ctx_pop_current: CuCtxPopCurrentFn,
    cu_ctx_synchronize: CuCtxSynchronizeFn,
    cu_mem_alloc: CuMemAllocFn,
    cu_mem_free: CuMemFreeFn,
    cu_memcpy_htod: CuMemcpyHtoDFn,
    cu_memcpy_dtoh: CuMemcpyDtoHFn,
    cu_memset_d8: CuMemsetD8Fn,
}

impl NvidiaCudaDriverApi {
    fn load(candidates: &[&'static str]) -> std::result::Result<(Self, &'static str), String> {
        let (library, library_name) = DynamicLibrary::open_first(candidates)?;
        let api = unsafe {
            Self {
                cu_init: library.load_symbol(b"cuInit\0")?,
                cu_driver_get_version: library.load_symbol(b"cuDriverGetVersion\0")?,
                cu_device_get_count: library.load_symbol(b"cuDeviceGetCount\0")?,
                cu_device_get: library.load_symbol(b"cuDeviceGet\0")?,
                cu_device_get_name: library.load_symbol(b"cuDeviceGetName\0")?,
                cu_device_compute_capability: library
                    .load_symbol(b"cuDeviceComputeCapability\0")?,
                cu_device_total_mem: library.load_symbol(b"cuDeviceTotalMem_v2\0")?,
                cu_device_primary_ctx_retain: library.load_symbol(b"cuDevicePrimaryCtxRetain\0")?,
                cu_device_primary_ctx_release: library
                    .load_symbol(b"cuDevicePrimaryCtxRelease_v2\0")?,
                cu_ctx_push_current: library.load_symbol(b"cuCtxPushCurrent_v2\0")?,
                cu_ctx_pop_current: library.load_symbol(b"cuCtxPopCurrent_v2\0")?,
                cu_ctx_synchronize: library.load_symbol(b"cuCtxSynchronize\0")?,
                cu_mem_alloc: library.load_symbol(b"cuMemAlloc_v2\0")?,
                cu_mem_free: library.load_symbol(b"cuMemFree_v2\0")?,
                cu_memcpy_htod: library.load_symbol(b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_dtoh: library.load_symbol(b"cuMemcpyDtoH_v2\0")?,
                cu_memset_d8: library.load_symbol(b"cuMemsetD8_v2\0")?,
                _library: library,
            }
        };
        Ok((api, library_name))
    }

    fn initialize(&self) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_init)(0) };
        cuda_check(status, "cuInit")
    }

    fn driver_version(&self) -> std::result::Result<NvidiaCudaDriverVersion, String> {
        let mut version = 0;
        let status = unsafe { (self.cu_driver_get_version)(&mut version) };
        cuda_check(status, "cuDriverGetVersion")?;
        Ok(NvidiaCudaDriverVersion::from_raw(version))
    }

    fn devices(&self) -> std::result::Result<Vec<NvidiaCudaDeviceInfo>, String> {
        let mut count = 0;
        let status = unsafe { (self.cu_device_get_count)(&mut count) };
        cuda_check(status, "cuDeviceGetCount")?;
        if count <= 0 {
            return Ok(Vec::new());
        }

        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            devices.push(self.device_info(ordinal as usize)?);
        }
        Ok(devices)
    }

    fn device_info(&self, ordinal: usize) -> std::result::Result<NvidiaCudaDeviceInfo, String> {
        let mut device = 0;
        let status = unsafe { (self.cu_device_get)(&mut device, ordinal as c_int) };
        cuda_check(status, &format!("cuDeviceGet({ordinal})"))?;

        let name = self.device_name(device, ordinal);
        let (compute_capability_major, compute_capability_minor) =
            self.compute_capability(device).unwrap_or((0, 0));
        let total_memory_bytes = self.total_memory_bytes(device).unwrap_or(0);

        Ok(NvidiaCudaDeviceInfo {
            ordinal,
            name,
            compute_capability_major,
            compute_capability_minor,
            total_memory_bytes,
            raw_device: device,
        })
    }

    fn device_name(&self, device: CuDevice, ordinal: usize) -> String {
        let mut buffer = [0 as c_char; CUDA_DEVICE_NAME_BYTES];
        let status = unsafe {
            (self.cu_device_get_name)(buffer.as_mut_ptr(), buffer.len() as c_int, device)
        };
        if status != CUDA_SUCCESS {
            return format!("cuda_device_{ordinal}");
        }

        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string()
    }

    fn compute_capability(&self, device: CuDevice) -> std::result::Result<(u32, u32), String> {
        let mut major = 0;
        let mut minor = 0;
        let status = unsafe { (self.cu_device_compute_capability)(&mut major, &mut minor, device) };
        cuda_check(status, "cuDeviceComputeCapability")?;
        Ok((major as u32, minor as u32))
    }

    fn total_memory_bytes(&self, device: CuDevice) -> std::result::Result<u64, String> {
        let mut total = 0usize;
        let status = unsafe { (self.cu_device_total_mem)(&mut total, device) };
        cuda_check(status, "cuDeviceTotalMem_v2")?;
        Ok(total as u64)
    }

    fn retain_primary_context(
        &self,
        device: CuDevice,
    ) -> std::result::Result<CuContextHandle, String> {
        let mut context = std::ptr::null_mut();
        let status = unsafe { (self.cu_device_primary_ctx_retain)(&mut context, device) };
        cuda_check(status, "cuDevicePrimaryCtxRetain")?;
        if context.is_null() {
            return Err("cuDevicePrimaryCtxRetain returned a null CUDA context".to_string());
        }
        Ok(context)
    }

    fn release_primary_context(&self, device: CuDevice) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_device_primary_ctx_release)(device) };
        cuda_check(status, "cuDevicePrimaryCtxRelease_v2")
    }

    fn push_current_context(&self, context: CuContextHandle) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_ctx_push_current)(context) };
        cuda_check(status, "cuCtxPushCurrent_v2")
    }

    fn pop_current_context(&self) -> std::result::Result<CuContextHandle, String> {
        let mut previous = std::ptr::null_mut();
        let status = unsafe { (self.cu_ctx_pop_current)(&mut previous) };
        cuda_check(status, "cuCtxPopCurrent_v2")?;
        Ok(previous)
    }

    fn ctx_synchronize(&self) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_ctx_synchronize)() };
        cuda_check(status, "cuCtxSynchronize")
    }

    fn mem_alloc(&self, len_bytes: usize) -> std::result::Result<CuDevicePtr, String> {
        let mut device_ptr = 0u64;
        let status = unsafe { (self.cu_mem_alloc)(&mut device_ptr, len_bytes) };
        cuda_check(status, "cuMemAlloc_v2")?;
        if device_ptr == 0 {
            return Err("cuMemAlloc_v2 returned a null device pointer".to_string());
        }
        Ok(device_ptr)
    }

    fn mem_free(&self, device_ptr: CuDevicePtr) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_mem_free)(device_ptr) };
        cuda_check(status, "cuMemFree_v2")
    }

    fn memcpy_host_to_device(
        &self,
        dst_device: CuDevicePtr,
        src_host: *const c_void,
        len_bytes: usize,
    ) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_memcpy_htod)(dst_device, src_host, len_bytes) };
        cuda_check(status, "cuMemcpyHtoD_v2")
    }

    fn memcpy_device_to_host(
        &self,
        dst_host: *mut c_void,
        src_device: CuDevicePtr,
        len_bytes: usize,
    ) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_memcpy_dtoh)(dst_host, src_device, len_bytes) };
        cuda_check(status, "cuMemcpyDtoH_v2")
    }

    fn memset_d8(
        &self,
        dst_device: CuDevicePtr,
        value: u8,
        count: usize,
    ) -> std::result::Result<(), String> {
        let status = unsafe { (self.cu_memset_d8)(dst_device, value, count) };
        cuda_check(status, "cuMemsetD8_v2")
    }
}

fn cuda_check(status: c_int, context: &str) -> std::result::Result<(), String> {
    if status == CUDA_SUCCESS {
        return Ok(());
    }

    let reason = match status {
        1 => "invalid value",
        2 => "out of memory",
        3 => "not initialized",
        35 => "insufficient driver",
        100 => "no CUDA-capable device",
        201 => "invalid context",
        700 => "launch failed",
        999 => "unknown internal driver error",
        _ => "driver call failed",
    };
    Err(format!("{context} returned CUDA error {status} ({reason})"))
}

fn cuda_runtime_error(message: String) -> FerrisError {
    FerrisError::new(ErrorKind::Runtime, message)
}

fn validate_buffer_len(len_bytes: usize) -> Result<()> {
    if len_bytes == 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            "CUDA device buffers must be larger than zero bytes",
        ));
    }

    Ok(())
}

fn validate_copy_len(buffer_len: usize, host_len: usize, operation: &str) -> Result<()> {
    if buffer_len != host_len {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!(
                "CUDA {operation} requires matching lengths; buffer has {buffer_len} bytes but host slice has {host_len} bytes"
            ),
        ));
    }

    Ok(())
}

fn validate_tensor_metadata(
    expected_dtype: DType,
    expected_shape: &Shape,
    actual_dtype: DType,
    actual_shape: &Shape,
    operation: &str,
) -> Result<()> {
    if expected_dtype != actual_dtype {
        return Err(FerrisError::new(
            ErrorKind::InvalidType,
            format!(
                "CUDA {operation} requires matching dtype; expected {} but got {}",
                expected_dtype.name(),
                actual_dtype.name()
            ),
        ));
    }

    if expected_shape != actual_shape {
        return Err(FerrisError::new(
            ErrorKind::InvalidShape,
            format!(
                "CUDA {operation} requires matching shape; expected {:?} but got {:?}",
                expected_shape.dims(),
                actual_shape.dims()
            ),
        ));
    }

    Ok(())
}

fn checked_tensor_byte_len(dtype: DType, shape: &Shape) -> Result<usize> {
    shape
        .element_count()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| FerrisError::new(ErrorKind::Runtime, "CUDA tensor byte size overflow"))
}

fn cuda_driver_library_candidates() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["nvcuda.dll"]
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &["libcuda.so.1", "libcuda.so"]
    }

    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    {
        &[]
    }
}

struct DynamicLibrary {
    handle: *mut c_void,
}

impl DynamicLibrary {
    fn open_first(
        candidates: &[&'static str],
    ) -> std::result::Result<(Self, &'static str), String> {
        let mut errors = Vec::new();
        for &candidate in candidates {
            match unsafe { platform::open_library(candidate) } {
                Ok(handle) => return Ok((Self { handle }, candidate)),
                Err(error) => errors.push(format!("{candidate}: {error}")),
            }
        }

        Err(format!(
            "searched CUDA driver libraries [{}]",
            errors.join(", ")
        ))
    }

    unsafe fn load_symbol<T: Copy>(
        &self,
        symbol_name: &'static [u8],
    ) -> std::result::Result<T, String> {
        let symbol = platform::load_symbol(self.handle, symbol_name)?;
        debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        Ok(std::mem::transmute_copy(&symbol))
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            platform::close_library(self.handle);
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(path: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub unsafe fn open_library(path: &str) -> std::result::Result<*mut c_void, String> {
        let wide = OsStr::new(path)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let handle = LoadLibraryW(wide.as_ptr());
        if handle.is_null() {
            let error = std::io::Error::last_os_error();
            Err(error.to_string())
        } else {
            Ok(handle)
        }
    }

    pub unsafe fn load_symbol(
        handle: *mut c_void,
        symbol_name: &'static [u8],
    ) -> std::result::Result<*mut c_void, String> {
        let symbol = GetProcAddress(handle, symbol_name.as_ptr());
        if symbol.is_null() {
            Err(format!(
                "missing symbol {}",
                String::from_utf8_lossy(&symbol_name[..symbol_name.len().saturating_sub(1)])
            ))
        } else {
            Ok(symbol)
        }
    }

    pub unsafe fn close_library(handle: *mut c_void) {
        if !handle.is_null() {
            let _ = FreeLibrary(handle);
        }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use super::*;

    const RTLD_NOW: c_int = 2;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    pub unsafe fn open_library(path: &str) -> std::result::Result<*mut c_void, String> {
        let path = CString::new(path).map_err(|error| error.to_string())?;
        let handle = dlopen(path.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            Err(last_dlerror())
        } else {
            Ok(handle)
        }
    }

    pub unsafe fn load_symbol(
        handle: *mut c_void,
        symbol_name: &'static [u8],
    ) -> std::result::Result<*mut c_void, String> {
        let symbol = dlsym(handle, symbol_name.as_ptr() as *const c_char);
        if symbol.is_null() {
            Err(last_dlerror())
        } else {
            Ok(symbol)
        }
    }

    pub unsafe fn close_library(handle: *mut c_void) {
        if !handle.is_null() {
            let _ = dlclose(handle);
        }
    }

    unsafe fn last_dlerror() -> String {
        let error = dlerror();
        if error.is_null() {
            "unknown dynamic loader error".to_string()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}

#[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
mod platform {
    use super::*;

    pub unsafe fn open_library(_path: &str) -> std::result::Result<*mut c_void, String> {
        Err("dynamic library loading is not implemented on this platform".to_string())
    }

    pub unsafe fn load_symbol(
        _handle: *mut c_void,
        _symbol_name: &'static [u8],
    ) -> std::result::Result<*mut c_void, String> {
        Err("dynamic symbol lookup is not implemented on this platform".to_string())
    }

    pub unsafe fn close_library(_handle: *mut c_void) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_cuda_driver_version_parses_major_and_minor() {
        let version = NvidiaCudaDriverVersion::from_raw(12030);
        assert_eq!(version.raw(), 12030);
        assert_eq!(version.major(), 12);
        assert_eq!(version.minor(), 3);
    }

    #[test]
    fn cuda_driver_library_candidates_match_platform_expectations() {
        let candidates = cuda_driver_library_candidates();

        #[cfg(windows)]
        assert_eq!(candidates, &["nvcuda.dll"]);

        #[cfg(all(unix, not(target_os = "macos")))]
        assert_eq!(candidates, &["libcuda.so.1", "libcuda.so"]);
    }

    #[test]
    fn validate_copy_len_rejects_mismatched_lengths() {
        let error = validate_copy_len(16, 8, "device-to-host copy").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidShape);
        assert!(error.message().contains("16 bytes"));
        assert!(error.message().contains("8 bytes"));
    }

    #[test]
    fn validate_buffer_len_rejects_zero_bytes() {
        let error = validate_buffer_len(0).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidShape);
    }

    #[test]
    fn checked_tensor_byte_len_matches_shape_and_dtype() {
        let len =
            checked_tensor_byte_len(DType::F32, &Shape::from_slice(&[2, 3]).unwrap()).unwrap();
        assert_eq!(len, 24);
    }

    #[test]
    fn validate_tensor_metadata_rejects_dtype_mismatch() {
        let error = validate_tensor_metadata(
            DType::F32,
            &Shape::from_slice(&[4]).unwrap(),
            DType::U8,
            &Shape::from_slice(&[4]).unwrap(),
            "host-to-device tensor upload",
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidType);
    }

    #[test]
    fn validate_tensor_metadata_rejects_shape_mismatch() {
        let error = validate_tensor_metadata(
            DType::F32,
            &Shape::from_slice(&[4]).unwrap(),
            DType::F32,
            &Shape::from_slice(&[2, 2]).unwrap(),
            "device-to-host tensor download",
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidShape);
    }
}
