#[cfg(all(unix, not(target_os = "macos")))]
use std::ffi::CString;
use std::ffi::{c_char, c_int, c_void, CStr};

use ferrisinfer_core::{DeviceKind, ExecutionConfig, FerrisError, Result, Tensor};

use crate::backend::{Backend, BackendAvailability, BackendCapabilities};

const CUDA_SUCCESS: i32 = 0;
const CUDA_DEVICE_NAME_BYTES: usize = 256;

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

#[derive(Debug, Clone)]
pub struct NvidiaCudaBackend {
    config: ExecutionConfig,
    probe: NvidiaCudaProbe,
}

impl NvidiaCudaBackend {
    pub fn new(mut config: ExecutionConfig) -> Self {
        config.preferred_device = DeviceKind::Cuda;
        Self {
            config,
            probe: probe_nvidia_cuda(),
        }
    }

    pub fn probe(&self) -> &NvidiaCudaProbe {
        &self.probe
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

pub fn probe_nvidia_cuda() -> NvidiaCudaProbe {
    if cfg!(target_os = "macos") {
        return NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA is not supported on macOS",
            None,
            None,
            None,
        );
    }

    let candidates = cuda_driver_library_candidates();
    if candidates.is_empty() {
        return NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA probing is unsupported on this platform",
            None,
            None,
            None,
        );
    }

    let (driver, library_name) = match NvidiaCudaDriverApi::load(candidates) {
        Ok(driver) => driver,
        Err(detail) => {
            return NvidiaCudaProbe::unavailable(
                "NVIDIA CUDA driver library was not found",
                Some(detail),
                None,
                None,
            );
        }
    };

    if let Err(detail) = driver.initialize() {
        return NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA driver failed to initialize",
            Some(detail),
            Some(library_name),
            None,
        );
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
            return NvidiaCudaProbe::unavailable(
                "NVIDIA CUDA device enumeration failed",
                Some(detail),
                Some(library_name),
                driver_version,
            );
        }
    };

    if devices.is_empty() {
        return NvidiaCudaProbe::unavailable(
            "NVIDIA CUDA driver reported zero devices",
            probe_detail,
            Some(library_name),
            driver_version,
        );
    }

    NvidiaCudaProbe::available(library_name, driver_version, devices, probe_detail)
}

type CuDevice = c_int;
type CuInitFn = unsafe extern "C" fn(flags: u32) -> c_int;
type CuDriverGetVersionFn = unsafe extern "C" fn(driver_version: *mut c_int) -> c_int;
type CuDeviceGetCountFn = unsafe extern "C" fn(count: *mut c_int) -> c_int;
type CuDeviceGetFn = unsafe extern "C" fn(device: *mut CuDevice, ordinal: c_int) -> c_int;
type CuDeviceGetNameFn =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, device: CuDevice) -> c_int;
type CuDeviceComputeCapabilityFn =
    unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int, device: CuDevice) -> c_int;
type CuDeviceTotalMemFn = unsafe extern "C" fn(bytes: *mut usize, device: CuDevice) -> c_int;

struct NvidiaCudaDriverApi {
    _library: DynamicLibrary,
    cu_init: CuInitFn,
    cu_driver_get_version: CuDriverGetVersionFn,
    cu_device_get_count: CuDeviceGetCountFn,
    cu_device_get: CuDeviceGetFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_compute_capability: CuDeviceComputeCapabilityFn,
    cu_device_total_mem: CuDeviceTotalMemFn,
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
}

fn cuda_check(status: c_int, context: &str) -> std::result::Result<(), String> {
    if status == CUDA_SUCCESS {
        return Ok(());
    }

    let reason = match status {
        35 => "insufficient driver",
        100 => "no CUDA-capable device",
        999 => "unknown internal driver error",
        _ => "driver call failed",
    };
    Err(format!("{context} returned CUDA error {status} ({reason})"))
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
}
