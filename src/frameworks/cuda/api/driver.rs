use std::os::raw::c_void;
use std::ptr;
use super::{ContextFlag, ContextHandle, DeviceHandle, Memory, sys};
use super::error::{Error, ErrorKind, Result};

/// Initialize the CUDA driver API.
///
/// Initializes the driver API and must be called before any other function from the 
/// driver API. Currently, the `flags` parameter must be 0. If cuInit() has not been called, 
/// any function from the driver API will return `NotInitialized`.
///
/// # Parameters
///
/// * `flags` - Initialization flag for CUDA.
///
/// # Returns
///
/// Returns `()` if initialization was successful, otherwise returns an error kind of 
/// `InvalidValue` or `InvalidDevice`.
///
/// # Note
///
/// Note that this function may also return error codes from previous, asynchronous launches.
pub fn init() -> Result {

    let result = unsafe { sys::cuInit(0) };

    match result {
        sys::cudaError_enum::CUDA_SUCCESS => Ok(()),

        e @ _ => {
            let message = "An error occurred while attempting to initialize the CUDA driver API";
            Err(Error::new(e, message))
        }
    }
}

/// Returns the number of compute-capable devices.
///
/// Returns the number of devices with compute capability greater than or equal to 1.0 that 
/// are available for execution. If there is no such device, returns 0.
pub fn ndevices() -> Result<u32> {
    let mut ndevices = 0;

    unsafe {
        match sys::cuDeviceGetCount(&mut ndevices) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(ndevices as u32),

            e @ _ => 
                Err(Error::from(e.into(): ErrorKind))
        }
    }
}

/// Returns a handle to a compute device.
///
/// # Parameters
///
/// * `ordinal` - Device number to get handle for.
pub fn device(n: u32) -> Result<DeviceHandle> {

    let mut device_handle: i32 = 0;

    unsafe {
        match sys::cuDeviceGet(&mut device_handle, n as i32) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(DeviceHandle(device_handle)),

            e @ _ => 
                Err(Error::from(e.into(): ErrorKind))
        }
    }
}

/// Create a CUDA context.
///
/// # Parameters
///
/// * `f` - Context creation flags
pub fn create_context(f: ContextFlag, dev: &DeviceHandle) -> Result<ContextHandle> {
    unsafe {
        let mut ctx = ptr::null_mut();

        match sys::cuCtxCreate_v2(&mut ctx, f as u32, dev.0) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(ContextHandle(ctx)),

            e @ _ =>
                Err(Error::from(e.into(): ErrorKind)),
        }
    }
}

/// Allocates byte_size bytes of linear memory on the device and returns in *dptr a pointer to the 
/// allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory 
/// is not cleared. If bytesize is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE.
pub fn mem_alloc(byte_size: usize) -> Result<Memory> {

    unsafe {
        let mut dptr = 0u64;

        match sys::cuMemAlloc_v2(&mut dptr, byte_size) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(Memory(dptr)),

            e @ _ =>
                Err(Error::from(e.into(): ErrorKind)),
        }
    }
}

/// Copies from host memory to device memory. dstDevice and srcHost are the base addresses of the 
/// destination and source, respectively. ByteCount specifies the number of bytes to copy. Note 
/// that this function is synchronous.
pub fn mem_cpy_h_to_d(dst_device: &Memory, src_host: *const c_void, byte_count: usize) -> Result {

    unsafe {

        match sys::cuMemcpyHtoD_v2(dst_device.0, src_host, byte_count) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(()),

            e @ _ =>
                Err(Error::from(e.into(): ErrorKind)),
        }
    }
}

/// Copies from device to host memory. dstHost and srcDevice specify the base pointers of the 
/// destination and source, respectively. ByteCount specifies the number of bytes to copy. Note 
/// that this function is synchronous.
pub fn mem_cpy_d_to_h(dst_host: *mut c_void, src_device: &Memory, byte_count: usize) -> Result {

    unsafe {

        match sys::cuMemcpyDtoH_v2(dst_host, src_device.0, byte_count) {
            sys::cudaError_enum::CUDA_SUCCESS => 
                Ok(()),

            e @ _ =>
                Err(Error::from(e.into(): ErrorKind)),   
        }
    }
}