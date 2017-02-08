use cuda_sys;
use error::{Error, ErrorKind, Result};
use std::ptr;
use super::{CudaContextFlag, CudaContextHandle, CudaDeviceHandle};

#[derive(Debug)]
pub struct CudaDriver;

impl CudaDriver {
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
    pub fn init() -> Result<Self> {

        let result = unsafe { cuda_sys::cuInit(0) };

        match result {
            cuda_sys::cudaError_enum::CUDA_SUCCESS => Ok(CudaDriver),

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
            match cuda_sys::cuDeviceGetCount(&mut ndevices) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
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
    pub fn device(n: u32) -> Result<CudaDeviceHandle> {

        let mut device_handle: i32 = 0;

        unsafe {
            match cuda_sys::cuDeviceGet(&mut device_handle, n as i32) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(CudaDeviceHandle(device_handle)),

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
    pub fn create_context(f: CudaContextFlag, dev: CudaDeviceHandle) -> Result<CudaContextHandle> {
        unsafe {
            let mut ctx = ptr::null_mut();

            match cuda_sys::cuCtxCreate_v2(&mut ctx, f as u32, dev.0) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(CudaContextHandle(ctx)),

                e @ _ =>
                    Err(Error::from(e.into(): ErrorKind)),
            }
        }
    }
}