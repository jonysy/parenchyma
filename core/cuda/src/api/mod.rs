pub use cuda_sys::CUdevice_attribute as CudaAttribute;
pub use self::device::CudaDeviceHandle;
pub use self::driver::CudaDriver;

mod device;
mod driver;