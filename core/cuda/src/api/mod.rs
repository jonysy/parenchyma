pub mod driver;
pub use cuda_sys::CUdevice_attribute as Attribute;
pub use cuda_sys::CUctx_flags_enum as ContextFlag;
pub use self::context::ContextHandle;
pub use self::device::DeviceHandle;

mod context;
mod device;