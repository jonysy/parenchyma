//! Exposes the specific framework implementations.

pub use self::native::{HOST, Native, NativeContext, NativeDevice, NativeMemory};
pub use self::open_cl::{OpenCL, OpenCLBuf, OpenCLContext, OpenCLDevice, OpenCLMemory};

mod native;
mod open_cl;