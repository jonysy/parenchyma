//! OpenCL backend support - heterogeneous computing.

pub mod foreign;
pub mod high;

pub use self::interface::{OpenCL, OpenCLContext, OpenCLDevice, OpenCLEvent, OpenCLImage, OpenCLMemory};

mod interface;