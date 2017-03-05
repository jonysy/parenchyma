pub use self::error::{Error, ErrorKind, Result};
pub use self::high::{nplatforms, platforms};
pub use self::high::{Buffer, Context, Device, Event, Kernel, KernelArg, Platform, Program, Queue};

mod error;
mod high;
mod utility;