pub use self::error::{Error, ErrorKind, Result};
pub use self::high::{init, ndevices};
pub use self::high::{Context, Device, Memory};

mod error;
mod high;
mod utility;