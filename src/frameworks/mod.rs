#[cfg(feature = "native")]
pub use self::native::{Native, NativeContext, NativeDevice, NativeMemory};

#[cfg(feature = "native")]
mod native;