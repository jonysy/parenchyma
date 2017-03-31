//! Native backend support.

pub use self::context::NativeContext;
pub use self::device::NativeDevice;
pub use self::framework::Native;
pub use self::memory::NativeMemory;

mod context;
mod device;
mod framework;
mod memory;

/// Host CPU
pub const HOST: &NativeDevice = &NativeDevice;