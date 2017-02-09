//! Native backend support for Parenchyma.

pub use self::context::NativeContext;
pub use self::device::NativeDevice;
pub use self::framework::Native;
pub use self::memory::NativeMemory;

mod context;
mod device;
mod framework;
mod memory;