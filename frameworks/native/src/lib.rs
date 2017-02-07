#![feature(associated_consts, pub_restricted)]
#![cfg_attr(feature = "unstable_alloc", feature(alloc))]

extern crate parenchyma;

#[cfg(feature = "unstable_alloc")]
extern crate alloc;

pub use self::context::NativeContext;
pub use self::device::NativeDevice;
pub use self::framework::Native;
pub use self::memory::NativeMemory;

mod context;
mod device;
mod framework;
mod memory;