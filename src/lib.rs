#![allow(warnings)]
#![cfg_attr(feature = "unstable_alloc", feature(alloc))]
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![feature(associated_consts, pub_restricted)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate enum_primitive;
#[macro_use]
extern crate lazy_static;

#[cfg(feature = "unstable_alloc")]
extern crate alloc;
extern crate byteorder;
extern crate libc;
extern crate linear_map;
extern crate num;

pub mod frameworks;

pub use backend::Backend;
pub use context::{Context, ContextImp};
pub use device::{Device, DeviceKind};
pub use error::Error;
pub use framework::Framework;
pub use memory::MemoryImp;

mod backend;
mod context;
mod device;
mod error;
mod framework;
mod memory;
mod tensor;