#![allow(warnings)]
#![cfg_attr(feature = "unstable_alloc", feature(alloc))]
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]

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

pub mod api;
pub mod error;
pub mod frameworks;