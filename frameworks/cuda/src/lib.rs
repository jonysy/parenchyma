#![feature(pub_restricted, type_ascription)]

extern crate cuda_sys;

pub mod error;
pub use self::api::*;

mod api;