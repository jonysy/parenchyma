#![allow(unused_variables)]
#![feature(non_modrs_mods)]

extern crate ocl;
extern crate parenchyma;

pub use self::extension_package::{Extension, Package};
pub mod frameworks;

mod extension_package;