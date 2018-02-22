#![allow(unused_variables)]
#![feature(non_modrs_mods)]

extern crate ocl;
extern crate parenchyma;

pub use self::extension_package::{Extension, Package};

mod extension_package;
mod frameworks;