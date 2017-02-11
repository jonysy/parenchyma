//! * [Specification](https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf)
//! * [Reference Card](https://www.khronos.org/files/opencl-1-1-quick-reference-card.pdf)

#![feature(pub_restricted, type_ascription)]

extern crate opencl_sys;

pub mod api;
pub mod error;