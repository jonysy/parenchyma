#![allow(warnings)]
#![feature(associated_consts, box_syntax, conservative_impl_trait, get_type_id)]

extern crate futures;

mod backend;
mod context;
mod device;
mod error;
mod framework;
mod hardware;
mod memory;
mod tensor;
mod utility;