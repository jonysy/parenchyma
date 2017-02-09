//! Provides a simple and unified API to run fast and highly parallel computations on different
//! processors such as CPUs and GPUs, across different computation languages such as OpenCL and
//! CUDA and allows you to swap your backend on run-time.
//!
//! Parenchyma is a hard fork of [Collenchyma]. Collenchyma was started at [Autumn] to create an 
//! easy and performant abstraction over different backends for the Machine Intelligence 
//! Framework [Leaf], with no hard dependency on any driver or libraries so that it can easily be 
//! used without the need for a long and painful build process.
//!
//! ## Abstract
//!
//! Code often is executed on the native CPU, but could be executed on other processors such as GPUs
//! and Accelerators. These processors are accessible through frameworks like OpenCL and CUDA
//! but have a more complicated interfaces than your every-day native CPU which makes the use of 
//! these devices a painful experience. Some of the pain points, when writing such device code, are:
//!
//! * non-portable: frameworks have different interfaces, devices support different versions and
//! machines might have different hardware - all this leads to code that will be executable only on
//! a very specific set of machines and platforms.
//! * steep learning curve: executing code on a device through a framework is quite different to
//! running code on the native CPU and comes with a lot of hurdles. OpenCLs 1.2 specification for
//! example has close to 400 pages.
//! * custom code: integrating support for devices into your project, requires the need for writing
//! a lot of custom code e.g. kernels, memory management, general business logic.
//!
//! But writing code for devices would often be a good choice as these devices can execute many
//! operations a lot faster than the native CPUs. GPUs for example can execute operations roughly
//! one to two orders of magnitudes faster, thanks to better support of parallelizing operations.
//! OpenCL and CUDA make parallelizing operations super easy.
//!
//! Parenchyma eliminates the pain points of writing device code, so you can run your code like any 
//! other code, don't need to learn about kernels, events, or memory synchronization, and can 
//! deploy your code with ease to servers, desktops or mobiles and your code will make full use of 
//! the underlying hardware.
//!
//! ## Architecture
//!
//! The single entry point of Parenchyma is a [Backend](./backend/struct.Backend.html).
//!
//! TODO..
//!
//! ## Example
//!
//! ```rust,ignore
//! // TODO..
//! ```
//!
//! ## Development
//!
//! TODO..
//!
//! [Collenchyma]: https://github.com/autumnai/collenchyma
//! [Autumn]: https://github.com/autumnai
//! [Leaf]: https://github.com/autumnai/leaf

#![allow(warnings)]
#![cfg_attr(feature = "unstable_alloc", feature(alloc))]
#![feature(associated_consts, get_type_id, pub_restricted)]

#[cfg(feature = "unstable_alloc")]
extern crate alloc;

pub mod error;
pub use self::context::Context;
pub use self::backend::Backend;
pub use self::framework::Framework;
pub use self::native::{Native, NativeContext, NativeDevice, NativeMemory};
pub use self::processor::Processor;
pub use self::tensor::Tensor;

mod context;
mod backend;
mod framework;
mod native;
mod processor;
mod tensor;