//! Provides a simple, unified API for running highly parallel computations on different
//! devices across different GPGPU frameworks, allowing you to swap your backend at runtime.
//!
//! Parenchyma is a hard fork of [Collenchyma], a now-defunct project started at [Autumn].
//!
//! ## Abstract
//!
//! Code is often executed on the CPU, but can be executed on other devices, such as GPUs
//! and accelerators. These devices are accessible through GPGPU frameworks. Most interfaces are 
//! complicated, making the use of these devices a painful experience. Some of the pain points when 
//! writing such code for a particular device are:
//!
//! * portability: not only do frameworks have different interfaces, devices support different 
//! versions and machines might have different hardware - all of this leads to code that will be 
//! executable only on a very specific set of machines and platforms.
//! * learning curve: executing code on a device through a framework is quite different to
//! running code on the native CPU and comes with a lot of hurdles. OpenCL's 1.2 specification, for
//! example, has close to 400 pages.
//! * custom code: integrating support for devices into your project requires the need for writing
//! a lot of low-level code, e.g., kernels, memory management, and general business logic.
//!
//! Writing code for non-CPU devices is often a good choice, as these devices can execute
//! operations a lot faster than native CPUs. GPUs, for example, can execute operations roughly
//! one to two orders of magnitudes faster, thanks to better support of parallelizing operations.
//!
//! Parenchyma eliminates the pain points of writing device code, so you can run your code like any 
//! other code without needing to learn about kernels, events, or memory synchronization. Parenchyma
//! also allows you to deploy your code with ease to servers, desktops and mobile device, all while 
//! enabling your code to make full use of the underlying hardware. 
//!
//! ## Architecture
//!
//! The single entry point of Parenchyma is the [Backend](./struct.Backend.html) type. A
//! backend is agnostic over the device it runs operations on. In order to be agnostic over the 
//! device, such as native host CPU, GPUs, accelerators or any other devices, the backend needs to be
//! agnostic over the framework as well. The framework is important, as it provides the interface 
//! to execute operations on devices, among other things. Since different vendors of hardware use 
//! different frameworks, it becomes important that the backend is agnostic over the framework.
//! This allows us to run computations on any machine without having to worry about hardware 
//! availability, which gives us the freedom to write code once and deploy it on different machines 
//! where it will execute on the most potent hardware by default.
//!
//! ### Frameworks
//!
//! The default framework is simply the host CPU for common computation. To make use of other
//! devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to access the 
//! processing capabilities of the device(s).
//!
//! ### Extensions
//!
//! Operations are introduced by a Parenchyma extension. An extension extends your backend with 
//! ready-to-execute operations. All you need to do is add the Parenchyma extension crate(s)
//! to your `Cargo.toml` file alongside the Parenchyma crate. Your backend will then be extended with
//! operations provided by the extension(s). The interface is simply the language you're using to 
//! work with Parenchyma. For example, you'd simply call `backend.dot(..)` using Rust-lang and 
//! a BLAS extension. Whether or not the dot operation is executed on one GPU, multiple GPUS or on 
//! a CPU device depends solely on how you configured the backend.
//!
//! ### Bundles
//!
//! The concept of Parenchyma extensions has one more component - the [Bundle](./trait.Bundle.html)
//! trait. As opposed to executing code on the native CPU, other devices need to compile and build 
//! the extension manually at runtime, which makes up a significant part of a framework. We need an
//! instance that's able to be initialized at runtime for holding the sate and compiled 
//! operations - which is the bundle's main purpose.
//!
//! ### Memory
//!
//! The last piece of Parenchyma is the memory. An operation happens over data, but this data needs
//! to be accessible to the device on which the operation is executed. That memory space needs to be
//! allocated on the device and then, in a later step, synced from the host to the device or from
//! the device back to the host. Thanks to the [Tensor](./struct.SharedTensor.html) type, we do not
//! have to care about memory management between devices for the execution of operations. The tensor
//! tracks and automatically manages data and its memory across devices, which is often the host and
//! the device. Memory can also be passed around to different backends. Operations take tensors
//! as arguments while handling the synchronization and allocation for you.
//!
//! ## Development
//!
//! At the moment, Parenchyma itself will provide Rust APIs for the important 
//! frameworks - OpenCL and CUDA.
//!
//! If a framework isn't specified, the backend will try to use the most potent framework given
//! the underlying hardware - which would probably be in this order: CUDA -> OpenCL -> Native. The 
//! process might take longer, as every framework needs to be check and devices need to be loaded in
//! order to identify the best setup. The time it takes to go through that process is a reasonable
//! compromise as it would allow you to deploy a Parenchyma-backed application to almost any
//! machine - server, desktops, mobiles, etc.
//!
//! [Collenchyma]: https://github.com/autumnai/collenchyma
//! [Autumn]: https://github.com/autumnai

// #![deny(missing_docs, unused_import_braces, unused_qualifications)]
#![feature(associated_consts, field_init_shorthand, libc, type_ascription, untagged_unions)]

#[macro_use] extern crate enum_primitive;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate log;

extern crate libc;
extern crate libloading as lib;
extern crate ndarray;

pub mod changelog;
pub mod error;
pub mod frameworks;

pub use self::backend::Backend;
pub use self::buffer::Buffer;
pub use self::context::Context;
pub use self::device::Device;
pub use self::framework::Framework;
pub use self::processor::Processor;
pub use self::tensor::{Shape, SharedTensor};

mod backend;
mod buffer;
mod context;
mod device;
mod framework;
mod processor;
mod tensor;
mod utility;