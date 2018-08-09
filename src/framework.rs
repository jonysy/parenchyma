//! Provides the generic functionality of a hardware supporting frameworks such 
//! as native CPU, Open CL, CUDA, etc..
//!
//! The default framework is simply the host CPU for common computation. To make use 
//! of other devices such as GPUs, you may choose a GPGPU framework (such as OpenCL or CUDA) to 
//! access the processing capabilities of the device(s). To start backend-agnostic and highly 
//! parallel computation, you start by initializing one of the `Framework` implementations, 
//! resulting in an initialized Framework, that contains, among other things, a list of all 
//! available hardwares through that framework.
//!
//! # Example
//!
//! ```
//! extern crate parenchyma;
//! 
//! use parenchyma::frameworks::Native;
//! use parenchyma::prelude::*;
//!
//! // A ready to go backend can be created by simply providing the framework type.
//! let backend: Backend = Backend::new::<Native>().unwrap();
//! ```

use super::error::Result;
use super::hardware::Hardware;

/// A trait implemented for all frameworks. `Framework`s contain a list of all available 
/// devices as well as other objects specific to the implementor.
pub trait Framework: 'static {
    /// Returns the name of the framework, which is mainly used for the purposes of debugging 
    /// and reporting errors.
    fn name(&self) -> &'static str;
    /// Returns the cached and available hardware.
    ///
    /// note: this method will likely be replaced 
    /// with a [field](https://github.com/rust-lang/rfcs/pull/1546).
    fn hardware(&self) -> &[Hardware];
}

/// The non-object-safe part of the framework trait.
///
/// A separate trait is used because it violates object-safety rules, i.e., `Framework` is the 
/// object-safe version of `FrameworkCtor` (or `FrameworkCtor` is the non-object-safe 
/// version of `Framework`). `FrameworkCtor` is simply a constructor (hence the name `*Ctor`). In 
/// other words, this trait is split into object-safe and non-object-safe parts.
///
/// todo: generic associated types may help here..
pub trait FrameworkCtor: Framework + Sized {
    /// The context representation for the framework.
    type Context;
    /// Initializes a `Framework`.
    fn new() -> Result<Self>;
}