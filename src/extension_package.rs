//! A package can be a binary, a source file, c code, a single kernel, etc., or a collective which
//! share related functionalities. A package is provided by a specific library such as BLAS. Notice
//! that packages are analogous to those of Rust (i.e., crates):
//!
//! compiled crate <-> package
//! library (one or more modules) <-> bundle
//!
//! A package needs to be _built_, which is handled by the specific implementation of a binary
//! representation, and returns initialized operations based on the library. Interacting directly
//! with the package itself is possible, but it should be used to construct the backend-agnostic
//! operations, which can then be executed and parallelized via a unified interface.
//!
//! ## Extensions
//!
//! A library can be a binary, a source file, c code, a single kernel, etc., or a collective.
//!
//! A backend is a Rust struct like any other, therefore you probably would like to implement
//! certain methods for the Backend. As the whole purpose of a Backend is to provide an
//! abstraction over various computation devices and computation languages, these implemented
//! methods will than be able to execute on different devices and use the full power of 
//! the machine's underlying hardware.
//!
//! Extending the backend with operations is easy. In Parenchyma we call crates, which provide
//! operations for the backend, _extensions_. Extensions are usually a group of related 
//! operations of a common field. Two examples for Parenchyma extensions 
//! are [BLAS][parenchyma-blas] and [NN][parenchyma-nn].
//!
//! An extension provides generic traits and the explicit implementation of these traits for 
//! one or (even better) all available Parenchyma frameworks - common host CPU, OpenCL, CUDA.
//!
//! The structure of an extension is pretty simple with as little overhead as possible. Macros 
//! and build-scripts make implementations even easier. If you would like to use specific 
//! extension for you backend, all you need to do is set them as dependencies in your Cargo 
//! file in addition to the Parenchyma crate. The extension then automatically extends the 
//! backend provided by Parenchyma.
//!
//! Extending the backend with your own extension is a straightforward process. For now we 
//! recommend that you take a look at the general code structure 
//! of [Parenchyma-BLAS][parenchyma-blas] or its documentation. Let us now about your extension 
//! on the Gitter chat, we are happy to feature your Parenchyma Extension on the README.

use super::context::Context;
use super::error::Result;

/// Represents a package dependency.
pub trait Dependency<P>: ExtensionPackage {
    /// Returns the dependency.
    fn dependency(&self) -> &P;
}

impl<P> Dependency<P> for P where P: ExtensionPackage {
    fn dependency(&self) -> &P {
        &self
    }
}

/// Provides the generic functionality for a backend-specific implementation of a library.
pub trait ExtensionPackage: 'static {
    type Extension: ?Sized;

    /// The name of the package.
    ///
    /// This associated constant is primarily used for logging/debugging purposes. The naming 
    /// convention is as follows: "[organization]/[package-name]" (e.g., "parenchyma/nn").
    fn package_name(&self) -> &'static str;
}

/// Builds a package and provides the functionality for turning a library into backend-specific, 
/// executable operations, and tailored for the target framework.
///
/// note: the `Context` trait is used here simply as a marker trait.
pub trait ExtensionPackageCtor<TargetContext>: Sized
    /*where Self: ExtensionPackage + Sized, 
          TargetContext: Context*/ {
    /// Compiles the library into a package after initializing and configuring the library.
    ///
    /// This associated constant is primarily used for logging/debugging purposes. The naming 
    /// convention is as follows: "[organization]/[package-name]" (e.g., "parenchyma/nn").
    fn package(target: &mut TargetContext) -> Result<Self>;
}

impl ExtensionPackage for () {
    type Extension = ::std::any::Any;
    /// The default package.
    fn package_name(&self) -> &'static str {
        return "parenchyma/default";
    }
}

impl<T> ExtensionPackageCtor<T> for () where T: Context {
    fn package(_target: &mut T) -> Result<Self> {
        return Ok(());
    }
}