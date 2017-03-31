use context::Context;

/// Provides the generic functionality for a backend-specific implementation of a library.
///
/// A package can be a binary, a source file, c code, a single kernel, etc., or a collective which
/// share related functionalities. A package is provided by a specific library such as BLAS. Notice
/// that packages are analogous to those of Rust (i.e., crates):
///
/// compiled crate <-> package
/// library (one or more modules) <-> bundle
///
/// A package needs to be _built_, which is handled by the specific implementation of a binary
/// representation, and returns initialized operations based on the library. Interacting directly
/// with the package itself is possible, but it should be used to construct the backend-agnostic
/// operations, which can then be executed and parallelized via a unified interface.
///
/// ## Extensions
///
/// A library can be a binary, a source file, c code, a single kernel, etc., or a collective.
///
/// A backend is a Rust struct like any other, therefore you probably would like to implement
/// certain methods for the Backend. As the whole purpose of a Backend is to provide an
/// abstraction over various computation devices and computation languages, these implemented
/// methods will than be able to execute on different devices and use the full power of 
/// the machine's underlying hardware.
///
/// Extending the backend with operations is easy. In Parenchyma we call crates, which provide
/// operations for the backend, _extensions_. Extensions are usually a group of related 
/// operations of a common field. Two examples for Parenchyma extensions 
/// are [BLAS][parenchyma-blas] and [NN][parenchyma-nn].
///
/// An extension provides generic traits and the explicit implementation of these traits for 
/// one or (even better) all available Parenchyma frameworks - common host CPU, OpenCL, CUDA.
///
/// The structure of an extension is pretty simple with as little overhead as possible. Macros 
/// and build-scripts make implementations even easier. If you would like to use specific 
/// extension for you backend, all you need to do is set them as dependencies in your Cargo 
/// file in addition to the Parenchyma crate. The extension then automatically extends the 
/// backend provided by Parenchyma.
///
/// Extending the backend with your own extension is a straightforward process. For now we 
/// recommend that you take a look at the general code structure 
/// of [Parenchyma-BLAS][parenchyma-blas] or its documentation. Let us now about your extension 
/// on the Gitter chat, we are happy to feature your Parenchyma Extension on the README.
pub trait ExtensionPackage {
    /// The name of the package.
    ///
    /// This associated constant is primarily used for logging/debugging purposes. The naming 
    /// convention is as follows: "[organization]/[package-name]" (e.g., "parenchyma/nn").
    const PACKAGE_NAME: &'static str;
}

impl<I> ExtensionPackage for I where I: Context {

    /// An _unextended_ backend/context.
    const PACKAGE_NAME: &'static str = "parenchyma";
}