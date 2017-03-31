use framework::Framework;

/// The heart of Parenchyma - provides an interface for running parallel computations on one or 
/// more devices.
///
/// The `Backend` type is an abstraction over a [framework](./trait.Framework.html) and is used as 
/// a way to interact with your devices. A backend provides you with the functionality of managing 
/// the memory of the devices and copying memory objects to/from the host. Additionally, backends 
/// allow you to execute operations in parallel through kernel functions on devices.
///
/// ## Architecture
///
/// Backends are initialized by providing a framework and a selection of devices compatible with 
/// the framework to the [`Backend::new`](#method.new) associated function, or by simply 
/// calling [`Backend::default`](#method.default). The framework determines which devices are 
/// available and how parallel kernel functions can be executed.
///
/// ## Examples
///
/// ```
/// ..
/// ```
#[derive(Debug)]
pub struct Backend<I> where I: ?Sized {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be 
    /// used and determines which hardwares will be available and how parallel kernel 
    /// functions can be executed.
    framework: Box<Framework>,
    /// Provides a context, created from one or many hardware, which are ready to execute 
    /// kernel methods and synchronize memory.
    ///
    /// Contexts are the heart of both OpenCL and CUDA applications. Contexts are created from one 
    /// or more devices that are capable of executing methods and synchronizing memory. See 
    /// the [`Context`] trait for more information.
    ///
    /// [`Context`]: (./trait.Context.html)
    context: Box<I>,
}