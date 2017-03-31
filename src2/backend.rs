use context::{Context, ContextConfig};
use error::Result;
use extension::ExtensionPackage;
use framework::{Framework, FrameworkExt};
use hardware::{Hardware, HardwareDevice, HardwareType};
use std::marker::Unsize;
use std::ops::Deref;

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
pub struct Backend<I = Context> where I: ?Sized {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be used and
    /// determines which hardwares will be available and how parallel kernel functions can be
    /// executed.
    framework: Box<Framework>,
    /// Provides a context, created from one or many hardware, which are ready to execute kernel
    /// methods and synchronize memory.
    ///
    /// Contexts are the heart of both OpenCL and CUDA applications. Contexts are created from one 
    /// or more devices that are capable of executing methods and synchronizing memory. See 
    /// the [`Context`] trait for more information.
    ///
    /// [`Context`]: (./trait.Context.html)
    context: Box<I>,
}

impl<I> Backend<I> where I: Context + ?Sized {

    /// Attempts to create a default backend of type `F`.
    pub fn new<F>() -> Backend<I>
        where F: 'static + Default + Framework + FrameworkExt, 
              F::Context: Unsize<I> 
              {

        let framework = F::default();
        let selection = framework.hardware().to_vec();
        Backend::with(framework, selection)
    }

    /// Constructs a backend from the provided `framework` and hardware `selection`.
    pub fn with<F>(f: F, selection: Vec<Hardware>) -> Backend<I>
        where F: 'static + Framework + FrameworkExt, 
              F::Context: Unsize<I> 
              {

        let context = box F::Context::from(ContextConfig { framework: &f, selection }) as Box<I>;
        let framework = Box::new(f) as Box<Framework>;
        Backend { framework, context }
    }
}

impl<I> AsRef<HardwareDevice> for Backend<I> where I: ?Sized + Context {

    fn as_ref(&self) -> &(HardwareDevice + 'static) {
        self.context.active()
    }
}

impl<I> Deref for Backend<I> where I: ?Sized {

    type Target = I;

    fn deref(&self) -> &I {
        &self.context
    }
}

/// Provides a configured backend.
#[derive(Debug)]
pub struct BackendConfig<F> {
    pub framework: F,
    pub hardware: Vec<Hardware>,
    pub preferred_type: HardwareType,
}

impl<F, I> From<BackendConfig<F>> for Backend<I> 
    where F: 'static + Framework + FrameworkExt, 
          F::Context: Unsize<I>, 
          I: Context + ExtensionPackage + ?Sized {

    /// TODO use `TryFrom` when stable.
    ///
    /// [#33417](https://github.com/rust-lang/rust/issues/33417)
    fn from(configuration: BackendConfig<F>) -> Backend<I> {
        let BackendConfig { framework, hardware, preferred_type } = configuration;
        let mut backend = Self::with(framework, hardware);
        if let Err(error) = backend.context.select(&|h| h.processor == preferred_type) {
            let message = "failed to activate the preferred device";
            let warning = "certain operations provided by the extension package may fail";
            let pkg = I::PACKAGE_NAME;

            error!("{message} - {warning} (extension package: {pkg}): {error}", 
                message=message,
                warning=warning, 
                pkg=pkg,
                error=error
            );
        }

        backend
    }
}