use super::{ContextView, DeviceView};

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
/// ```rust
/// use parenchyma::{Backend, Framework, Native};
///
/// 
/// // Construct a new framework.
/// let framework = Native::new().expect("failed to initialize framework");
///
/// // Available devices can be obtained through the framework.
/// let selection = framework.available_devices.clone();
///
/// // Create a ready to go backend from the framework.
/// let backend = Backend::new(framework, selection).expect("failed to construct backend");
///
/// // ..
/// ```
///
/// Construct a default backend:
///
/// ```rust
/// use parenchyma::{Backend, Native};
///
/// // A default native backend.
/// let backend: Backend<Native> = Backend::default().expect("something went wrong!");
///
/// // ..
/// ```
#[derive(Debug)]
pub struct Backend { context: ContextView }

impl Backend {

    pub fn devices(&self) -> Vec<DeviceView> {
        match self.context {
            ContextView::OpenCL(ref c) => 
                c.devices().iter().map(|d| DeviceView::OpenCL(d.clone())).collect(),
            ContextView::Native(ref c) => 
                c.devices().iter().map(|d| DeviceView::Native(d.clone())).collect(),
        }
    }

    pub fn context(&self) -> &ContextView {
        &self.context
    }
}

impl Default for Backend {
    /// Construct a `Backend` from a [`framework`](./trait.Framework.html), such as OpenCL, CUDA, etc.,
    /// and a `selection` of devices.
    fn default() -> Backend {
        use super::opencl::{OpenCL, OpenCLContext};

        let try_opencl_context = OpenCL::try_new().and_then(|frwk|
            OpenCLContext::new(frwk.available_platforms[0].available_devices.clone())
        );

        match try_opencl_context {
            Ok(opencl_context) => Backend {
                context: ContextView::OpenCL(
                    opencl_context
                ), 
            },

            Err(opencl_error) => {
                use super::native::{Native, NativeContext};

                error!("[OpenCL] {}", opencl_error);

                let native_frwk = Native::new();
                let native_context = NativeContext::new(native_frwk.available_devices);

                Backend {
                    context: ContextView::Native(native_context),
                }
            }
        }
    }
}