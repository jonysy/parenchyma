use ocl;
use std::ffi::CString;
use std::marker::Unsize;
use super::{OpenCL, OpenCLDevice};
use super::super::super::compute_device::ComputeDevice;
use super::super::super::context::{Context, ContextCtor};
use super::super::super::error::{Error, ErrorKind, Result};
use super::super::super::extension_package::{ExtensionPackage, ExtensionPackageCtor};
use super::super::super::hardware::Hardware;

/// Defines a Open CL context.
///
/// A context is responsible for managing OpenCL objects and resources (command-queues, program 
/// objects, kernel objects, executing kernels, etc.). The usual configuration is a single context 
/// encapsulating multiple devices. The resources, such as [buffers][buffer] and [events][event], 
/// can be shared across multiple devices in a single context. Other possible setups include:
///
/// * a single context for multiple devices
/// * a single context for a single device
/// * a context for each device
///
/// note: multi-platform contexts are not supported in OpenCL.
///
/// ## Programs
///
/// An OpenCL context can have multiple programs associated with it. Programs can be compiled
/// individually to avoid possible name clashes due to using packages from multiple package 
/// authors.
///
/// [buffer]: ./frameworks/opencl/struct.Memory.html
/// [event]: ./frameworks/opencl/struct.Event.html
pub struct OpenCLContext<P> {
    /// The context.
    context: ocl::Context,
    /// The index of the _active_ device.
    active: usize,
    /// A list of devices associated with the context.
    selected_devices: Vec<OpenCLDevice>,
    /// The `Device`s' corresponding `Hardware`.
    selected_hardware: Vec<Hardware>,
    // todo document this:
    // package is stored here because
    // a) the program depends on the selected devices
    // b) the lazy static would new the context
    //   1) mutating would be possible but wouldn't be worth the cost and trouble
    extension_package: P,
}

impl<P> OpenCLContext<P> {
    pub fn device(&self) -> &OpenCLDevice {
        &self.selected_devices[self.active]
    }

    pub fn extension_package(&self) -> &P {
        &self.extension_package
    }
    
    /// Builds and returns a program.
    pub fn program(&self, src_strings: Vec<CString>) -> Result<ocl::Program> {
        let cmplr_opts = CString::new("").unwrap();
        let device_ids: Vec<_> = self.selected_devices.iter().map(|d| d.device.clone()).collect();

        Ok(ocl::Program::new(
            self.context.core(), 
            src_strings, 
            Some(&device_ids), 
            cmplr_opts
        )?)
    }
}

impl<Package> Context for OpenCLContext<Package> 
    where Package: ExtensionPackage, 
          OpenCLContext<Package>: Unsize<Package::Extension> {

    type Package = Package;

    fn active_codev(&self) -> &ComputeDevice {
        &self.selected_devices[self.active]
    }

    fn extension(&self) -> &<Package as ExtensionPackage>::Extension {
        self
    }

    fn activate(&mut self, index: usize) -> Result {
        if index >= self.selected_devices.len() {
            return Err(Error::new(ErrorKind::Other, "device index out of range"));
        }

        self.active = index;

        Ok(())
    }
}

impl<P> ContextCtor<P> for OpenCLContext<P>
    where P: 'static + ExtensionPackage + ExtensionPackageCtor<OpenCLContext<()>>, 
          OpenCLContext<P>: Unsize<P::Extension> {
            
    type F = OpenCL<P>;

    fn new(framework: &Self::F, selection: &[Hardware]) -> Result<Self> {

        let props = ocl::builders::ContextProperties::new().platform(framework.implementation);
        let s = ocl::builders::DeviceSpecifier::Indices(selection.iter().map(|h| h.id).collect());
        let ctx = ocl::Context::new(Some(props), Some(s), None, None)?;

        let mut devices = vec![];

        for hardware in selection.iter() {
            let d = ocl::Device::by_idx_wrap(framework.implementation, hardware.id);
            let queue = ocl::Queue::new(&ctx, d, Some(ocl::flags::QUEUE_PROFILING_ENABLE))?;

            devices.push(OpenCLDevice {
                device: d,
                context: ctx.clone(),
                queue,
            });
        }

        let mut unpackaged = OpenCLContext { 
            context: ctx, 
            active: 0, 
            selected_devices: devices, 
            selected_hardware: selection.to_vec(),
            extension_package: (),
        };

        let package = P::package(&mut unpackaged)?;

        Ok(OpenCLContext {
            context: unpackaged.context,
            active: unpackaged.active,
            selected_devices: unpackaged.selected_devices,
            selected_hardware: unpackaged.selected_hardware,
            extension_package: package,
        })
    }
}