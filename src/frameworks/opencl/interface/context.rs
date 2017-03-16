use {Context, Device, Error, ErrorKind, ExtensionPackage, Result};
use std::marker::Unsize;
use super::OpenCLDevice;
use super::super::high;
use utility::Uninitialized;

/// Represents an OpenCL context.
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
#[derive(Debug)]
pub struct OpenCLContext<X> {
    /// The high-level context.
    pub(super) context: high::Context,
    /// Holds the extension package implementation.
    pub(super) package: X,
    /// A list of devices associated with the context.
    pub(super) selection: Vec<OpenCLDevice>,
    /// The index of the _active_ device.
    pub(super) active: usize,
}

impl<X> OpenCLContext<X> {

    /// Returns the `package`.
    ///
    /// [package author]
    pub fn package(&self) -> &X {
        &self.package
    }

    /// Returns the _active_ OpenCL device.
    ///
    /// [package author]
    pub fn device(&self) -> &OpenCLDevice {
        &self.selection[self.active]
    }
}

impl OpenCLContext<Uninitialized> {
    /// Creates and returns a program.
    pub fn create_program<I>(&mut self, src: &[I]) -> Result<high::Program> where I: AsRef<str> {
        let program = self.context.create_program_with_source(src)?;
        let raw_devices: Vec<_> = self.selection.iter().map(|d| d.device.clone()).collect();
        program.build(&raw_devices, None /* TODO */)?;
        Ok(program)
    }
}

impl<X> Context for OpenCLContext<X> 
    where X: ExtensionPackage, 
          OpenCLContext<X>: Unsize<X::Extension> {

    type Package = X;

    fn active_device(&self) -> &Device {

        &self.selection[self.active]
    }

    fn set_active(&mut self, idx: usize) -> Result {
        if idx >= self.selection.len() {
            return Err(Error::new(ErrorKind::Other, "device index out of range"));
        }

        self.active = idx;

        Ok(())
    }

    fn extension(&self) -> &X::Extension {

        self
    }
}