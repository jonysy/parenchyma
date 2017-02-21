use super::{OpenCLDevice, OpenCLQueue, Result};
use super::api;

// notes:
// shared context if more than one device is passed in
// Multi-platforms contexts are not supported in OpenCL.

// TODO prevent mixing devices from different contexts

/// A context is responsible for managing OpenCL objects and resources.
///
/// Contexts are used by the OpenCL runtime for managing objects such as command-queues,
/// memory, program and kernel objects and for executing kernels on one or more devices
/// specified in the context. A single context for multiple devices, a single context for a 
/// single device, and a context for each device are all possible.
///
/// Memory objects can be copied to host memory, from host memory, or to other memory objects.
/// Copying from the host to a device is considered _writing_. Copying from a device to the host is
/// considered _reading_.
#[derive(Clone, Debug)]
pub struct OpenCLContext {
    ptr: api::Context,
    selected_devices: Vec<OpenCLDevice>,
}

impl OpenCLContext {

    /// Constructs a context from a selection of devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - a list of devices.
    pub fn new(mut devices: Vec<OpenCLDevice>) -> Result<Self> {

        let raw_devices: Vec<_> = devices.iter().map(|d| d.ptr.clone()).collect();

        let raw_context = api::Context::new(&raw_devices)?;

        for device in devices.iter_mut() {
            let queue = OpenCLQueue { ptr: api::Queue::new(&raw_context, &device.ptr, 0)? };

            device.prepare(raw_context.clone(), queue);
        }

        Ok(OpenCLContext { ptr: raw_context, selected_devices: devices })
    }

    pub fn devices(&self) -> &[OpenCLDevice] {
        &self.selected_devices
    }

    pub fn ptr(&self) -> &api::Context {
        &self.ptr
    }
}