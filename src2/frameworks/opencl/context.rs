use context::{self, ContextConfig};
use error::{Error, ErrorKind, Result};
use hardware::{Hardware, HardwareDevice};
use std::os::raw::c_void;
use super::api::{core, import};
use super::{ComputeDevice, OpenCL};

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
pub struct Context {
    /// The high-level context.
    context: core::Context,
    /// A list of devices provided to the context.
    selection: Vec<Hardware>,
    /// A list of _activatable_ hardware.
    ///
    /// Each item in the list directly corresponds to its general representation 
    /// found in `selection`.
    activatable_hardware: Vec<ComputeDevice>,
    /// The index of the _active_ device.
    active: usize,

    // TODO
    // work_dim must be greater than zero and less than or equal 
    // to CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS.
    // max_work_item_dims: u32,
}

impl context::Context for Context {

    fn selection(&self) -> &[Hardware] {
        &self.selection
    }


    fn active(&self) -> &(HardwareDevice + 'static) {
        &self.activatable_hardware[self.active]
    }

    /// Set the device at the specified `index` as the active device.
    ///
    /// Only one device can be the _active_ device - the device in which operations are executed -
    /// if used through the context.
    fn activate(&mut self, index: usize) -> Result {
        if index >= self.selection.len() {
            return Err(Error::new(ErrorKind::Other, "the provided `index` is out of range"));
        }

        self.active = index;

        Ok(())
    }
}

impl<'a> From<ContextConfig<'a, OpenCL>> for Context {

    fn from(co: ContextConfig<'a, OpenCL>) -> Context {

        let devices: Vec<core::Device> = 
            co.selection.iter().map(|h| core::Device::from(h.id as *mut c_void)).collect();
        let context = core::Context::new(&devices).expect("failed to create the context");
        let mut activatable_hardware = vec![];

        for device in devices.into_iter() {
            // create a command queue (with profiling enabled, needed for timing kernels)
            let queue = core::CommandQueue::new(
                    &context, 
                    &device, 
                    import::CL_QUEUE_PROFILING_ENABLE
                )
                .expect("failed to create a command queue for the device");

            activatable_hardware.push(ComputeDevice {
                device,
                context: context.clone(),
                queue
            });
        }

        Context { context, selection: co.selection, activatable_hardware, active: 0 }
    }
}