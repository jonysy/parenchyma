use parenchyma::error::Result;
use std::hash::{Hash, Hasher};
use super::{cl, core, parenchyma};
use super::{Device, Memory, OpenCL, Queue};

#[derive(Debug)]
pub struct Context {
    id: core::Context,
    /// <sup>*</sup>Multi-platforms contexts are not supported in OpenCL.
    platform_id: core::PlatformId,
    selected_devices: Vec<Device>,
    queue: Option<Queue>,
}

impl PartialEq for Context {

    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Context { }

impl Hash for Context {

    fn hash<H>(&self, state: &mut H) where H: Hasher {
        (unsafe { self.id.as_ptr() as isize }).hash(state);
    }
}

impl parenchyma::Context for Context {
    type Framework = OpenCL;

    /// Constructs a context from a selection of devices.
    ///
    /// Contexts are used by the OpenCL runtime for managing objects such as command-queues,
    /// memory, program and kernel objects and for executing kernels on one or more devices
    /// specified in the context.
    fn new(devices: Vec<Device>) -> Result<Self> {

        let selected = cl::Device::list_from_core(devices.iter().map(|d| d.id).collect());

        // > Thread safety and destruction for any enclosed pointers are all handled 
        // > automatically. Clone, store, and share between threads to your heart's content.
        let cl_context = cl::builders::ContextBuilder::new().devices(&selected).build().unwrap();

        let id = cl_context.core_as_ref().clone();
        let platform_id = *cl_context.platform().unwrap().as_core();

        let context = Context {
            id: id,
            platform_id: platform_id,
            selected_devices: devices,
            queue: None,
        };

        Ok(context)
    }

    /// Allocates memory
    fn allocate_memory(&self, size: usize) -> Result<Memory> {

        unimplemented!()
    }
}