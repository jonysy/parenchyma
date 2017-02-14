use api;
use parenchyma::Processor;
use std::borrow::Cow;
use super::Result;

#[derive(Clone, Debug)]
pub struct OpenCLDevice {
    pub(super) ptr: api::Device,
    /// maximum compute units
    pub compute_units: u32,
    /// The name of the device
    pub name: String,
    /// The maximum work-group size.
    pub workgroup_size: usize,

    pub processor: Processor,
}

impl OpenCLDevice {

    pub fn new(ptr: api::Device) -> Result<Self> {

        let compute_units = ptr.max_compute_units()?;
        let name = ptr.name()?;
        let workgroup_size = ptr.max_work_group_size()?;

        let processor = match ptr.type_()? {
            api::sys::CL_DEVICE_TYPE_CPU => Processor::Cpu,
            api::sys::CL_DEVICE_TYPE_GPU => Processor::Gpu,
            api::sys::CL_DEVICE_TYPE_ACCELERATOR => Processor::Accelerator,
            p @ _ => Processor::Other(Cow::from(p.to_string()))
        };

        Ok(OpenCLDevice {
            ptr: ptr,
            compute_units: compute_units,
            name: name,
            workgroup_size: workgroup_size,
            processor: processor,
        })
    }
}