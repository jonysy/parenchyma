use api::{self, enqueue};
use parenchyma::{Device, DeviceKind, NativeDevice, NativeMemory};
use std::borrow::Cow;
use super::{OpenCL, OpenCLMemory, OpenCLQueue, Result};

#[derive(Clone, Debug)]
pub struct OpenCLDevice {
    pub(super) ptr: api::Device,
    /// maximum compute units
    pub compute_units: u32,
    /// The name of the device
    pub name: String,
    /// The maximum work-group size.
    pub workgroup_size: usize,
    /// Device type
    pub kind: DeviceKind,

    context: Option<api::Context>,
    queue: Option<OpenCLQueue>,
}

impl OpenCLDevice {

    pub fn new(ptr: api::Device) -> Result<Self> {

        let compute_units = ptr.max_compute_units()?;
        let name = ptr.name()?;
        let workgroup_size = ptr.max_work_group_size()?;

        let kind = match ptr.type_()? {
            api::sys::CL_DEVICE_TYPE_CPU => DeviceKind::Cpu,
            api::sys::CL_DEVICE_TYPE_GPU => DeviceKind::Gpu,
            api::sys::CL_DEVICE_TYPE_ACCELERATOR => DeviceKind::Accelerator,
            p @ _ => DeviceKind::Other(Cow::from(p.to_string()))
        };

        Ok(OpenCLDevice {
            ptr: ptr,
            compute_units: compute_units,
            name: name,
            workgroup_size: workgroup_size,
            kind: kind,
            context: None,
            queue: None,
        })
    }

    pub fn kind(&self) -> &DeviceKind {
        &self.kind
    }

    pub fn prepare(&mut self, context: api::Context, queue: OpenCLQueue) {
        self.context = Some(context);
        self.queue = Some(queue);
    }
}

impl Device for OpenCLDevice {

    type F = OpenCL;

    fn allocate_memory(&self, size: usize) -> Result<OpenCLMemory> {

        let mem_obj = api::Memory::create_buffer(self.context.as_ref().unwrap(), size)?;

        Ok(OpenCLMemory { obj: mem_obj })
    }

    fn synch_in(&self, destn: &mut OpenCLMemory, _: &NativeDevice, src: &NativeMemory) -> Result {

        let s_size = src.len();
        let s_ptr = src.as_slice().as_ptr();
        let p = self.queue.as_ref().unwrap().ptr();
        let _ = enqueue::write_buffer(p, &destn.obj, true, 0, s_size, s_ptr, &[])?;

        Ok(())
    }

    fn synch_out(&self, src: &OpenCLMemory, _: &NativeDevice, destn: &mut NativeMemory) -> Result {

        let d_size = destn.len();
        let d_ptr = destn.as_mut_slice().as_mut_ptr();
        let p = self.queue.as_ref().unwrap().ptr();
        let _ = enqueue::read_buffer(p, &src.obj, true, 0, d_size, d_ptr, &[])?;

        Ok(())
    }
}

impl PartialEq for OpenCLDevice {

    fn eq(&self, other: &Self) -> bool {
        self.context == other.context;
        self.ptr == other.ptr
    }
}

impl Eq for OpenCLDevice { }