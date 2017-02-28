use std::borrow::Cow;
use super::{OpenCLMemory, OpenCLQueue, Result};
use super::api;
use super::super::super::{DeviceKind, DeviceView, MemoryView};

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
    pub queue: Option<OpenCLQueue>,
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

impl OpenCLDevice {

    /// Allocates memory on a device.
    pub fn allocate_memory(&self, size: usize) -> Result<OpenCLMemory> {

        let mem_obj = api::Memory::create_buffer(self.context.as_ref().unwrap(), size)?;

        Ok(OpenCLMemory { obj: mem_obj })
    }

    /// Synchronizes `memory` from `source`.
    pub fn synch_in(&self, destn: &mut OpenCLMemory, _: &DeviceView, src: &MemoryView) -> Result {
        match *src {
            MemoryView::Native(ref src_native_memory) => {
                let s_size = src_native_memory.len();
                let s_ptr = src_native_memory.as_slice().as_ptr();

                self.queue.as_ref().unwrap().ptr().write_buffer(&destn.obj, true, 0, s_size, s_ptr, &[])?;

                Ok(())
            },

            _ => unimplemented!()
        }
    }

    /// Synchronizes `memory` to `destination`.
    pub fn synch_out(&self, src: &OpenCLMemory, _: &DeviceView, destn: &mut MemoryView) -> Result {
        match *destn {
            MemoryView::Native(ref d_opencl_memory) => {
                let d_size = d_opencl_memory.len();
                let d_ptr = d_opencl_memory.as_mut_slice().as_mut_ptr();

                self.queue.as_ref().unwrap().ptr().read_buffer(&src.obj, true, 0, d_size, d_ptr, &[])?;

                Ok(())
            },

            _ => unimplemented!()
        }
    }
}

impl PartialEq for OpenCLDevice {

    fn eq(&self, other: &Self) -> bool {
        self.context == other.context;
        self.ptr == other.ptr
    }
}

impl Eq for OpenCLDevice { }