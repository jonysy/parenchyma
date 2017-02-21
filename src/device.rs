use std::borrow::Cow;
use super::{Error, ErrorKind, MemoryView, Result};
use super::native::NativeDevice;
use super::opencl::OpenCLDevice;

pub enum DeviceView {
    Native(NativeDevice),
    OpenCL(OpenCLDevice),
}

/// General categories for devices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceKind {
    /// Accelerators
    Accelerator,
    /// CPU devices (host processors)
    Cpu,
    /// GPU devices
    Gpu,
    /// Used for anything else
    Other(Cow<'static, str>),
}

impl DeviceView {

    /// Allocates memory on a device.
    pub fn allocate_memory(&self, size: usize) -> Result<MemoryView> {
        match *self {
            DeviceView::Native(ref dev) => Ok(MemoryView::Native(dev.allocate_memory(size))),
            DeviceView::OpenCL(ref dev) => Ok(MemoryView::OpenCL(dev.allocate_memory(size)?)),
        }
    }

    /// Synchronizes `memory` from `source`.
    pub fn synch_in(&self, destn_m: &mut MemoryView, src_d: &DeviceView, src_m: &MemoryView) 
        -> Result {

        match *self {
            DeviceView::Native(ref dev) => {
                let destn_native = destn_m.as_mut_native()
                    .ok_or(ErrorKind::MemorySynchronizationFailed.into(): Error);

                Ok(dev.synch_in(destn_native?, src_d, src_m))
            },

            DeviceView::OpenCL(ref dev) => {
                let destn_opencl = destn_m.as_mut_opencl()
                    .ok_or(ErrorKind::MemorySynchronizationFailed.into(): Error);

                Ok(dev.synch_in(destn_opencl?, src_d, src_m)?)
            },
        }
    }

    /// Synchronizes `memory` to `destination`.
    pub fn synch_out(&self, src_m: &MemoryView, destn_d: &DeviceView, destn_m: &mut MemoryView) 
        -> Result {

        match *self {
            DeviceView::Native(ref dev) => {
                let src_native = src_m.as_native()
                    .ok_or(ErrorKind::MemorySynchronizationFailed.into(): Error);

                Ok(dev.synch_out(src_native?, destn_d, destn_m))
            },

            DeviceView::OpenCL(ref dev) => {
                let src_opencl = src_m.as_opencl()
                    .ok_or(ErrorKind::MemorySynchronizationFailed.into(): Error);

                Ok(dev.synch_out(src_opencl?, destn_d, destn_m)?)
            },
        }
    }
}