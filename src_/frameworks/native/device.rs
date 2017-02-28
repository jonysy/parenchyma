use super::super::super::{DeviceKind, DeviceView, MemoryView};
use super::NativeMemory;

/// Native device
#[derive(Clone, Debug)]
pub struct NativeDevice {
	pub(super) name: &'static str,
	pub(super) compute_units: isize,
	pub(super) kind: DeviceKind,
}

impl NativeDevice {
    /// Allocates memory on a device.
    pub fn allocate_memory(&self, size: usize) -> NativeMemory {
        let bx: Box<[u8]> = super::memory::allocate_boxed_slice(size);
        NativeMemory::from(bx)
    }

    /// Synchronizes `memory` from `source`.
    pub fn synch_in(&self, destn: &mut NativeMemory, _: &DeviceView, src: &MemoryView) {

        match *src {
            MemoryView::Native(ref native_mem_src) =>
                destn.as_mut_slice::<u8>().clone_from_slice(native_mem_src.as_slice::<u8>()),
            _ => unimplemented!(),
        }
    }

    /// Synchronizes `memory` to `destination`.
    pub fn synch_out(&self, src: &NativeMemory, _: &DeviceView, destn: &mut MemoryView) {

        match *destn {
            MemoryView::Native(ref native_mem_destn) =>
                native_mem_destn.as_mut_slice::<u8>().clone_from_slice(src.as_slice::<u8>()),
            _ => unimplemented!(),
        }
    }
}

impl PartialEq for NativeDevice {

    fn eq(&self, _: &Self) -> bool { true }
}

impl Eq for NativeDevice { }