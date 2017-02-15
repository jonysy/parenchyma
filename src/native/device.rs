use super::super::{DeviceKind, Device};
use super::super::error::Result;
use super::{Native, NativeMemory};

/// Native device
#[derive(Clone, Debug)]
pub struct NativeDevice {
	pub(super) name: &'static str,
	pub(super) compute_units: isize,
	pub(super) kind: DeviceKind,
}

impl Device for NativeDevice {
    type F = Native;

    fn allocate_memory(&self, size: usize) -> Result<NativeMemory> {
        let bx: Box<[u8]> = super::memory::allocate_boxed_slice(size);
        let mem = NativeMemory::from(bx);
        Ok(mem)
    }

    fn synch_in(&self, destn: &mut NativeMemory, _: &NativeDevice, src: &NativeMemory) -> Result {

        destn.as_mut_slice::<u8>().clone_from_slice(src.as_slice::<u8>());

        Ok(())
    }

    fn synch_out(&self, src: &NativeMemory, _: &NativeDevice, destn: &mut NativeMemory) -> Result {

        destn.as_mut_slice::<u8>().clone_from_slice(src.as_slice::<u8>());

        Ok(())
    }
}

impl PartialEq for NativeDevice {

    fn eq(&self, _: &Self) -> bool { true }
}

impl Eq for NativeDevice { }