use super::super::Context;
use super::super::error::Result;
use super::{Native, NativeDevice, NativeMemory};

/// Native context
#[derive(Clone, Debug)]
pub struct NativeContext {
    /// Selected devices
    selected_devices: NativeDevice,
}

impl Context for NativeContext {
    type F = Native;

    fn new(devices: NativeDevice) -> Result<Self> {
        
        Ok(NativeContext { selected_devices: devices })
    }

    fn allocate_memory(&self, size: usize) -> Result<NativeMemory> {
        let bx: Box<[u8]> = super::memory::allocate_boxed_slice(size);
        let mem = NativeMemory::from(bx);
        Ok(mem)
    }

    fn synch_in(&self, destn: &mut NativeMemory, _: &NativeContext, src: &NativeMemory) -> Result {

        destn.as_mut_slice::<u8>().clone_from_slice(src.as_slice::<u8>());

        Ok(())
    }

    fn synch_out(&self, src: &NativeMemory, _: &NativeContext, destn: &mut NativeMemory) -> Result {

        destn.as_mut_slice::<u8>().clone_from_slice(src.as_slice::<u8>());

        Ok(())
    }
}

impl PartialEq for NativeContext {

    fn eq(&self, _: &Self) -> bool { true }
}

impl Eq for NativeContext { }