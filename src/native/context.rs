use super::super::Context;
use super::super::error::Result;
use std::hash::{Hash, Hasher};
use super::{Native, NativeDevice, NativeMemory};

#[derive(Clone, Debug)]
pub struct NativeContext {
    selected_devices: Vec<NativeDevice>,
}

impl Context for NativeContext {
    type F = Native;

    fn new(devices: Vec<NativeDevice>) -> Result<Self> {
        
        Ok(NativeContext { selected_devices: devices })
    }

    fn allocate_memory(&self, size: usize) -> Result<NativeMemory> {
        let memory = NativeMemory::allocate(size);
        Ok(memory)
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

impl Hash for NativeContext {

    fn hash<H: Hasher>(&self, state: &mut H) {
        static ID: isize = 0;
        ID.hash(state);
    }
}