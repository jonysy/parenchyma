use parenchyma::Context;
use parenchyma::error::Result;
use std::hash::{Hash, Hasher};
use super::{NativeDevice, NativeMemory};

#[derive(Clone, Debug)]
pub struct NativeContext {
    selected_devices: Vec<NativeDevice>,
}

impl Context for NativeContext {
    type D = NativeDevice;
    type M = NativeMemory;

    fn new(devices: Vec<Self::D>) -> Result<Self> {
        
        Ok(NativeContext { selected_devices: devices })
    }

    fn allocate_memory(&self, size: usize) -> Result<Self::M> {
        let memory = NativeMemory::allocate(size);
        Ok(memory)
    }

    // fn sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any) -> Result {

    //  if let Some(_) = src_device.downcast_ref::<NativeContext>() {
    //      let my_mem = my_memory.downcast_mut::<NativeMemory>().unwrap();
    //      let src_mem = src_memory.downcast_ref::<NativeMemory>().unwrap();
    //      my_mem.as_mut_slice::<u8>().clone_from_slice(src_mem.as_slice::<u8>());
    //      return Ok(());
    //  }

    //  Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
    // }

    // fn sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any) -> Result {

    //  if let Some(_) = dst_device.downcast_ref::<NativeContext>() {
    //      let my_mem = my_memory.downcast_ref::<NativeMemory>().unwrap();
    //      let dst_mem = dst_memory.downcast_mut::<NativeMemory>().unwrap();
    //      dst_mem.as_mut_slice::<u8>().clone_from_slice(my_mem.as_slice::<u8>());
    //      return Ok(());
    //  }

    //  Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
    // }
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