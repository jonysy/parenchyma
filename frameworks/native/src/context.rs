use parenchyma::{Context, Error};
use parenchyma::error;
use std::any::Any;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};
use super::{memory, NativeDevice, NativeMemory};

#[derive(Clone)]
pub struct NativeContext {
	devices: Vec<NativeDevice>,
}

impl NativeContext {
	
	fn id(&self) -> &isize {
		static ID: isize = 0;
		&ID
	}
}

impl Context for NativeContext {
	type Memory = NativeMemory;

	// anti-pattern?
	fn allocate_memory(&self, size: usize) -> Result<NativeMemory, Error> {

		let bx: Box<[u8]> = memory::allocate_boxed_slice(size);
		let mem = NativeMemory::from(bx);

		Ok(mem)
	}

	fn sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any) -> Result<(), Error> {

		if let Some(_) = src_device.downcast_ref::<NativeContext>() {
			let my_mem = my_memory.downcast_mut::<NativeMemory>().unwrap();
			let src_mem = src_memory.downcast_ref::<NativeMemory>().unwrap();
			my_mem.as_mut_slice::<u8>().clone_from_slice(src_mem.as_slice::<u8>());
			return Ok(());
		}

		Err(Error::new(error::MemoryCategory::NoMemorySyncRoute, "No memory sync route"))
	}

	fn sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any) -> Result<(), Error> {

		if let Some(_) = dst_device.downcast_ref::<NativeContext>() {
			let my_mem = my_memory.downcast_ref::<NativeMemory>().unwrap();
			let dst_mem = dst_memory.downcast_mut::<NativeMemory>().unwrap();
			dst_mem.as_mut_slice::<u8>().clone_from_slice(my_mem.as_slice::<u8>());
			return Ok(());
		}

		Err(Error::new(error::MemoryCategory::NoMemorySyncRoute, "No memory sync route"))
	}
}

impl TryFrom<Vec<NativeDevice>> for NativeContext {
	type Err = Error;

	fn try_from(devices: Vec<NativeDevice>) -> Result<Self, Self::Err> {
		
		Ok(NativeContext { devices: devices })
	}
}

impl PartialEq for NativeContext {

	fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for NativeContext { }

impl Hash for NativeContext {

    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}