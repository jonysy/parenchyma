use std::any::Any;
use super::Error;

pub trait Context: 'static + Clone + Eq + Sized {
	type Memory: Any;

	// anti-pattern?
	fn allocate_memory(&self, size: usize) -> Result<Self::Memory, Error>;

	fn sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any) -> Result<(), Error>;

	fn sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any) -> Result<(), Error>;

	fn as_any(&self) -> &Any {
		self
	}
}

pub trait ObjectSafeContext {

	// anti-pattern?
	fn _allocate_memory(&self, size: usize) -> Result<Box<Any>, Error>;

	fn _sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any) -> Result<(), Error>;

	fn _sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any) -> Result<(), Error>;

	fn _as_any(&self) -> &Any;
}

impl<C> ObjectSafeContext for C where C: Context {

	// anti-pattern?
	fn _allocate_memory(&self, size: usize) -> Result<Box<Any>, Error> {
		match self.allocate_memory(size) {
			Ok(memory) => Ok(Box::new(memory)),
			Err(e) => Err(e)
		}
	}

	fn _sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any) -> Result<(), Error> {
		self.sync_in(my_memory, src_device, src_memory)
	}

	fn _sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any) -> Result<(), Error> {
		self.sync_out(my_memory, dst_device, dst_memory)
	}

	fn _as_any(&self) -> &Any {

		self.as_any()
	}
}