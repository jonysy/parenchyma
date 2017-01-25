use {Context, ContextImp, Error, MemoryImp};
use super::{Native, NativeDevice, NativeMemory};

#[derive(Clone)]
pub struct NativeContext {
	pub(super) devices: Vec<NativeDevice>,
}

impl Context for NativeContext {

	type Memory = NativeMemory;

	fn id(&self) -> &isize {
		static ID: isize = 0;

		&ID
	}

	fn alloc(&self, size: usize) -> Result<Self::Memory, Error> {
		let bx: Box<[u8]> = super::allocate_boxed_slice(size);
		Ok(NativeMemory::from(bx))
	}

	fn sync_in(&self, 
		source: &ContextImp, 
		source_data: &MemoryImp, 
		dest_data: &mut Self::Memory) -> Result<(), Error> {

		match source {
			&ContextImp::Native(_) => unimplemented!(),
		}
	}
}