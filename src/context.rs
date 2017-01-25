use {Device, Error, MemoryImp};
use frameworks::NativeContext;

pub enum ContextImp {
	#[cfg(feature = "native")]
	Native(NativeContext),
}

pub trait Context: Sized {

	type Memory;

	fn id(&self) -> &isize;

	//fn devices(&self) -> &[Device<Self>];

	fn alloc(&self, size: usize) -> Result<Self::Memory, Error>;

	fn sync_in(&self, 
		source: &ContextImp, 
		source_data: &MemoryImp, 
		dest_data: &mut Self::Memory) -> Result<(), Error>;
}