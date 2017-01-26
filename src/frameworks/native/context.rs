use api::{Context, Memory};
use error::Error;
use std::hash::{Hash, Hasher};
use super::{NativeDevice, NativeMemory};

#[derive(Clone)]
pub struct NativeContext {
	pub devices: Vec<NativeDevice>
}

impl NativeContext {

	pub fn id(&self) -> &isize {
		static ID: isize = 0;

		&ID
	}

	pub fn alloc(&self, size: usize) -> NativeMemory {
		let b: Box<[u8]> = super::allocate_boxed_slice(size);
		NativeMemory::from(b)
	}

	pub fn sync_in(&self, context: &Context, memory: &Memory, destination: &mut Memory)
		-> Result<(), Error> {

		let native_destination = destination.as_mut_native()?;

		match context {
			&Context::Native(_) => unimplemented!(),
		}
	}
}

impl Eq for NativeContext { }

impl Hash for NativeContext {

	fn hash<H: Hasher>(&self, state: &mut H) {

		self.id().hash(state);
	}
}

impl PartialEq for NativeContext {

	fn eq(&self, _: &Self) -> bool { true }
}