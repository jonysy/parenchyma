use error::Error;
use super::Memory;

#[derive(Clone, Eq, Hash, PartialEq)]
pub enum Context {
	#[cfg(feature = "native")]
	Native(::frameworks::NativeContext),
}

impl Context {
	fn id(&self) -> &isize {
		match self {
			#[cfg(feature = "native")]
			&Context::Native(ref ctx) => ctx.id(),
		}
	}

	pub fn alloc(&self, size: usize) -> Result<Memory, Error> {

		let mem = match self {
			#[cfg(feature = "native")]
			&Context::Native(ref ctx) => Memory::Native(ctx.alloc(size)),
		};

		Ok(mem)
	}

	pub fn sync_in(&self, context: &Context, memory: &Memory, destination: &mut Memory)
		-> Result<(), Error> {

		match self {
			#[cfg(feature = "native")]
			&Context::Native(ref ctx) => {
				ctx.sync_in(context, memory, destination)
			},
		}
	}
}