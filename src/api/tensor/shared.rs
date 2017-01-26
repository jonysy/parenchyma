use api::{Context, Memory};
use error::Error;
use linear_map::LinearMap;
use std::marker::PhantomData;
use std::mem;
use super::VecExtn;

/// `SharedTensor` handles the synchronization of memory of type `T`.
pub struct SharedTensor<M> {
	context: Context, // latest location?
	copies: LinearMap<Context, Memory>,
	de: VecExtn,
	memory: Memory, // latest copy?
	phantom: PhantomData<M>,
}

impl<M> SharedTensor<M> {

	pub fn new<I>(context: Context, de: I) -> Result<SharedTensor<M>, Error> where I: Into<VecExtn> {
		let de = de.into();
		let copy = alloc::<M>(&context, de.size())?;
		let copies = LinearMap::new();

		Ok(SharedTensor {
			context: context,
			copies: copies,
			de: de,
			memory: copy,
			phantom: PhantomData,
		})
	}

	/// Change the size and shape of the Tensor.
	///
	/// **Caution**: Drops all copies which are not on the current device.
	///
	/// `reshape` is preferred over this method if the size of the old and new shape are identical 
	/// because it will not reallocate memory.
	pub fn resize<I>(&mut self, de: I) -> Result<(), Error> where I: Into<VecExtn> {

		self.copies.clear();

		let de = de.into();
		let copy = alloc::<M>(&self.context, de.size())?;
		self.memory = copy;
		self.de = de;

		Ok(())
	}

	// pub fn sync(&mut self, context: &Context) -> Result<(), Error> {

	// 	if &self.context != context {

	// 		let latest = self.context
	// 	}

	// 	Ok(())
	// }

	pub fn get(&self, context: &Context) -> Option<&Memory> {

		if &self.context == context {

			return Some(&self.memory);
		}

		self.copies.get(context)
	}

	pub fn get_mut(&mut self, context: &Context) -> Option<&mut Memory> {

		if &self.context == context {

			return Some(&mut self.memory);
		}

		self.copies.get_mut(context)
	}

	fn sync_from_to(&mut self, source: &Context, destination: &Context) -> Result<(), Error> {

		if source != destination {

			match self.copies.get_mut(destination) {
				Some(ref mut destination_copy) => {
					destination.sync_in(&self.context, &self.memory, destination_copy)
				},

				None => Err(
					Error::missing_destination("Tensor does not hold a copy on `destination`.")
				),
			}
		} else {

			Ok(())
		}
	}

	// pub fn remove_copy(&mut self, context: &Context) -> Result<Memory, Error> {

	// 	if &self.context == context {
	// 		let first =
	// 	}
	// }

	fn return_copy(&mut self, context: Context, memory: Memory) {

		let _ = self.copies.insert(context, memory);
	}

	pub fn add(&mut self, context: &Context) -> Result<&mut Self, Error> {

		if &self.context == context {

			return Err(Error::invalid_memory_allocation(
				"Tensor already tracks memory for this device. No memory allocation."
			));
		}

		match self.copies.get(context) {
			Some(_) => Err(Error::invalid_memory_allocation(
				"Tensor already tracks memory for this device. No memory allocation."
			)),

			_ => {
				let copy = context.alloc(mem_size::<M>(self.capacity()))?;
				self.copies.insert(context.clone(), copy);
				Ok(self)
			}
		}
	}

	pub fn capacity(&self) -> usize {

		self.de.size()
	}
}

fn alloc<M>(context: &Context, size: usize) -> Result<Memory, Error> {

	context.alloc(mem_size::<M>(size))
}

fn mem_size<M>(capacity: usize) -> usize {
	mem::size_of::<M>() * capacity
}