use std::mem;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ops::Deref;
use super::{Location, Tensor};
use super::super::{Context, ObjectSafeContext};
use super::super::error::{ErrorKind, Result};

/// `BitMap` type for keeping track of up-to-date locations. If the number of locations provided by
/// the integer isn't enough, this type can be easily replaced with `BitSet` at the cost of a heap
/// allocation and an extra indirection on access.
type BitMap = u64;

/// The number of bits in `BitMap`. It's currently not possible to get this information from `BitMap`
/// in a clean manner, though there are plans to add a static method or an associated constant.
const BIT_MAP_SIZE: usize = 64;

pub struct SharedTensor<T> {
	tensor: Tensor,
	locations: RefCell<Vec<Location>>,
	up_to_date: Cell<BitMap>,
	phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> {

	pub fn new<I>(shape: I) -> SharedTensor<T> where I: Into<Tensor> {

		SharedTensor {
			tensor: shape.into(),
			locations: RefCell::new(vec![]),
			up_to_date: Cell::new(0),
			phantom: PhantomData
		}
	}

	/// Change the shape of the Tensor.
	pub fn reshape<I>(&mut self, shape: I) -> Result where I: Into<Tensor> {

		let tensor = shape.into();

		if self.tensor.ncomponents == tensor.ncomponents {

			self.tensor = tensor;

			Ok(())
		} else {

			Err(ErrorKind::InvalidReshapedTensorSize.into())
		}
	}

	pub fn resize<I>(&mut self, shape: I) where I: Into<Tensor> {
		self.locations.borrow_mut().clear();
		self.up_to_date.set(0);
		self.tensor = shape.into();
	}

	pub fn drop_context<C>(&mut self, context: &C) -> Result where C: Context {

		match self.get_location_index(context) {
			Some(i) => {
				self.locations.borrow_mut().remove(i);

				let up_to_date = self.up_to_date.get();
				let mask = (1 << i) - 1;
				let lower = up_to_date & mask;
				let upper = (up_to_date >> 1) & (!mask);
				self.up_to_date.set(lower | upper);

				Ok(())
			},
			_ => {
				Err(ErrorKind::AllocatedMemoryNotFoundForContext.into())
			}
		}
	}

	pub fn mem_size(capacity: usize) -> usize {

		mem::size_of::<T>() * capacity
	}

	pub fn read<'mem, C>(&'mem self, context: &C) -> Result<&'mem C::Memory> 
		where C: Context
	{
		if self.up_to_date.get() == 0 {
			return Err(ErrorKind::UninitializedMemory.into());
		}

		let i = self.get_or_create_location_index(context)?;
		self.sync_if_needed(i)?;
		self.up_to_date.set(self.up_to_date.get() | (1 << i));

		let locations = self.locations.borrow();
		let mem: &C::Memory = &locations[i].memory.deref().downcast_ref()
			.expect("Broken invariant: wrong memory type");

		let mem_mem_lifetime: &'mem C::Memory = unsafe { mem::transmute(mem) };

		Ok(mem_mem_lifetime)
	}

	pub fn read_write<'mem, C>(&'mem self, context: &C) -> Result<&'mem mut C::Memory>
		where C: Context
	{

		if self.up_to_date.get() == 0 {
			return Err(ErrorKind::UninitializedMemory.into());
		}

		let i = self.get_or_create_location_index(context)?;
		self.sync_if_needed(i)?;
		self.up_to_date.set(1 << i);

		let mut locations = self.locations.borrow_mut();
		let mem: &mut C::Memory = &mut locations[i].memory.as_mut().downcast_mut()
			.expect("Broken invariant: wrong memory type");

        let mem_mem_lifetime: &'mem mut C::Memory = unsafe { mem::transmute(mem) };

        Ok(mem_mem_lifetime)
	}

	pub fn write_only<'mem,  C>(&'mem mut self, context: &C) -> Result<&'mem C::Memory> 
		where C: Context
	{
		let i = self.get_or_create_location_index(context)?;
		self.up_to_date.set(1 << i);

		let mut locations = self.locations.borrow_mut();
		let mem: &mut C::Memory = 
			&mut locations[i]
				.memory
				.as_mut()
				.downcast_mut()
				.expect("Broken invariant: wrong memory type");

		let mem_mem_lifetime: &'mem mut C::Memory = unsafe { mem::transmute(mem) };

		Ok(mem_mem_lifetime)
	}

	fn sync_if_needed(&self, dst_i: usize) -> Result {
		if self.up_to_date.get() & (1 << dst_i) != 0 {

        	return Ok(());
    	}

    	let src_i = self.up_to_date.get().trailing_zeros() as usize;

    	assert!(src_i != BIT_MAP_SIZE);

    	assert!(src_i != dst_i);

    	let mut locations = self.locations.borrow_mut();

    	let (src_loc, mut dst_loc) = if src_i < dst_i {

    		let (left, right) = locations.split_at_mut(dst_i);

    		(&left[src_i], &mut right[0])
    	} else {

    		let (left, right) = locations.split_at_mut(src_i);

    		(&right[0], &mut left[dst_i])
    	};

    	match src_loc.context._sync_out(src_loc.memory.deref(), dst_loc.context._as_any(), dst_loc.memory.as_mut()) {
    		Err(ref e) if e.kind() == ErrorKind::NoAvailableSynchronizationRouteFound => { },

    		x @ _ => return x
    	}

    	dst_loc.context._sync_in(
    		dst_loc.memory.as_mut(), 
    		src_loc.context._as_any(),
    		src_loc.memory.deref()
    	)

    	// TODO: try transfer indirectly via Native backend
	}

	fn get_location_index<C>(&self, context: &C) -> Option<usize> where C: Context {

		for (i, location) in self.locations.borrow().iter().enumerate() {
			if let Some(ref ctx) = location.context._as_any().downcast_ref::<C>() {
				if *ctx == context {
					return Some(i);
				}
			}
		}

		None
	}

	fn get_or_create_location_index<C>(&self, context: &C) -> Result<usize> where C: Context {

		if let Some(i) = self.get_location_index(context) {
			return Ok(i)
		}

		if self.locations.borrow().len() == BIT_MAP_SIZE {
			return Err(ErrorKind::BitMapCapacityExceeded.into());
		}

		let nbytes = Self::mem_size(self.ncomponents);

		self.locations.borrow_mut().push(Location {
			memory: context._allocate_memory(nbytes)?,
			context: Box::new(context.clone()), 
		});

		Ok(self.locations.borrow().len() - 1)
	}
}

impl<T> Deref for SharedTensor<T> {

	type Target = Tensor;

	fn deref(&self) -> &Tensor {
		&self.tensor
	}
}