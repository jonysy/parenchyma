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

/// Container that handles synchronization of `Memory` of type `T` and provides the functionality 
/// for memory management across contexts.
///
/// A tensor is a potentially multi-dimensional matrix containing information about the actual 
/// data and its structure. A Parenchyma tensor tracks the memory copies of the numeric data of 
/// a tensor across the context of the backend and manages:
///
/// * the location of these memory copies
/// * the location of the latest memory copy and
/// * the synchronization of memory copies between contexts
///
/// This is important, as this provides a unified data interface for executing tensor operations 
/// on CUDA, OpenCL and common host CPU.
///
/// ## Terminology
///
/// In Parenchyma, a tensor is a homogeneous multi-dimensional matrix. A scalar value like `3` 
/// represents a tensor with a rank of 0, and a Rust array like `[1, 2, 3]` represents a tensor 
/// with a rank of 1. An array of arrays like `[[1, 2, 3], [4, 5, 6]]` represents a tensor with a 
/// rank of 2.
///
/// ## Examples
///
/// Create a `SharedTensor` and fill it with some numbers:
///
/// ```rust
///	// TODO..
/// ```
pub struct SharedTensor<T> {
	tensor: Tensor,
	locations: RefCell<Vec<Location>>,
	up_to_date: Cell<BitMap>,
	phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> {

	/// Create new `SharedTensor` by allocating memory for the context.
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

	/// Change the size and shape of the Tensor.
	pub fn replace<I>(&mut self, shape: I) where I: Into<Tensor> {
		self.locations.borrow_mut().clear();
		self.up_to_date.set(0);
		self.tensor = shape.into();
	}

	// FIXME: synchronize memory elsewhere if possible?
    /// Drops memory allocation on the specified device. Returns error if
    /// no memory has been allocated on this device.
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

	// Functions `read()`, `read_write()`, `write_only()` use `unsafe` to
    // extend lifetime of retured reference to internally owned memory chunk.
    // Borrow guarantees that SharedTensor outlives all of its Tensors, and
    // there is only one mutable borrow. So we only need to make sure that
    // memory locations won't be dropped or moved while there are live Tensors.
    // It's quite easy to do: by convention we only allow to remove elements from
    // `self.locations` in methods with `&mut self`. Since we store device's memory
    // objects in a Box, reference to it won't change during Vec reallocations.

    /// Get memory for reading for the specified `context`.
    ///
    /// ## Note
    ///
    /// Can fail if memory allocation fails or if tensor wasn't initialized yet.
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

	/// Get memory for reading and writing for the specified `context`.
	/// Can fail if memory allocation fails, or if tensor wasn't initialized yet.
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

	/// Get memory for writing only.
	///
    /// This function skips synchronization and initialization checks, since
    /// contents will be overwritten anyway. By convention caller must fully
    /// initialize returned memory. Failure to do so may result in use of
    /// uninitialized data later. If caller has failed to overwrite memory,
    /// for some reason, it must call `invalidate()` to return vector to
    /// uninitialized state.
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

	// TODO: chose the best source to copy data from.
    // That would require some additional traits that return costs for
    // transferring data between different backends.
    // Actually I think that there would be only transfers between
    // `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    // so it's best to not over-engineer here.
	fn sync_if_needed(&self, dst_i: usize) -> Result {
		if self.up_to_date.get() & (1 << dst_i) != 0 {

        	return Ok(());
    	}

    	let src_i = self.up_to_date.get().trailing_zeros() as usize;

    	assert!(src_i != BIT_MAP_SIZE);

    	// We need to borrow two different Vec elements: src and mut dst.
        // `Borrow` doesn't allow for that to be done in a straightforward way, so here is 
        // workaround.
    	assert!(src_i != dst_i);

    	let mut locations = self.locations.borrow_mut();

    	let (src_loc, mut dst_loc) = if src_i < dst_i {

    		let (left, right) = locations.split_at_mut(dst_i);

    		(&left[src_i], &mut right[0])
    	} else {

    		let (left, right) = locations.split_at_mut(src_i);

    		(&right[0], &mut left[dst_i])
    	};

    	// Backends may define transfers asymmetrically. E. g. CUDA may know how
        // to transfer to and from Native backend, while Native may know nothing
        // about CUDA at all. So if first attempt fails we change order and
        // try again.
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

	/// Looks up `context` in `self.locations` and returns its index. If lookup fails then new 
	/// location is created and its index is returned.
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

	/// Returns the number of elements for which the Tensor has been allocated.
	pub fn capacity(&self) -> usize {
		self.ncomponents
	}
}

impl<T> Deref for SharedTensor<T> {

	type Target = Tensor;

	fn deref(&self) -> &Tensor {
		&self.tensor
	}
}