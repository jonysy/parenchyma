use std::mem;
use std::marker::PhantomData;
use std::ops::Deref;
use super::{RVec, U64Map};
use super::super::{Context, Framework};
use super::super::context::Action;
use super::super::error::{Error, ErrorKind, Result};

/// A shared tensor for framework-agnostic, memory-aware, n-dimensional storage.
///
/// Container that handles synchronization of `Memory` of type `T` and provides the functionality 
/// for memory management across contexts.
///
/// A tensor is essentially a generalization of vectors. A Parenchyma tensor tracks the memory 
/// copies of the numeric data of a tensor across the context of the backend and manages:
///
/// * the location of these memory copies
/// * the location of the latest memory copy and
/// * the synchronization of memory copies between contexts
///
/// This is important, as it provides a unified data interface for executing tensor operations 
/// on CUDA, OpenCL and common host CPU.
///
/// ## Terminology
///
/// In Parenchyma, multidimensional Rust arrays represent tensors. A vector, a tensor with a 
/// rank of 1, in an n-dimensional space is represented by a one-dimensional Rust array of 
/// length n. Scalars, tensors with a rank of 0, are represented by numbers (e.g., `3`). An array of 
/// arrays, such as `[[1, 2, 3], [4, 5, 6]]`, represents a tensor with a rank of 2.
///
/// ## Examples
///
/// Create a `SharedTensor` and fill it with some numbers:
///
/// ```rust
/// // TODO..
/// ```
#[derive(Debug)]
pub struct SharedTensor<T> {
    /// The total number of indices.
    ///
    /// # Example
    ///
    /// The following tensor has a rank of 2:
    ///
    /// ```ignore
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    rank: usize,
    /// The number of components.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // The following tensor has 9 components
    ///
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    capacity: usize,
    /// The dimensions of the tensor.
    shape: Vec<usize>,

    rvec: RVec,
    u64map: U64Map,

    phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> {

    /// Returns the number of elements for which the tensor has been allocated.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the allocated memory size in bytes.
    pub fn mem_size(capacity: usize) -> usize {
        mem::size_of::<T>() * capacity
    }

    // Functions `read()`, `read_write()`, `write_only()` use `unsafe` to
    // extend lifetime of returned reference to internally owned memory chunk.
    // Borrow guarantees that SharedTensor outlives all of its Tensors, and
    // there is only one mutable borrow. So we only need to make sure that
    // memory entries won't be dropped or moved while there are live Tensors.
    // It's quite easy to do: by convention we only allow to remove elements from
    // `self.entries` in methods with `&mut self`. Since we store context's memory
    // objects in a Box, reference to it won't change during Vec realentries.

    /// Get memory for reading for the specified `context`.
    ///
    /// ## Note
    ///
    /// Can fail if memory allocation fails or if tensor wasn't initialized yet.
    pub fn read<'m, C>(&'m self, context: &C) 
        -> Result<&'m <C::F as Framework>::M> where C: Context
    {

        if self.u64map.get() == 0 {

            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(context)?;
        self.sync_if_needed(i)?;
        self.u64map.set(self.u64map.get() | (1 << i));

        let re = self.rvec.re.borrow();
        let memory: &<C::F as Framework>::M = re[i].memory.deref().downcast_ref().ok_or(
            Error::new(ErrorKind::Other, "Broken invariant: wrong memory type")
        )?;

        let memory: &'m <C::F as Framework>::M = unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    /// Get memory for reading and writing for the specified `context`.
    /// Can fail if memory allocation fails, or if tensor wasn't initialized yet.
    pub fn read_write<'m, C>(&'m self, context: &C) 
        -> Result<&'m mut <C::F as Framework>::M> where C: Context
    {

        if self.u64map.get() == 0 {

            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(context)?;
        self.sync_if_needed(i)?;
        self.u64map.set(1 << i);

        let mut re = self.rvec.re.borrow_mut();
        let memory: &mut <C::F as Framework>::M = re[i].memory.as_mut().downcast_mut().ok_or(
            Error::new(ErrorKind::Other, "Broken invariant: wrong memory type")
        )?;

        let memory: &'m mut <C::F as Framework>::M = unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    /// Get memory for writing only.
    ///
    /// This function skips synchronization and initialization checks, since
    /// contents will be overwritten anyway. By convention caller must fully
    /// initialize returned memory. Failure to do so may result in use of
    /// uninitialized data later. If caller has failed to overwrite memory,
    /// for some reason, it must call `invalidate()` to return vector to
    /// uninitialized state.
    pub fn write_only<'m,  C>(&'m mut self, context: &C) 
        -> Result<&'m mut <C::F as Framework>::M> where C: Context
    {

        let i = self.get_or_create_location_index(context)?;
        self.u64map.set(1 << i);

        let mut re = self.rvec.re.borrow_mut();
        let memory: &mut <C::F as Framework>::M = re[i].memory.as_mut().downcast_mut().ok_or(
            Error::new(ErrorKind::Other, "Broken invariant: wrong memory type")
        )?;

        let memory: &'m mut <C::F as Framework>::M = unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    // FIXME: synchronize memory elsewhere if possible?
    /// Drops memory allocation on the specified context. Returns error if
    /// no memory has been allocated on this context.
    pub fn drop_context<C>(&mut self, context: &C) -> Result where C: Context {
        match self.get_location_index(context) {
            Some(i) => {
                self.rvec.remove(i);
                let up_to_date = self.u64map.get();
                let mask = (1 << i) - 1;
                let lower = up_to_date & mask;
                let upper = (up_to_date >> 1) & (!mask);
                self.u64map.set(lower | upper);

                Ok(())
            },

            _ => {
                Err(ErrorKind::AllocatedMemoryNotFoundForContext.into())
            }
        }
    }

    // ====================================================
    // ====================================================

    // TODO: chose the best source to copy data from.
    // That would require some additional traits that return costs for
    // transferring data between different backends.
    // Actually I think that there would be only transfers between
    // `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    // so it's best to not over-engineer here.
    fn sync_if_needed(&self, dst_i: usize) -> Result {

        if self.u64map.get() & (1 << dst_i) != 0 {

            return Ok(());
        }

        let src_i = self.u64map.get().trailing_zeros() as usize;

        assert!(src_i != U64Map::CAPACITY);

        // We need to borrow two different Vec elements: src and mut dst.
        // `Borrow` doesn't allow for that to be done in a straightforward way, so here is 
        // workaround.
        assert!(src_i != dst_i);

        let mut re = self.rvec.re.borrow_mut();

        let (src_loc, mut dst_loc) = if src_i < dst_i {

            let (left, right) = re.split_at_mut(dst_i);

            (&left[src_i], &mut right[0])
        } else {

            let (left, right) = re.split_at_mut(src_i);

            (&right[0], &mut left[dst_i])
        };

        // Backends may define transfers asymmetrically. E. g. CUDA may know how
        // to transfer to and from Native backend, while Native may know nothing
        // about CUDA at all. So if first attempt fails we change order and
        // try again.

        {
            let read = Action::Read {
                memory: src_loc.memory.deref(),  
                destn_context: dst_loc.context.deref(),  
                destn_memory: dst_loc.memory.as_mut()
            };

            match src_loc.context.synch(read) {
        
                Err(ref e) if e.kind() == ErrorKind::NoAvailableSynchronizationRouteFound => { },

                x @ _ => return x
            }
        }

        {
            let write = Action::Write {
                memory: dst_loc.memory.as_mut(),  
                source_context: src_loc.context.deref(),  
                source_memory: src_loc.memory.deref()
            };

            dst_loc.context.synch(write)
        }

        // TODO: try transfer indirectly via Native backend
    }

    /// Looks up `context` in `self.entries` and returns its index. If lookup fails then new 
    /// location is created and its index is returned.
    fn get_or_create_location_index<C>(&self, context: &C) -> Result<usize> where C: Context {

        if let Some(i) = self.get_location_index(context) {

            return Ok(i);
        }

        if self.rvec.len() == U64Map::CAPACITY {

            return Err(ErrorKind::BitMapCapacityExceeded.into());
        }

        let size = Self::mem_size(self.capacity);

        let memory = context.allocate_memory(size).map_err(Error::from_framework::<C::F>)?;

        self.rvec.push(Box::new(context.clone()), Box::new(memory));

        Ok(self.rvec.len() - 1)
    }

    fn get_location_index<C>(&self, ctx: &C) -> Option<usize> where C: Context {

        self.rvec.position(|e| {
            match e.context.downcast_ref::<C>() { Some(ref c) if ctx.eq(c) => true, _ => false }
        })
    }
}

impl<I, T> From<I> for SharedTensor<T> where I: Into<Vec<usize>> {

    /// Create new `SharedTensor` by allocating memory for the context.
    fn from(shape: I) -> SharedTensor<T> {

        let shape = shape.into();
        let rank = shape.len();
        let capacity = shape.iter().fold(1, |s, &a| s * a);

        SharedTensor {
            rank: rank,
            capacity: capacity,
            shape: shape,
            rvec: RVec::new(),
            u64map: U64Map::new(),
            phantom: PhantomData,
        }
    }
}