//! A `SharedTensor` is used for the purpose of tracking the location of memory across devices 
//! for one similar piece of data. `SharedTensor` handles synchronization of memory of type `T`, by 
//! which it is parameterized, and provides the functionality for memory management across devices.
//!
//! `SharedTensor` holds copies and their version numbers. A user can request any number of
//! immutable `Tensor`s or a single mutable `Tensor` (enforced by borrowck). It's possible to 
//! validate at runtime that tensor data is initialized when a user requests a tensor for reading
//! and skip the initialization check if a tensor is requested only for writing.
//!
//! ## Terminology
//!
//! In Parenchyma, multidimensional Rust arrays represent tensors. A vector, a tensor with a 
//! rank of 1, in an n-dimensional space is represented by a one-dimensional Rust array of 
//! length n. Scalars, tensors with a rank of 0, are represented by numbers (e.g., `3`). An array of 
//! arrays, such as `[[1, 2, 3], [4, 5, 6]]`, represents a tensor with a rank of 2.
//!
//! A tensor is essentially a generalization of vectors. A Parenchyma shared tensor tracks the memory 
//! copies of the numeric data of a tensor across the device of the backend and manages:
//!
//! * the location of these memory copies
//! * the location of the latest memory copy and
//! * the synchronization of memory copies between devices
//!
//! This is important, as it provides a unified data interface for executing tensor operations 
//! on CUDA, OpenCL and common host CPU.
//!
//! ## Read/Write
//!
//! The methods `read`, `read_write`, and `write` use `unsafe` to extend the lifetime of the returned 
//! reference to the internally owned memory chunk. The borrowck guarantees that the shared tensor 
//! outlives all of its tensors, and that there is only one mutable borrow. 
//!
//! ### TODO:
//!
//! * Therefore, we only need to make sure the memory locations won't be dropped or moved while 
//! there are active tensors.
//!
//! * Contexts and devices should also remain in scope, although it's unlikely that a context will
//! have the same ID as a previous context...
//!
//! ### Summary
//!
//! If the caller reads (`read` or `read_write`), memory is synchronized and the latest memory 
//! object is returned. If the caller mutably borrows memory (`read_write` and `write`), it's expected 
//! that the memory will be overwritten, so the other memory locations are immediately considered 
//! outdated.

pub use self::into_tensor::IntoTensor;
pub use self::tensor_shape::TensorShape;
pub use self::tensor_type::TensorType;

mod into_tensor;
mod tensor_map;
mod tensor_memories;
mod tensor_shape;
mod tensor_type;
mod utility;

use std::cell::RefCell;
use std::fmt;
use std::ops::{Deref, DerefMut};

use self::tensor_map::TensorMap;
use self::tensor_memories::TensorMemories;

use super::compute_device::{Allocate, ComputeDevice};
use super::error::{Error, ErrorKind, Result};
use super::memory::{Memory, TransferDirection};

/// A shared tensor for framework-agnostic, memory-aware, n-dimensional storage.
pub struct SharedTensor<T = f32> {
    /// A list of memory copies wrapped in a `RefCell`.
    memories: TensorMemories<T>,
    /// The shape of the shared tensor.
    shape: TensorShape,
    /// Tracks unsynchronized/synchronized memory (synchronization state).
    ///
    /// There are only two possible states:
    ///
    /// * Outdated or uninitialized
    /// * Up-to-date
    ///
    /// The _bools_ are packed into an integer and the integer can be set/reset in one operation.
    /// The integer type used is `u64` (used to store bitmasks), therefore the maximum number of 
    /// memories is 64.
    ///
    /// note: `BitSet` can be used instead (for the purpose of having multiple nodes in a cluster?) 
    /// of a single integer in exchange for some runtime cost and will likely be allowed in the 
    /// near future via a parameter at the type level or a feature flag.
    ///
    /// `u64` requires no extra allocations and no access indirection, but is limited. `BitSet` is
    /// slower.
    ///
    /// note: currently relies on the associated constant `u64Map::CAPACITY`, though there are 
    /// plans to add an associated constant or `const fn` to `u64` itself.
    ///
    /// Each time a `Tensor` is mutably borrowed from `SharedTensor`, the version of the 
    /// corresponding memory is _ticked_ or increased. The value `0` means that the memory object 
    /// at that specific location is uninitialized or outdated.
    synch_map: TensorMap,
}

impl<I,T> From<I> for SharedTensor<T> 
    where I: Into<TensorShape>, 
          T: 'static + Clone + ::num::Zero, 
          ComputeDevice: Allocate<T> {

    /// Creates an empty shared tensor with the provided `shape`.
    fn from(shape: I) -> SharedTensor<T> {
        let tensor_shape: TensorShape = shape.into();
        let length = tensor_shape.capacity();
        SharedTensor::<T>::with(tensor_shape, vec![T::zero(); length]).unwrap()
    }
}

impl<T> SharedTensor<T> where T: 'static, ComputeDevice: Allocate<T> {
    pub fn scalar(value: T) -> Result<SharedTensor<T>> {
        Ok(array![value].into())
    }
    /// Constructs a new  shared tensor containing the provided `data` with a `shape`.
    pub fn with<I, V>(shape: I, data: V) -> Result<SharedTensor<T>> 
        where I: Into<TensorShape>, 
              V: Into<Vec<T>> {
        use ndarray::ArrayBase;
        use super::frameworks::NativeMemory;

        let shape: TensorShape = shape.into();

        let memory = NativeMemory(
            ArrayBase::from_shape_vec(shape.dimensions(), data.into())
                .map_err(|e| Error::new(ErrorKind::IncompatibleShape, e))?
        );
        let memories = RefCell::new(vec![box memory as Box<Memory<T>>]);
        let synch_map = TensorMap::with(1 << 0);

        Ok(SharedTensor { memories, shape, synch_map })
    }
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }
    /// Changes the shape of the Tensor.
    ///
    /// # Returns
    ///
    /// Returns an error if the size of the new shape is not equal to the size of the old shape.
    /// If you want to change the shape to one of a different size, use `SharedTensor::realloc`.
    pub fn reshape<I>(&mut self, shape: I) -> Result where I: Into<TensorShape> {
        let shape = shape.into();

        if shape.capacity != self.shape.capacity {
            let e: Error = ErrorKind::InvalidReshapedTensorSize.into();
            return Err(e);
        }

        Ok(self.shape = shape)
    }
    /// Changes the capacity and shape of the tensor.
    ///
    /// **Caution**: Drops all copies, **including** the ones that are on the current device.
    ///
    /// `SharedTensor::reshape` should be preferred to this method if the size of the old and 
    /// new shape are identical because it will not reallocate memory.
    ///
    /// ## TODO
    ///
    /// Should the copies on the current device remain and be reallocated (e.g., 
    /// Collenchyma's implementation)?
    pub fn resize<S>(&mut self, shape: S) -> Result 
        where S: Into<TensorShape>, T: Clone + ::num::Zero {
        // self.memories.borrow_mut().clear();
        // self.synch_map.set(0);
        // self.shape = shape.into();
        *self = SharedTensor::from(shape);
        Ok(())
    }
    /// Synchronizes data with the active device on the specified `backend`.
    pub fn synch<I>(&self, codev: &ComputeDevice) -> Result {
        let _ = self.autosync(codev, false)?;
        Ok(())
    }
    /// Synchronizes with the active device on the specified `backend` and then 
    /// returns the `SharedTensor`.
    pub fn synchronize_return<I>(self, codev: &ComputeDevice) -> Result<SharedTensor<T>> {
        let _ = self.autosync(codev, false)?;
        Ok(self)
    }
    /// Changes the capacity and shape of the tensor.
    ///
    /// **Caution**: Drops all copies, **including** the ones that are on the current device.
    ///
    /// `SharedTensor::reshape` should be preferred to this method if the size of the old and 
    /// new shape are identical because it will not reallocate memory.
    ///
    /// ## TODO
    ///
    /// Should the copies on the current device remain and be reallocated (e.g., 
    /// Collenchyma's implementation)?
    pub unsafe fn realloc<S>(&mut self, s: S) where S: Into<TensorShape> {
        self.memories.borrow_mut().clear();
        self.synch_map.set(0);
        self.shape = s.into();
    }
    /// Constructs a new `SharedTensor` with uninitialized memory.
    ///
    /// **Consider initializing memory via the associated `new` function.** 
    pub unsafe fn uninitialized<S>(s: S) -> SharedTensor<T> where S: Into<TensorShape> {
        SharedTensor {
            memories: RefCell::new(vec![]), 
            shape: s.into(), 
            synch_map: TensorMap::new(),
        }
    }
}

impl<T> SharedTensor<T> where T: 'static, ComputeDevice: Allocate<T> {
    /// View an underlying tensor for reading on the active device.
    ///
    /// This method can fail if memory allocation fails or if no memory is initialized.
    /// The borrowck guarantees that the shared tensor outlives all of its tensors.
    ///
    /// Summary:
    ///
    /// 1) Check if there is initialized data anywhere
    /// 2) Lookup memory and its version for `device`, allocate it if it doesn't exist
    /// 3) Check version, if it's old, synchronize
    pub fn reference<'a, M>(&self, codev: &ComputeDevice) -> Result<&'a M> 
        where M: Memory<T> {

        let i = self.autosync(codev, false)?;
        let borrowed_copies = self.memories.borrow();
        let c = &borrowed_copies[i];
        let memory = unsafe { utility::extend_lifetime(c.deref()) };

        memory.downcast_ref::<M>().ok_or(ErrorKind::MemoryDowncasting.into())
    }
    /// View an underlying tensor for reading and writing on the active device. The memory 
    /// location is set as the latest.
    ///
    /// This method can fail is memory allocation fails or if no memory is initialized.
    ///
    /// Summary:
    ///
    /// 1) Check if there is initialized data anywhere
    /// 2) Lookup memory and its version for `device`, allocate it if it doesn't exist
    /// 3) Check version, if it's old, synchronize
    /// 4) Increase memory version and latest_version
    pub fn mut_reference<'a, M>(&mut self, codev: &ComputeDevice) -> Result<&'a mut M> 
        where M: Memory<T> {

        let i = self.autosync(codev, true)?;
        let mut borrowed_copies = self.memories.borrow_mut();
        let c = &mut borrowed_copies[i];
        let memory = unsafe { utility::extend_lifetime_mut(c.deref_mut()) };

        memory.downcast_mut::<M>().ok_or(ErrorKind::MemoryDowncasting.into())
    }
    /// View an underlying tensor for writing only.
    ///
    /// This method skips synchronization and initialization logic since its data will
    /// be overwritten anyway. The caller must initialize all elements contained in the tensor. This
    /// convention isn't enforced, but failure to do so may result in undefined data later.
    ///
    /// Summary:
    ///
    /// 1) *Skip initialization check
    /// 2) Lookup memory and its version for `device`, allocate it if it doesn't exist
    /// 3) *Skip synchronization
    /// 4) Increase memory version and latest_version
    ///
    /// TODO
    ///
    /// * Add an `invalidate` method:
    ///
    ///     If the caller fails to overwrite memory, it must call `invalidate` to return the vector
    ///     to an uninitialized state.
    pub fn mut_reference_unsynched<'a, M>(&mut self, codev: &ComputeDevice) -> Result<&'a mut M> 
        where M: Memory<T> {

        let i = self.fetchsert(codev)?;
        self.synch_map.set(1 << i);
        let mut borrowed_copies = self.memories.borrow_mut();
        let c = &mut borrowed_copies[i];
        let memory = unsafe { utility::extend_lifetime_mut(c.deref_mut()) };

        memory.downcast_mut::<M>().ok_or(ErrorKind::MemoryDowncasting.into())
    }
    /// Returns an immutable reference to a slice synchronized with the native/host CPU.
    ///
    /// note: Take a look at the documentation for the `reference` method.
    pub fn as_slice<'a>(&self) -> Result<&'a [T]> {
        use super::frameworks::{HOST, NativeMemory};

        self.reference::<NativeMemory<T>>(&HOST)?.as_slice()
            .ok_or(Error::new(
                ErrorKind::Other, "the array’s data is not contiguous and in standard order"))
    }
    /// Returns a mutable reference to a slice synchronized with the native/host CPU.
    ///
    /// note: Take a look at the documentation for the `mut_reference` method.
    pub fn as_mut_slice<'a>(&mut self) -> Result<&'a mut [T]> {
        use super::frameworks::{HOST, NativeMemory};

        self.mut_reference::<NativeMemory<T>>(&HOST)?.as_slice_mut()
            .ok_or(Error::new(
                ErrorKind::Other, "the array’s data is not contiguous and in standard order"))
    }
    /// Returns a mutable reference to the underlying buffer that may be unsynchronized.
    ///
    /// note: Take a look at the documentation for the `mut_reference_unsynched` method.
    pub fn as_mut_slice_unsynched<'a>(&mut self) -> Result<&'a mut [T]> {
        use super::frameworks::{HOST, NativeMemory};

        self.mut_reference_unsynched::<NativeMemory<T>>(&HOST)?.as_slice_mut()
            .ok_or(Error::new(
                ErrorKind::Other, "the array’s data is not contiguous and in standard order"))
    }

    /// Write into a native Parenchyma `Memory`.
    pub fn write_slice(&mut self, data: &[T]) -> Result where T: Copy {
        self.write_offset_slice(data, 0)
    }
    
    /// Write into a native Parenchyma `Memory` with an `offset`.
    pub fn write_offset_slice(&mut self, data: &[T], offset: usize) -> Result where T: Copy {
        let buf = self.as_mut_slice_unsynched()?;

        for (i, datum) in data.iter().enumerate() {
            buf[i + offset] = *datum;
        }
        
        Ok(())
    }

    /// Write into a native Parenchyma `Memory`.
    pub fn write_iter<I>(&mut self, data: I) -> Result where T: Copy, I: Iterator<Item=T> {
        self.write_offset_iter(data, 0)
    }

    /// Write into a native Parenchyma `Memory` with an `offset`.
    pub fn write_offset_iter<I>(&mut self, data: I, offset: usize) -> Result 
        where T: Copy, 
              I: Iterator<Item=T> {
        let buf = self.as_mut_slice_unsynched()?;

        for (i, datum) in data.enumerate() {
            buf[i + offset] = datum;
        }
        
        Ok(())
    }
}

impl<T> SharedTensor<T> where T: 'static, ComputeDevice: Allocate<T> {
    /// `autosync` synchronizes data only if necessary.
    ///
    /// **TODO**: 
    ///
    /// * Choose the best source to copy data from.
    ///     * That would require some additional traits that return costs for transferring data 
    ///       between different backends.
    ///     * Typically, there would be transfers between `Native` <-> `GPU` in foreseeable 
    ///       future, so it's best to not over-engineer here.
    fn autosync<'a>(&'a self, codev: &ComputeDevice, overwritable: bool) -> Result<usize> {

        if self.synch_map.empty() {
            // TODO auto initialize
            Err(ErrorKind::UninitializedMemory.into())
        } else {
            let i = self.fetchsert(codev).and_then(|i| 
                if self.synchronized(i) {
                    Ok(i)
                } else {
                    self.synchronize(i).map(|_| i)
                }
            )?;

            if overwritable {
                // the memory is expected to been overwritten -> set the active 
                // copy at `i` as the latest copy
                self.synch_map.set(1 << i);
            } else {
                // the copy at `i` has been synced
                self.synch_map.insert(i);
            }

            Ok(i)
        }
    }
    /// Synchronizes the memory at the provided index.
    fn synchronize<'a>(&'a self, destination_index: usize) -> Result {
        let source_index = self.synch_map.latest() as usize;
        assert_ne!(source_index, TensorMap::CAPACITY);

        // We need to borrow two different Vec elements: `src` and `mut dst`.
        // Borrowck doesn't allow to do it in a straightforward way, so here is workaround.
        assert_ne!(source_index, destination_index);
        let mut borrowed_copies = self.memories.borrow_mut();

        let (source, destination) = unsafe {
            if source_index < destination_index {
                let (left, right) = borrowed_copies.split_at_mut(destination_index);

                (
                    utility::extend_lifetime_mut(&mut left[source_index]), 
                    utility::extend_lifetime_mut(&mut right[0])
                )
            } else {
                let (left, right) = borrowed_copies.split_at_mut(source_index);

                (
                    utility::extend_lifetime_mut(&mut right[0]), 
                    utility::extend_lifetime_mut(&mut left[destination_index])
                )
            }
        };

        // Backends may define transfers asymmetrically. E.g. CUDA may know how to transfer to and 
        // from Native backend, while Native may know nothing about CUDA at all. So if first 
        // attempt fails we change order and try again.

        match source.transfer(TransferDirection::TransferOut, destination.deref_mut()) {
            Err(ref e) if e.kind() == ErrorKind::NoAvailableSynchronizationRouteFound => {
                destination.transfer(TransferDirection::TransferIn, source.deref_mut())
            }

            r @ _ => r
        }

        // TODO refactor

        // TODO: try transfer indirectly via Native backend

        // A last resort when no synchronization route is available.
        //
        // Sync to host -> sync to/from host to/from `chunk`
    }
    /// Returns true if the memory at the provided `index` is in sync.
    fn synchronized(&self, index: usize) -> bool {
        self.synch_map.contains(index)
    }
    /// Returns the index of the device that matches the provided `context`'s active device, returns
    /// `None` if a match is not found.
    fn position(&self, codev: &ComputeDevice) -> Option<usize> {
        self.memories.borrow().iter()
            .enumerate()
            .filter(|&(_, memory_copy)| memory_copy.synchronized(codev))
            .map(|(i, _)| i)
            .nth(0)
    }
    /// Returns the index of the device that matches the provided `backend`'s active.
    ///
    /// **note**: A copy is created if a matching one is not found.
    fn fetchsert(&self, codev: &ComputeDevice) -> Result<usize> {
        if let Some(i) = self.position(codev) {
            Ok(i)
        } else {
            if self.memories.borrow().len() == TensorMap::CAPACITY {
                Err(ErrorKind::CapacityExceeded.into())
            } else {

                // pass in the size of the allocated memory in bytes.
                let m = codev.allocate(&self.shape)?;
                self.memories.borrow_mut().push(m);

                Ok(self.memories.borrow().len() - 1)
            }
        }
    }
}

impl<T> fmt::Debug for SharedTensor<T> where T: fmt::Debug + 'static, ComputeDevice: Allocate<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use super::frameworks::{HOST, NativeMemory};

        write!(f, "{:?}", self.reference::<NativeMemory<T>>(&HOST).unwrap().0) // TODO
    }
}

impl<T> fmt::Display for SharedTensor<T> where T: fmt::Display + 'static, ComputeDevice: Allocate<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use super::frameworks::{HOST, NativeMemory};

        write!(f, "{}", self.reference::<NativeMemory<T>>(&HOST).unwrap().0) // TODO
    }
}

// -------------
pub fn reference<'a, T, M>(t: &SharedTensor<T>, codev: &ComputeDevice) -> Result<&'a M> 
    where   T: 'static, ComputeDevice: Allocate<T>,
            M: Memory<T> {
    t.reference(codev)
}

pub fn mut_reference<'a, T, M>(t: &mut SharedTensor<T>, codev: &ComputeDevice) -> Result<&'a mut M> 
    where   T: 'static, ComputeDevice: Allocate<T>,
            M: Memory<T> {
    t.mut_reference(codev)
}

pub fn mut_reference_unsynched<'a, T, M>(t: &mut SharedTensor<T>, codev: &ComputeDevice) 
    -> Result<&'a mut M> 
    where   T: 'static, ComputeDevice: Allocate<T>,
            M: Memory<T> {
    t.mut_reference_unsynched(codev)
}