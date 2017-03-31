pub use self::map::u64Map;
mod map;

use error::{Error, ErrorKind, Result};
use futures::future::{self, Future};
use hardware::HardwareDevice;
use memory::{Alloc, BoxChunk, Chunk, Memory, Synch};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ops::DerefMut;
use utility;

/// A shared tensor for framework-agnostic, memory-aware, n-dimensional storage. 
///
/// A `SharedTensor` is used for the purpose of tracking the location of memory across devices 
/// for one similar piece of data. `SharedTensor` handles synchronization of memory of type `T`, by 
/// which it is parameterized, and provides the functionality for memory management across devices.
///
/// `SharedTensor` holds copies and their version numbers. A user can request any number of
/// immutable `Tensor`s or a single mutable `Tensor` (enforced by borrowck). It's possible to 
/// validate at runtime that tensor data is initialized when a user requests a tensor for reading
/// and skip the initialization check if a tensor is requested only for writing.
///
/// ## Terminology
///
/// In Parenchyma, multidimensional Rust arrays represent tensors. A vector, a tensor with a 
/// rank of 1, in an n-dimensional space is represented by a one-dimensional Rust array of 
/// length n. Scalars, tensors with a rank of 0, are represented by numbers (e.g., `3`). An array of 
/// arrays, such as `[[1, 2, 3], [4, 5, 6]]`, represents a tensor with a rank of 2.
///
/// A tensor is essentially a generalization of vectors. A Parenchyma shared tensor tracks the memory 
/// copies of the numeric data of a tensor across the device of the backend and manages:
///
/// * the location of these memory copies
/// * the location of the latest memory copy and
/// * the synchronization of memory copies between devices
///
/// This is important, as it provides a unified data interface for executing tensor operations 
/// on CUDA, OpenCL and common host CPU.
///
/// ## Read/Write
///
/// The methods `read`, `read_write`, and `write` use `unsafe` to extend the lifetime of the returned 
/// reference to the internally owned memory chunk. The borrowck guarantees that the shared tensor 
/// outlives all of its tensors, and that there is only one mutable borrow. 
///
/// ### TODO:
///
/// * Therefore, we only need to make sure the memory locations won't be dropped or moved while 
/// there are active tensors.
///
/// * Contexts and devices should also remain in scope, although it's unlikely that a context will
/// have the same ID as a previous context...
///
/// ### Summary
///
/// If the caller reads (`read` or `read_write`), memory is synchronized and the latest memory 
/// object is returned. If the caller mutably borrows memory (`read_write` and `write`), it's expected 
/// that the memory will be overwritten, so the other memory locations are immediately considered 
/// outdated.
pub struct SharedTensor<T = f32> {
    /// The shape of the shared tensor.
    shape: TensorShape,
    /// A list of chunks wrapped in a `RefCell`.
    chunks: RefCell<Vec<BoxChunk>>,
    /// Indicates whether or not memory is synchronized (synchronization state).
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
    versions: u64Map,
    /// A marker for `T`.
    phantom: PhantomData<T>,
}

/// This block contains the read/write/auto-sync logic.
impl<T> SharedTensor<T> where HardwareDevice: Alloc<T>, Chunk: Synch<T> {

    /// Constructs a new `SharedTensor` with a `shape`.
    pub fn new<I>(shape: I) -> Self where I: Into<TensorShape> {

        let shape = shape.into();
        let chunks = RefCell::new(vec![]);
        let versions = u64Map::new();

        SharedTensor { shape, chunks, versions, phantom: PhantomData }
    }

    /// Constructs a new `SharedTensor` containing the provided `data` with a `shape`.
    pub fn with<A, I>(device: &A, shape: I, data: Vec<T>) -> Result<Self>
        where A: AsRef<HardwareDevice>,
              I: Into<TensorShape>,
              {

        let shape = shape.into();
        shape.check(&data)?;
        let device = device.as_ref();
        let chunk = device.alloc_place(data)?;
        let chunks = RefCell::new(vec![chunk]);
        let versions = u64Map::with(1 << 0);

        Ok(SharedTensor { shape, chunks, versions, phantom: PhantomData })
    }

    /// Allocates memory on the active device and tracks it.
    pub fn prealloc<A, I>(device: &A, shape: I) -> Result<Self> 
        where A: AsRef<HardwareDevice>, 
              I: Into<TensorShape> 
              {

        let shape = shape.into();
        let capacity_bytes = utility::allocated::<T>(shape.capacity);
        let copy = device.as_ref().prealloc(capacity_bytes)?;
        let chunks = RefCell::new(vec![copy]);
        let versions = u64Map::with(1 << 0); // ? TODO consider it up-to-date?

        Ok(SharedTensor { shape, chunks, versions, phantom: PhantomData })
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
    pub fn realloc<I>(&mut self, sh: I) where I: Into<TensorShape> {
        self.chunks.borrow_mut().clear();
        self.versions.set(0);
        self.shape = sh.into();
    }

    /// Drops memory allocation on the specified device. Returns error if no memory has been 
    /// allocated on this device.
    ///
    // TODO FIXME: synchronize memory elsewhere if possible..?
    // TODO silence the error..?
    pub fn dealloc<A>(&mut self, device: &A) -> Result<BoxChunk> where A: AsRef<HardwareDevice> {

        match self.position(device) {
            Some(i) => {
                let chunk = self.chunks.borrow_mut().remove(i);

                let version = self.versions.get();
                let mask = (1 << i) - 1;
                let lower = version & mask;
                let upper = (version >> 1) & (!mask);
                self.versions.set(lower | upper);

                Ok(chunk)
            },

            _ => Err(ErrorKind::AllocatedMemoryNotFoundForDevice.into())
        }
    }

    /// Change the shape of the Tensor.
    ///
    /// # Returns
    ///
    /// Returns an error if the size of the new shape is not equal to the size of the old shape.
    /// If you want to change the shape to one of a different size, use `SharedTensor::realloc`.
    pub fn reshape<I>(&mut self, sh: I) -> Result where I: Into<TensorShape> {
        let shape = sh.into();

        if shape.capacity() != self.shape.capacity() {
            return Err(ErrorKind::InvalidReshapedTensorSize.into());
        }

        self.shape = shape;

        Ok(())
    }

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
    pub fn read<'shared, A>(&'shared self, device: &A) -> Result<&'shared Memory> 
        where A: AsRef<HardwareDevice> {

        // let i = self.autosync(device, false)?;

        // let chunks = self.chunks.borrow();

        // let (_, memory) = chunks[i].this();

        // Ok(unsafe { lifetime::extend::<'shared>(memory) })

        unimplemented!()
    }

    /// Sync if necessary
    ///
    /// TODO: 
    ///
    /// * Choose the best source to copy data from.
    ///      That would require some additional traits that return costs for transferring data 
    ///      between different backends.
    ///
    /// note: Typically, there would be transfers between `Native` <-> `GPU` in foreseeable 
    /// future, so it's best to not over-engineer here.
    pub fn autosync<A>(&self, device: &A, tick: bool) -> Result<usize> where A: AsRef<HardwareDevice> {
        if self.versions.empty() {
            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create(device)?;

        if !self.versions.contains(i) {
            self.synchronize(i).wait()?;
        }

        if tick {
            self.versions.set(1 << i);
        } else {
            self.versions.insert(i);
        }

        Ok(i)
    }

    fn synchronize<'a>(&'a self, destination_index: usize) -> Box<Future<Item=(), Error=Error> + 'a> {

        let source_index = self.versions.latest() as usize;
        assert_ne!(source_index, u64Map::CAPACITY);

        // We need to borrow two different Vec elements: `src` and `mut dst`.
        // Borrowck doesn't allow to do it in a straightforward way, so here is workaround.
        assert_ne!(source_index, destination_index);
        let mut borrowed_copies = self.chunks.borrow_mut();

        let (source, destination) = unsafe {
            if source_index < destination_index {
                let (left, right) = borrowed_copies.split_at_mut(destination_index);
                (lifetime::extend_mut::<'a>(&mut left[source_index]), lifetime::extend_mut::<'a>(&mut right[0]))
            } else {
                let (left, right) = borrowed_copies.split_at_mut(source_index);
                (lifetime::extend_mut::<'a>(&mut right[0]), lifetime::extend_mut::<'a>(&mut left[destination_index]))
            }
        };

        // Backends may define transfers asymmetrically. E.g. CUDA may know how to transfer to and 
        // from Native backend, while Native may know nothing about CUDA at all. So if first 
        // attempt fails we change order and try again.

        if source.syncable(destination.this().0) {
            source.synchronize_out(destination.deref_mut())
        } else if destination.syncable(source.this().0) {
            destination.synchronize_in(source.deref_mut())
        } else {
            Box::new(future::err(ErrorKind::NoAvailableSynchronizationRouteFound.into()))
        }

        // TODO refactor

        // TODO: try transfer indirectly via Native backend

        // A last resort when no synchronization route is available.
        //
        // Sync to host -> sync to/from host to/from `chunk`
    }

    fn position<A>(&self, device: A) -> Option<usize> where A: AsRef<HardwareDevice> {

        let device = device.as_ref();

        self.chunks.borrow().iter()
            .enumerate()
            .filter(|&(i, chunk)| chunk.located_on(device))
            .map(|(i, _)| i)
            .nth(0)
    }

    fn get_or_create<A>(&self, device: &A) -> Result<usize> where A: AsRef<HardwareDevice> {

        if let Some(i) = self.position(device) {
            return Ok(i);
        }

        if self.chunks.borrow().len() == u64Map::CAPACITY {
            return Err(ErrorKind::BitMapCapacityExceeded.into());
        }

        let capacity_bytes = utility::allocated::<T>(self.shape.capacity);
        let chunk = device.as_ref().prealloc(capacity_bytes)?;
        self.chunks.borrow_mut().push(chunk);

        Ok(self.chunks.borrow().len() - 1)
    }
}

/// Describes the shape of a tensor.
///
/// **note**: `From` conversion implementations are provided for low-rank shapes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorShape {
    /// The maximum number of components the associated tensor can store.
    ///
    /// # Example
    ///
    /// ```{.text}
    /// // The following tensor has 9 components
    ///
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    capacity: usize,
    /// A list of numbers with each representing the dimension at each index.
    ///
    /// # Example
    ///
    /// The following tensor has a shape of `[2, 1]`:
    ///
    /// ```{.text}
    /// [[a], [b]]
    /// ```
    dimsizes: Vec<usize>,

    // /// The stride tells the tensor how to interpret its flattened representation.
    // stride: Vec<usize>,
}

impl TensorShape {

    /// Checks that the shape of the provided `data` is compatible.
    pub fn check<T>(&self, data: &[T]) -> Result {
        if self.capacity != data.len() {
            return Err(ErrorKind::IncompatibleShape.into()); 
        }

        Ok(())
    }

    /// Returns the `dimensions`.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimsizes
    }

    /// Returns the number of elements the tensor can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the total number of indices required to identify each component uniquely (i.e, the
    /// tensor's rank, degree, or order).
    ///
    /// # Example
    ///
    /// The following tensor has a rank of 2:
    ///
    /// ```{.text}
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    pub fn rank(&self) -> usize {
        self.dimsizes.len()
    }
}

macro_rules! shape {
    ($($length:expr),*) => ($(impl From<[usize; $length]> for TensorShape {
        fn from(array: [usize; $length]) -> TensorShape {

            TensorShape {
                capacity: array.iter().fold(1, |acc, &dims| acc * dims),
                dimsizes: array.to_vec(),
            }
        }
    })*)
}

shape!(0, 1, 2, 3, 4, 5, 6);

impl From<Vec<usize>> for TensorShape {

    fn from(vector: Vec<usize>) -> TensorShape {

        TensorShape {
            capacity: vector.iter().fold(1, |acc, &dims| acc * dims),
            dimsizes: vector,
        }
    }
}

impl<'slice> From<&'slice [usize]> for TensorShape {

    fn from(slice: &[usize]) -> TensorShape {
        TensorShape {
            capacity: slice.iter().fold(1, |acc, &dims| acc * dims),
            dimsizes: slice.to_owned(),
        }
    }
}

impl From<usize> for TensorShape {

    fn from(dimensions: usize) -> TensorShape {
        TensorShape {
            capacity: dimensions,
            dimsizes: vec![dimensions],
        }
    }
}

mod lifetime {
    use std::mem;

    pub unsafe fn extend<'a, 'b, T>(t: &'a T) -> &'b T where T: ?Sized {
        mem::transmute::<&'a T, &'b T>(t)
    }

    pub unsafe fn extend_mut<'a, 'b, T>(t: &'a mut T) -> &'b mut T where T: ?Sized {
        mem::transmute::<&'a mut T, &'b mut T>(t)
    }
}