use std::mem;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

use {Alloc, ComputeDevice, Device, ErrorKind, Memory, Result, Synch};
use utility::Has;

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
#[derive(Debug)]
pub struct SharedTensor<T = f32> {
    /// The shape of the shared tensor.
    pub shape: Shape,

    /// A vector of buffers.
    copies: RefCell<Vec<(ComputeDevice, Memory<T>)>>,

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

impl<T> SharedTensor<T> where Device: Alloc<T> + Synch<T> {

    /// Constructs a new `SharedTensor` with a shape of `sh`.
    pub fn new<A>(sh: A) -> Self where A: Into<Shape> {

        let shape = sh.into();
        let copies = RefCell::new(vec![]);
        let versions = u64Map::new();

        SharedTensor { shape, copies, versions, phantom: PhantomData }
    }

    /// Constructs a new `SharedTensor` containing a `chunk` of data with a shape of `sh`.
    pub fn with<H, I>(con: &H, sh: I, chunk: Vec<T>) -> Result<Self>
        where H: Has<Device>,
              I: Into<Shape>,
              {

        let shape = sh.into();
        let device = con.get_ref();
        let buffer = device.allocwrite(&shape, chunk)?;
        let copies = RefCell::new(vec![(device.view(), buffer)]);
        let versions = u64Map::with(1);

        Ok(SharedTensor { shape, copies, versions, phantom: PhantomData })
    }

    /// Allocates memory on the active device and tracks it.
    pub fn alloc<H, I>(con: &H, sh: I) -> Result<Self> 
        where H: Has<Device>, 
              I: Into<Shape> 
              {

        let shape = sh.into();
        let device = con.get_ref();
        let buffer = device.alloc(&shape)?;
        let copies = RefCell::new(vec![(device.view(), buffer)]);
        let versions = u64Map::with(1); // ? TODO

        Ok(SharedTensor { shape, copies, versions, phantom: PhantomData })
    }

    /// Drops memory allocation on the specified device. Returns error if no memory has been 
    /// allocated on this device.
    ///
    // TODO FIXME: synchronize memory elsewhere if possible..?
    // TODO silence the error..?
    pub fn dealloc<H>(&mut self, con: &H) -> Result<Memory<T>> where H: Has<Device> {

        let device = con.get_ref();
        let location = device.view();

        match self.get_location_index(&location) {
            Some(i) => {
                let (_, memory) = self.copies.borrow_mut().remove(i);

                let version = self.versions.get();
                let mask = (1 << i) - 1;
                let lower = version & mask;
                let upper = (version >> 1) & (!mask);
                self.versions.set(lower | upper);

                Ok(memory)
            },

            _ => Err(ErrorKind::AllocatedMemoryNotFoundForDevice.into())
        }
    }

    /// Changes the capacity and shape of the tensor.
    ///
    /// **Caution**: Drops all copies which are not on the current device.
    ///
    /// `SharedTensor::reshape` is preferred over this method if the size of the old and new shape
    /// are identical because it will not reallocate memory.
    pub fn realloc<H, I>(&mut self, dev: &H, sh: I) -> Result 
        where H: Has<Device>, 
              I: Into<Shape> 
              {

        unimplemented!()
    }

    /// Change the shape of the Tensor.
    ///
    /// # Returns
    ///
    /// Returns an error if the size of the new shape is not equal to the size of the old shape.
    /// If you want to change the shape to one of a different size, use `SharedTensor::realloc`.
    pub fn reshape<I>(&mut self, sh: I) -> Result where I: Into<Shape> {
        let shape = sh.into();

        if shape.capacity() != self.shape.capacity() {
            return Err(ErrorKind::InvalidReshapedTensorSize.into());
        }

        self.shape = shape;

        Ok(())
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// This block contains the read/write/auto-sync logic.
impl<T> SharedTensor<T> where Device: Alloc<T> + Synch<T> {

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
    pub fn read<'shared, H>(&'shared self, dev: &H) -> Result<&'shared Memory<T>> 
        where H: Has<Device> {

        let i = self.autosync(dev, false)?;

        let borrowed_copies = self.copies.borrow();

        let (_, ref buffer) = borrowed_copies[i];

        let memory = unsafe { extend_lifetime::<'shared>(buffer) };

        Ok(memory)
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
    pub fn read_write<'shared, H>(&'shared mut self, dev: &H) -> Result<&'shared mut Memory<T>> 
        where H: Has<Device> {

        let i = self.autosync(dev, true)?;

        let mut borrowed_copies = self.copies.borrow_mut();

        let (_, ref mut buffer) = borrowed_copies[i];

        let memory = unsafe { extend_lifetime_mut::<'shared>(buffer) };

        Ok(memory)
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
    pub fn write<'shared, H>(&'shared mut self, con: &H) -> Result<&'shared mut Memory<T>>
        where H: Has<Device> {

        let i = self.get_or_create_location_index(con)?;
        self.versions.set(1 << i);

        let mut borrowed_copies = self.copies.borrow_mut();

        let (_, ref mut buffer) = borrowed_copies[i];

        let memory = unsafe { extend_lifetime_mut::<'shared>(buffer) };

        Ok(memory)
    }
}

impl<T> SharedTensor<T> where Device: Alloc<T> + Synch<T> {

    fn get_location_index(&self, location: &ComputeDevice) -> Option<usize> {

        for (i, l) in self.copies.borrow().iter().map(|&(ref l, _)| l).enumerate() {
            if l.eq(location) {
                return Some(i);
            }
        }

        None
    }

    fn get_or_create_location_index<H>(&self, con: &H) -> Result<usize> where H: Has<Device> {

        let device = con.get_ref();

        let location = device.view();

        if let Some(i) = self.get_location_index(&location) {
            return Ok(i);
        }

        if self.copies.borrow().len() == u64Map::CAPACITY {
            return Err(ErrorKind::BitMapCapacityExceeded.into());
        }

        let memory = device.alloc(&self.shape)?;
        self.copies.borrow_mut().push((location, memory));

        Ok(self.copies.borrow().len() - 1)
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
    pub fn autosync<H>(&self, dev: &H, tick: bool) -> Result<usize> where H: Has<Device> {
        if self.versions.empty() {
            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(dev)?;
        self.autosync_(i)?;

        if tick {
            self.versions.set(1 << i);
        } else {
            self.versions.insert(i);
        }

        Ok(i)
    }

    fn autosync_(&self, destination_index: usize) -> Result {

        if self.versions.contains(destination_index) {

            return Ok(());
        }

        let source_index = self.versions.latest() as usize;
        assert_ne!(source_index, u64Map::CAPACITY);

        // We need to borrow two different Vec elements: `src` and `mut dst`.
        // Borrowck doesn't allow to do it in a straightforward way, so here is workaround.

        assert_ne!(source_index, destination_index);

        let mut borrowed_copies = self.copies.borrow_mut();

        let (source, mut destination) = {
            if source_index < destination_index {
                let (left, right) = borrowed_copies.split_at_mut(destination_index);
                (&left[source_index], &mut right[0])
            } else {
                let (left, right) = borrowed_copies.split_at_mut(source_index);
                (&right[0], &mut left[destination_index])
            }
        };

        // TODO refactor

        // Backends may define transfers asymmetrically. E.g. CUDA may know how to transfer to and 
        // from Native backend, while Native may know nothing about CUDA at all. So if first 
        // attempt fails we change order and try again.
        match source.0.device().read(&source.1, &mut destination.0, &mut destination.1) {
            Err(ref e) if e.kind() == ErrorKind::NoAvailableSynchronizationRouteFound => { },
            ret @ _ => return ret,
        }

        destination.0.device().write(&mut destination.1, &source.0, &source.1)

        // TODO: try transfer indirectly via Native backend
    }
}

/// Describes the shape of a tensor.
#[derive(Clone, Debug)]
pub struct Shape {
    /// The number of components.
    ///
    /// # Example
    ///
    /// ```{.text}
    /// // The following tensor has 9 components
    ///
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    capacity: usize,
    /// The total number of indices.
    ///
    /// # Example
    ///
    /// The following tensor has a rank of 2:
    ///
    /// ```{.text}
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    rank: usize,
    /// The dimensions of the tensor.
    dims: Vec<usize>,
}

impl Shape {

    /// Returns the number of elements the tensor can hold without reallocating.
    pub fn capacity(&self) -> usize {

        self.capacity
    }

    /// Returns the tensor's dimensions.
    pub fn dimensions(&self) -> &[usize] {
        &self.dims
    }
}

impl From<usize> for Shape {

    fn from(n: usize) -> Shape {
        [n].into()
    }
}

impl From<[usize; 1]> for Shape {

    fn from(array: [usize; 1]) -> Shape {
        let capacity = array[0];
        let rank = 1;
        let dims = array.to_vec();

        Shape { capacity, rank, dims }
    }
}

impl From<[usize; 2]> for Shape {

    fn from(array: [usize; 2]) -> Shape {
        let capacity = array.iter().fold(1, |acc, &dims| acc * dims);
        let rank = 2;
        let dims = array.to_vec();

        Shape { capacity, rank, dims }
    }
}

impl From<[usize; 3]> for Shape {

    fn from(array: [usize; 3]) -> Shape {
        let capacity = array.iter().fold(1, |acc, &dims| acc * dims);
        let rank = 3;
        let dims = array.to_vec();

        Shape { capacity, rank, dims }
    }
}

/// A "newtype" with an internal type of `Cell<u64>`. `u64Map` uses [bit manipulation][1] to manage 
/// memory versions.
///
/// [1]: http://stackoverflow.com/a/141873/2561805
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct u64Map(Cell<u64>);

impl u64Map {
    /// The maximum number of bits in the bit map can contain.
    const CAPACITY: usize = 64;

    /// Constructs a new `u64Map`.
    fn new() -> u64Map {
        u64Map::with(0)
    }

    /// Constructs a new `u64Map` with the supplied `n`.
    fn with(n: u64) -> u64Map {
        u64Map(Cell::new(n))
    }

    fn get(&self) -> u64 {
        self.0.get()
    }

    fn set(&self, v: u64) {
        self.0.set(v)
    }

    fn empty(&self) -> bool {
        self.0.get() == 0
    }

    fn insert(&self, k: usize) {
        self.0.set(self.0.get() | (1 << k))
    }

    fn contains(&self, k: usize) -> bool {
        k < Self::CAPACITY && (self.0.get() & (1 << k) != 0)
    }

    fn latest(&self) -> u32 {
        self.0.get().trailing_zeros()
    }
}

unsafe fn extend_lifetime<'a, 'b, T>(t: &'a T) -> &'b T {
    mem::transmute::<&'a T, &'b T>(t)
}

unsafe fn extend_lifetime_mut<'a, 'b, T>(t: &'a mut T) -> &'b mut T {
    mem::transmute::<&'a mut T, &'b mut T>(t)
}