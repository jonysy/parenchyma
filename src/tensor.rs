use std::{convert, mem};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

use super::{Backend, Buffer};
use super::error::{ErrorKind, Result};

/// A shared tensor for framework-agnostic, memory-aware, n-dimensional storage. 
///
/// A `SharedTensor` is used for the purpose of tracking the location of memory across devices 
/// for one similar piece of data. `SharedTensor` handles synchronization of memory of type `T`, by 
/// which it is parameterized, and provides the functionality for memory management across devices.
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
#[derive(Debug)]
pub struct SharedTensor<T> {
    /// The shape of the shared tensor.
    pub shape: Shape,
    /// A vector of buffers.
    buffers: RefCell<Vec<(Location, Buffer<T>)>>,
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

impl<T> SharedTensor<T> /* TODO where T: Scalar | Float */ {

    /// Constructs a new `SharedTensor`.
    pub fn new<I>(sh: I) -> Result<Self> where I: Into<Shape> {

        let shape = sh.into();

        Ok(SharedTensor {
            shape,
            buffers: RefCell::new(vec![]),
            versions: u64Map::new(),
            phantom: PhantomData,
        })
    }

    /// Constructs a new `SharedTensor` from the supplied `chunk` of data.
    pub fn with<I, A>(backend: &Backend, sh: I, mut chunk: A) -> Result<Self> 
        where I: Into<Shape>, 
              A: AsMut<[T]> {

        let shape = sh.into();
        let mut slice = chunk.as_mut();
        let buffer = backend.device::<T>().allocate_with(&shape, &mut slice)?;
        let vec = vec![(backend.device::<T>().location(), buffer)];
        let buffers = RefCell::new(vec);
        let versions = u64Map::with(1);

        Ok(SharedTensor { shape, buffers, versions, phantom: PhantomData })
    }

    // /// Allocate memory on a new device and track it.
    // pub fn allocate(&mut self, backend: &Backend) -> Result {

    //     let buffer = backend.device::<T>().allocate(&self.shape)?;

    //     unimplemented!()
    // }
}

/// An `impl` block for the methods `SharedTensor::read`, `SharedTensor::read_write`, 
/// and `SharedTensor::write`. The borrowck guarantees that the shared tensor outlives all of its
/// tensors, and that there is only one mutable borrow. 
///
/// [TODO](https://github.com/alexandermorozov/collenchyma/blob/decoupling/src/tensor.rs#L391):
///
/// Therefore, we only need to make sure the memory locations won't be dropped or moved while 
/// there are active tensors.
impl<T> SharedTensor<T> {

    /// View an underlying tensor for reading on the active device.
    ///
    /// `SharedTensor::read` can fail if memory allocation fails or if the tensor isn't initialized.
    /// The borrowck guarantees that the shared tensor outlives all of its tensors.
    pub fn read<'buf>(&'buf self, backend: &Backend) -> Result<Tensor<'buf, T>> {
        if self.versions.empty() {
            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(backend)?;
        self.sync_if_necessary(backend, i)?;
        self.versions.insert(i, true);

        let borrowed_buffers = self.buffers.borrow();

        let (ref location, ref buffer) = borrowed_buffers[i];

        Ok(unsafe { Tensor {
            buffer: mem::transmute(buffer),
            location: mem::transmute(location),
        }})
    }

    // /// View an underlying tensor for reading and writing on the active device. The memory 
    // /// location is set as the latest.
    // ///
    // /// `SharedTensor::read_write` can fail is memory allocation fails or if the tensor 
    // /// isn't initialized.
    // pub fn read_write<'buf>(&'buf mut self, backend: &Backend) -> Result<TensorMut<'buf>> {

    // }

    // /// View an underlying tensor for writing only.
    // ///
    // /// `SharedTensor::write` skips synchronization and initialization logic since its data will
    // /// be overwritten anyway. The caller must initialize all elements contained in the tensor. This
    // /// convention isn't enforced, but failure to do so may result in undefined data later.
    // ///
    // /// If the caller fails to overwrite memory, it must call `invalidate` to return the vector
    // /// to an uninitialized state.
    // pub fn write<'buf>(&'buf mut self, backend: &Backend) -> Result<TensorMut<'buf>> {

    // }
}

impl<T> SharedTensor<T> {

    fn get_location_index(&self, location: &Location) -> Option<usize> {

        for (i, l) in self.buffers.borrow().iter().map(|&(ref l, _)| l).enumerate() {
            if l.eq(location) {
                return Some(i);
            }
        }

        None
    }

    fn get_or_create_location_index(&self, backend: &Backend) -> Result<usize> {

        let location = backend.device::<T>().location();

        if let Some(i) = self.get_location_index(&location) {
            return Ok(i);
        }

        if self.buffers.borrow().len() == u64Map::CAPACITY {
            return Err(ErrorKind::BitMapCapacityExceeded.into());
        }

        let buffer = backend.device::<T>().allocate(&self.shape)?;
        self.buffers.borrow_mut().push((location, buffer));

        Ok(self.buffers.borrow().len() - 1)
    }

    // TODO: 
    //
    // * Choose the best source to copy data from.
    //      That would require some additional traits that return costs for transferring data 
    //      between different backends.
    //
    // Actually I think that there would be only transfers between `Native` <-> `Cuda` 
    // and `Native` <-> `OpenCL` in foreseeable future, so it's best to not over-engineer here.
    fn sync_if_necessary(&self, backend: &Backend, destination_index: usize) -> Result {

        if self.versions.get() & (1 << destination_index) != 0 {

            return Ok(());
        }

        let source_index = self.versions.latest() as usize;
        assert_ne!(source_index, u64Map::CAPACITY);

        // We need to borrow two different Vec elements: `src` and `mut dst`.
        // Borrowck doesn't allow to do it in a straightforward way, so here is workaround.

        assert_ne!(source_index, destination_index);

        let mut borrowed_buffers = self.buffers.borrow_mut();

        let (source, mut destination) = {
            if source_index < destination_index {
                let (left, right) = borrowed_buffers.split_at_mut(destination_index);
                (&left[source_index], &mut right[0])
            } else {
                let (left, right) = borrowed_buffers.split_at_mut(source_index);
                (&right[0], &mut left[destination_index])
            }
        };

        backend.device().sync_out(&source.1, &mut destination.1)

        // TODO:
        //
        // Backends may define transfers asymmetrically. E.g. CUDA may know how to transfer to and 
        // from Native backend, while Native may know nothing about CUDA at all. So if first 
        // attempt fails we change order and try again.


        // dst_loc.mem_transfer.sync_in(
        //      dst_loc.mem.as_mut(), src_loc.device.deref(),
        //      src_loc.mem.deref()).map_err(|e| e.into())

        // TODO: try transfer indirectly via Native backend
    }
}

/// Represents the location of a device.
#[derive(Debug, Eq, PartialEq)]
pub struct Location { context: isize, device: isize, framework: &'static str }

/// Describes the shape of a tensor.
#[derive(Clone, Debug)]
pub struct Shape {
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
    /// The dimensions of the tensor.
    dims: Vec<usize>,
}

impl convert::From<[usize; 1]> for Shape {

    fn from(array: [usize; 1]) -> Shape {
        let capacity = array[0];
        let rank = 1;
        let dims = array.to_vec();

        Shape { capacity, rank, dims }
    }
}

impl convert::From<[usize; 2]> for Shape {

    fn from(array: [usize; 2]) -> Shape {
        let capacity = array.iter().fold(1, |acc, &dims| acc * dims);
        let rank = 2;
        let dims = array.to_vec();

        Shape { capacity, rank, dims }
    }
}

/// An immutable view.
///
/// TODO:
///
/// Parameterization over mutability would help here..
pub struct Tensor<'a, T: 'a> {
    buffer: &'a Buffer<T>,
    location: &'a Location,
}

/// A mutable view.
pub struct TensorMut<'a, T: 'a> {
    buffer: &'a mut Buffer<T>,
    location: &'a Location,
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

    fn empty(&self) -> bool {
        self.0.get() == 0
    }

    fn insert(&self, k: usize, v: bool) {
        self.0.set(self.0.get() | ((v as u64) << k))
    }

    fn latest(&self) -> u32 {
        self.0.get().trailing_zeros()
    }
}