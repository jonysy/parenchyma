use device::ComputeDevice;
use error::{Error, ErrorKind};

use futures::future;
use futures::{Future, IntoFuture};
use futures::future::Either;

use memory::Memory;

use std::borrow::Cow;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem;

use utility::bitmap::Bitmap;
use utility::lifetime;
use utility::typedef::{BoxChunk, BoxFuture, Result};

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
pub struct SharedTensor<T> {
    /// A list of chunks wrapped in a `RefCell`.
    chunks: RefCell<Vec<BoxChunk>>,
    /// The shape of the shared tensor.
    shape: TensorShape,
    /// Indicates whether or not memory is synchronized (synchronization state).
    ///
    /// This is used as a way to keep data synchronized. There are only two possible states:
    ///
    /// * Outdated or uninitialized
    /// * Up-to-date
    ///
    /// The _bools_ are packed into an integer and the integer can be set/reset in one operation.
    /// The integer type used is `u64` (used to store bitmasks), therefore the maximum number of 
    /// copies is 64.
    ///
    /// note: `BitSet` can be used instead (for the purpose of having multiple nodes in a cluster?) 
    /// of a single integer in exchange for some runtime cost and will likely be allowed in the 
    /// near future via a parameter at the type level or a feature flag.
    ///
    /// `u64` requires no extra allocations and no access indirection, but is limited. `BitSet` is
    /// slower.
    ///
    /// note: currently relies on the associated constant `Bitmap::CAPACITY`, though there are 
    /// plans to add an associated constant or `const fn` to `u64` itself.
    ///
    /// Each time a `Tensor` is mutably borrowed from `SharedTensor`, the version of the 
    /// corresponding memory is _ticked_ or increased. The value `0` means that the memory object 
    /// at that specific location is uninitialized or outdated.
    tracker: Bitmap,
    /// A marker for `T`.
    phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> {

    /// Constructs a new, empty `SharedTensor` with the provided `shape`.
    pub fn new<I>(shape: I) -> Self where I: Into<TensorShape> {

        let chunks = RefCell::new(vec![]);
        let tracker = Bitmap::new();

        SharedTensor { shape: shape.into(), chunks, tracker, phantom: PhantomData }
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
        self.tracker.clear();
        self.shape = sh.into();
    }

    /// Drops memory allocation on the specified device. Returns error if no memory has been 
    /// allocated on this device.
    ///
    // TODO FIXME: synchronize memory elsewhere if possible..?
    // TODO silence the error..?
    pub fn dealloc<A>(&mut self, device: &A) -> Result<BoxChunk> where A: AsRef<ComputeDevice> {

        match self.position(device.as_ref()) {
            Some(i) => {
                let chunk = self.chunks.borrow_mut().remove(i);

                let version = self.tracker.get();
                let mask = (1 << i) - 1;
                let lower = version & mask;
                let upper = (version >> 1) & (!mask);
                self.tracker.set(lower | upper);

                Ok(chunk)
            },

            _ => Err(ErrorKind::AllocatedMemoryNotFoundForDevice.into())
        }
    }
}


// notes
//
// devices can have pinned and non-pinned memory
//
// opencl device may have pinned memory while an opencl device associated with a separate context may
// have a non-pinned memory buffer
//
// Accessing pinned memory is fast (really fast!) and really cheap, but comes with a few caveats:
//
// 1) Limited memory
// 2) Pinned memory may have limits depending on the OS and framework implementation
//
//
// Data synchronizations boils down to just three steps..
//
// 1) ..
// 2) ..
// 3) ..
//
// pub fn synch() -> ! {
//
//     if source.syncable(destination.this().0) {
//
//         source.connect().join(destination.connect())
//             .and_then(move |(src_view, dest_view)| {
//                 src_view.synchronize_out(dest_view)
//             })
//
//     } else if destination.syncable(source.this().0) {
//
//
//         destination.synchronize_in(source.deref_mut())
//
//     } else {
//         // transfer indirectly via Native backend
//         // last resort
//         // synch to the native host and then to the device
//         let host = ..;
//         source.synchronize_out(host);
//         destination.synchronize_in(host);
//     }
// }
impl<T> SharedTensor<T> {

    pub fn read<'shared, I: 'shared>(&'shared self, device: &I) -> 
        impl Future<Item=&'shared Memory, Error=Error> + 'shared 
        where I: AsRef<ComputeDevice> 
    {

        self.autosync(device.as_ref(), false).map(move |i| unsafe {
            let chunks = self.chunks.borrow();
            let (_, memory) = lifetime::extend_lifetime::<'shared>(&chunks[i]).this();
            memory
        })
    }

    // // 1) Pinned:  convert bytes to slice
    // // 2) Non-pinned: transfer to Host first
    // fn view<'a, T>(&'a self) -> (impl Future<Item=Tensor<'a, T>, Error=Error> + 'a) {

    //     // ..
    // }
}

impl<T> SharedTensor<T> {

    /// `autosync` synchronizes data only if necessary.
    ///
    /// **TODO**: 
    ///
    /// * Choose the best source to copy data from.
    ///     * That would require some additional traits that return costs for transferring data 
    ///       between different backends.
    ///     * Typically, there would be transfers between `Native` <-> `GPU` in foreseeable 
    ///       future, so it's best to not over-engineer here.
    pub fn autosync<'a>(&'a self, device: &ComputeDevice, writing: bool) -> 
        impl Future<Item=usize, Error=Error> + 'a
    {
        if self.tracker.empty() {
            Err(ErrorKind::UninitializedMemory.into())
        } else {
            self.get_or_create(device)
        }
        .into_future()
        .and_then(move |i| {
            if self.out_of_sync(i) {
                Err(i)
            } else {
                Ok(i)
            }
            .into_future()
            .or_else(move |i| {
                self.sync(i).map(move |_| i)
            })
            .map(move |i| {
                if writing {
                    // the memory is expected to been overwritten -> set the active 
                    // copy at `i` as the latest copy
                    self.tracker.set(1 << i);
                } else {
                    // the copy at `i` has been synced
                    self.tracker.insert(i);
                }
                i
            })
        })
    }
    
    fn sync(&self, index: usize) -> BoxFuture {

        unimplemented!()
    }

    fn out_of_sync(&self, index: usize) -> bool {
        !self.tracker.contains(index)
    }

    fn get_or_create(&self, device: &ComputeDevice) -> Result<usize> {

        if let Some(i) = self.position(device) {
            return Ok(i);
        }

        if self.chunks.borrow().len() == Bitmap::CAPACITY {
            return Err(ErrorKind::BitmapCapacityExceeded.into());
        }

        let capacity_bytes = mem::size_of::<T>() * self.shape.capacity;
        // let chunk = device.as_ref().prealloc(capacity_bytes)?;
        // self.chunks.borrow_mut().push(chunk);

        // Ok(self.chunks.borrow().len() - 1)

        unimplemented!()
    }

    fn position(&self, device: &ComputeDevice) -> Option<usize> {

        self.chunks.borrow().iter()
            .enumerate()
            .filter(|&(i, chunk)| chunk.located_on(device))
            .map(|(i, _)| i)
            .nth(0)
    }
}

pub struct Tensor<'a, T> where T: 'a + Clone {
    data: Cow<'a, [T]>,
    shape: &'a TensorShape,
}

pub struct TensorMut<'a, T> where T: 'a + Clone {
    data: &'a mut [T],
    shape: &'a TensorShape,
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

    // /// Checks that the shape of the provided `data` is compatible.
    // pub fn check<T>(&self, data: &[T]) -> Result {
    //     if self.capacity != data.len() {
    //         return Err(ErrorKind::IncompatibleShape.into()); 
    //     }

    //     Ok(())
    // }

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