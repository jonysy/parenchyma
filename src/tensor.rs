use std::{convert, mem};
use std::marker::PhantomData;

use super::Backend;
use super::error::Result;

/// The maximum number of bits in the bit map can contain.
const BIT_MAP_CAPACITY: usize = 64;

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
    /// Indicates whether or not memory is synchronized (synchronization state).
    ///
    /// There are only two possible states:
    ///
    /// `false` = outdated or uninitialized
    /// `true` = latest or up-to-date)
    ///
    /// The `bool`s are packed into an integer and the integer can be set/reset in one operation.
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
    /// note: currently relies on the `const` `MAP_CAPACITY`, though there are plans to add an 
    /// associated constant or `const fn`.
    ///
    /// Each time a `Tensor` is mutably borrowed from `SharedTensor`, the version of the 
    /// corresponding memory is _ticked_ or increased. The value `0` means that the memory object 
    /// at that specific location is uninitialized.
    versions: u64,
    /// A marker for `T`.
    phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> /* TODO where T: Scalar | Float */ {

    /// Constructs a new `SharedTensor`.
    pub fn new<I>(sh: I) -> Result<Self> where I: Into<Shape> {

        let shape = sh.into();

        unimplemented!()
    }

    /// Constructs a new `SharedTensor` from the supplied `chunk` of data.
    pub fn with<I, A>(backend: &Backend, sh: I, mut chunk: A) -> Result<Self> 
        where I: Into<Shape>, 
              A: AsMut<[T]> {

        let shape = sh.into();
        let mut slice = chunk.as_mut();
        let buffer = backend.device::<T>().allocate_with(&shape, &mut slice)?;

        unimplemented!()
    }
}

impl<T> SharedTensor<T> {

    /// Allocate memory on a new device and track it.
    pub fn allocate(&mut self, backend: &Backend) -> Result {

        let buffer = backend.device::<T>().allocate(&self.shape)?;

        unimplemented!()
    }
}

// impl<T> SharedTensor<T> {

//     // /// Returns the native internal representation of the tensor.
//     // pub fn tensor(&self) -> Tensor<T> {

//     //     unimplemented!()
//     // }
// }

// /// An immutable view.
// pub struct Tensor<'a, T: 'a> {
//     buffer: &'a Buffer<T>,
//     address: _,
//     shape: &'a Shape,
// }

// /// A mutable view.
// pub struct TensorMut<T>;

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