use ndarray::{Array, IxDyn};
use std::{convert, mem};
use std::marker::PhantomData;

use super::Backend;
use super::error::Result;

/// A native tensor.
pub type Tensor<T> = Array<T, IxDyn>;

/// A shared tensor for framework-agnostic, memory-aware, n-dimensional storage.
///
/// Container that handles synchronization of memory of type `T`, by which it is parameterized, and 
/// provides the functionality for memory management across devices.
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
/// ## Terminology
///
/// In Parenchyma, multidimensional Rust arrays represent tensors. A vector, a tensor with a 
/// rank of 1, in an n-dimensional space is represented by a one-dimensional Rust array of 
/// length n. Scalars, tensors with a rank of 0, are represented by numbers (e.g., `3`). An array of 
/// arrays, such as `[[1, 2, 3], [4, 5, 6]]`, represents a tensor with a rank of 2.
#[derive(Debug)]
pub struct SharedTensor<T> {
    /// The shape of the shared tensor.
    pub shape: Shape,
    /// A marker for `T`.
    phantom: PhantomData<T>,
}

impl<T> SharedTensor<T> /*where T: Scalar | Float */ {

    /// Constructs a new `SharedTensor`.
    pub fn new<I>(backend: &Backend, shape: I) -> Result<Self> where I: Into<Shape> {

        unimplemented!()
    }

    /// Constructs a new `SharedTensor` from the supplied `chunk` of data.
    pub fn with<I, A>(backend: &Backend, shape: I, chunk: A) -> Result<Self> 
        where I: Into<Shape>, 
              A: AsMut<[T]> {

        let device = backend.device();
    }
}

// impl<T> SharedTensor<T> {

//     // /// Returns the internal representation of the tensor.
//     // pub fn tensor(&self) -> Tensor<T> {

//     //     unimplemented!()
//     // }

//     /// Returns the size of the allocated memory in bytes.
//     pub fn allocated(&self, capacity: usize) -> usize {
//         capacity * mem::size_of::<T>()
//     }
// }

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