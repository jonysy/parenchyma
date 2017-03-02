use std::{convert, mem};
use std::marker::PhantomData;
use super::{Backend, Buffer, Result};

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
    shape: Shape,

    // TODO
    // Location { framework_id: _, context_id: _, device_id: _ }
    latest_location: usize,
    latest_copy: Buffer,

    /// A marker for `T`.
    phantom: PhantomData<T>,
}

// /// Synchronization direction.
// pub enum Synch { In, Out, Bidirectional }

impl<T> SharedTensor<T> {

    /// Constructs a new `SharedTensor`.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the shared tensor.
    pub fn new<A>(backend: &Backend, sh: A, slice: &mut [T]) -> Result<SharedTensor<T>> 
        where A: Into<Shape> {

        let shape = sh.into();
        let size = Self::alloc_size(&shape);

        let latest_location = backend.selected_device();
        let latest_copy = backend.device().allocate(size, Some(slice))?;

        Ok(SharedTensor { shape, latest_location, latest_copy, phantom: PhantomData })
    }

    /// Returns the size of the allocated memory in bytes.
    pub fn alloc_size(shape: &Shape) -> usize {
        mem::size_of::<T>() * shape.capacity
    }

    // pub fn view(&self, device: &Device) -> Tensor<T> {
    //     unsafe {
    //         let mut buffer: Vec<T> = Vec::with_capacity(self.shape.capacity);
    //         buffer.set_len(self..shape.capacity);
    //         device.synch_out(&mut buffer)?;

    //         Tensor { buffer, shape: self.shape }
    //     }
    // }
}

/// A tensor.
#[derive(Debug)]
pub struct Tensor<T> {
    /// The internal representation of the tensor.
    buffer: Vec<T>,
    /// The shape of the tensor.
    shape: Shape,
}

/// Describes the shape of a tensor.
#[derive(Debug)]
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
    /// The strides for the allocated tensor.
    stride: Vec<usize>,
}

impl convert::From<[usize; 0]> for Shape {

    fn from(array: [usize; 0]) -> Shape {
        let capacity = 1;
        let rank = 0;
        let dims = Vec::with_capacity(1) /* TODO `with_capacity`? */;
        let stride = vec![];

        Shape { capacity, rank, dims, stride }
    }
}

impl convert::From<[usize; 1]> for Shape {

    fn from(array: [usize; 1]) -> Shape {
        let capacity = array[0];
        let rank = 1;
        let dims = array.to_vec();
        let stride = vec![1];

        Shape { capacity, rank, dims, stride }
    }
}

impl convert::From<[usize; 2]> for Shape {

    fn from(array: [usize; 2]) -> Shape {
        let capacity = array.iter().fold(1, |acc, &dims| acc * dims);
        let rank = 2;
        let dims = array.to_vec();
        let stride = compute_stride(&dims);

        Shape { capacity, rank, dims, stride }
    }
}

/// returns the default stride
fn compute_stride(dims: &Vec<usize>) -> Vec<usize> {
    let len = dims.len();
    let mut strides = vec![0; len];
    strides[len - 1] = 1;

    for i in 1..len {
        let index = len - i - 1;
        strides[index] = dims[index + 1] * strides[index + 1];
    }

    strides
}