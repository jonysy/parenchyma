use std::marker::PhantomData;
use super::Backend;
use super::error::Result;

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
    /// A marker for `T`.
    phantom: PhantomData<T>,
}

// /// Synchronization direction.
// pub enum Synch { In, Out, Bidirectional }

impl<T> SharedTensor<T> /*where T: Scalar*/ {

    /// Constructs a new `SharedTensor`.
    pub fn new(backend: &Backend) -> Result<Self> {

        unimplemented!()
    }

    /// Constructs a new `SharedTensor` with the data from the supplied `slice`.
    pub fn new(backend: &Backend, slice: &mut [T]) -> Result<Self> {

        unimplemented!()
    }
}

/// A native tensor.
#[derive(Debug)]
pub struct Tensor<T> {
    // /// The internal representation of the tensor.
    // buffer: NativeBuffer,
}