use ndarray::{Array, IxDyn};
use std::fmt;
use std::ops::{Deref, DerefMut};

/// A newtype (with an internal type of an n-dimensional array) representing a native memory buffer.
///
/// note: named `Memory` for consistency across frameworks.
pub struct NativeMemory<T>(Array<T, IxDyn>);

impl<T> NativeMemory<T> {

    /// Constructs a `NativeMemory` from the provided `array`.
    pub fn new(array: Array<T, IxDyn>) -> NativeMemory<T> {
        NativeMemory(array)
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a flattened, linear representation of the tensor.
    ///
    /// **caution**: this method uses `unwrap`.
    pub fn as_flat(&self) -> &[T] {
        self.0.as_slice_memory_order().expect("the array's data is not contiguous")
    }

    /// Returns a mutable flattened, linear representation of the tensor.
    ///
    /// **caution**: this method uses `unwrap`.
    pub fn as_mut_flat(&mut self) -> &mut [T] {
        self.0.as_slice_memory_order_mut().expect("the array's data is not contiguous")
    }
}

impl<T> Clone for NativeMemory<T> where Array<T, IxDyn>: Clone {

    fn clone(&self) -> NativeMemory<T> {
        NativeMemory(self.0.clone())
    }

    fn clone_from(&mut self, other: &Self) {
        self.0.clone_from(&other.0)
    }
}

impl<T> Deref for NativeMemory<T> {

    type Target = Array<T, IxDyn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for NativeMemory<T> {

    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> fmt::Debug for NativeMemory<T> where T: fmt::Debug {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}