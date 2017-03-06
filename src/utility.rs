use ndarray;
use std::mem;

/// Represents a native array.
pub type Array<T> = ndarray::Array<T, ndarray::IxDyn>;

/// Returns the size of the allocated memory in bytes.
pub fn allocated<T>(length: usize) -> usize {
    length * mem::size_of::<T>()
}