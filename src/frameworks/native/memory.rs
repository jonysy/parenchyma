use ndarray::{Array, IxDyn};

/// Represents a native array.
///
/// note: named `Memory` for consistency across frameworks.
pub type NativeMemory<T> = Array<T, IxDyn>;