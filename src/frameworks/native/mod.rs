//! Native backend support.

use ndarray::{Array, IxDyn};

/// Provides the native framework.
#[derive(Debug)]
pub struct Native;

/// Represents a native array.
///
/// note: named `Memory` for consistency across frameworks.
pub type Memory<T> = Array<T, IxDyn>;

/// The native context.
#[derive(Clone, Debug)]
pub struct NativeContext;

/// The native device.
#[derive(Clone, Debug)]
pub struct NativeDevice;