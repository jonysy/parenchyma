use super::{Buffer, Shape};
use super::error::Result;

/// The `T` type associated with the [`SharedTensor`](./struct.SharedTensor.html).
pub trait Device<T> {

    /// Allocates memory on the device.
    fn allocate(&self, shape: &Shape) -> Result<Buffer<T>>;

    /// Allocates memory on the device.
    fn allocate_with(&self, shape: &Shape, slice: &mut [T]) -> Result<Buffer<T>>;

    // /// Synchronizes `memory` from `source`.
    // fn sync_in(&self, memory: &mut Buffer<T>, source: &Buffer<T>) -> Result;

    // /// Synchronizes `memory` to `destination`.
    // fn sync_out(&self, memory: &Buffer<T>, destination: &mut Buffer<T>) -> Result;
}