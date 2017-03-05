use super::{Buffer, Result};

/// The `T` type associated with the [`SharedTensor`](./struct.SharedTensor.html).
pub trait Device<T> {

    /// Allocates memory on a device.
    fn allocate(&self, size: usize, host: Option<&mut [T]>) -> Result<Buffer>;

    /// Synchronizes `memory` from `source`.
    fn synch_in(&self, buffer: &mut Buffer, source: &[T]) -> Result;

    /// Synchronizes `memory` to host.
    fn synch_out(&self, buffer: &Buffer, destination: &mut [T], size: usize) -> Result;
}