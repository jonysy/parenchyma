use super::Buffer;
use super::error::Result;

/// The `T` type associated with the [`SharedTensor`](./struct.SharedTensor.html).
pub trait Device<T> {

    /// Allocates memory on a device.
    fn allocate(&self, size: usize, host: Option<&mut [T]>) -> Result<Buffer>;

    /// Synchronizes `memory` from `source`.
    fn sync_in(&self, memory: &mut Buffer, source: &[T]) -> Result;

    /// Synchronizes `memory` to `destination`.
    fn sync_out(&self, memory: &Buffer, destination: &mut [T]) -> Result;
}