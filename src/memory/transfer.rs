use super::Chunk;

use device::ComputeDevice;
use error::Error;
use futures::future;
use utility::typedef::{BoxFuture, Result};

/// Specifies asynchronous data transfer behavior across frameworks and contexts.
pub trait AsyncTransfer {
    /// Transfer memory from the source `chunk`.
    fn transfer_in(&mut self, chunk: &mut Chunk) -> Result;

    /// Transfer memory to the destination `chunk`.
    fn transfer_out(&mut self, chunk: &mut Chunk) -> Result;

    /// Determines whether or not there's a transfer route available (i.e., if this chunk 
    /// is _transferable_ to/from the specified device (i.e., _syncable_)).
    ///
    /// ## Common Transfers
    ///
    /// ```{.text}
    /// opencl device (context `a`) <-> opencl device (context `b`) = true (via an indirect tx)
    /// opencl/cuda <-> native/host = true
    /// native/host <-> cuda/opencl = false
    /// opencl <-> cuda = false
    /// native/host -> native/host = false (unreachable)
    /// native/host -> cuda/opencl = false
    /// ```
    ///
    /// **note**:
    ///
    /// Due to the decoupled structure of Parenchyma, the host is not aware of any of the 
    /// frameworks. Therefore, it is up to the framework implementations to determine how best to 
    /// transfer data to/from the host.
    ///
    /// ## Related Issues
    ///
    /// A proper transfer matrix is currently being explored ([issue#23]).
    ///
    /// [issue#23]: https://github.com/lychee-eng/parenchyma/issues/23
    fn transferable(&self, device: &ComputeDevice) -> bool;
    
    /// Connects to the memory for host access.
    ///
    /// This method is called:
    ///
    /// * before the transfer_([in](#method.transfer_in)/[out](#method.transfer_out)) methods
    /// * when attempting to _view_ the pinned data (a request is made)
    ///
    /// When connected, Parenchyma makes a couple assumptions:
    ///
    /// a) The host can access the memory in the _future_
    /// b) The host, and only the host, has access to valid memory
    ///
    /// This is commonly referred to as _mapping_ and is usually only used for mapped/pinned 
    /// buffers, so, by default, the connection is available immediately. This method is 
    /// overridable and should be overridden by each framework to match its map/pin logic. 
    fn connect<'a>(&'a self) -> BoxFuture<'a> {
        box future::ok::<_, Error>(())
    }

    /// Disconnects from the allocated memory.
    ///
    /// This method is called:
    ///
    /// * after memory has finished transferring
    /// * after _viewing_
    ///
    /// When disconnected, Parenchyma makes a couple assumptions:
    ///
    /// a) The host no longer has access to the memory
    /// b) The access will be reverted back to the device in the _future_
    ///
    /// This is commonly referred to as _unmapping_ and is usually only used for mapped/pinned 
    /// buffers, so, by default, it is disconnected immediately. Therefore, this method should be 
    /// overridden by each framework to match its map/pin logic. 
    fn disconnect<'a>(&'a self) -> BoxFuture<'a> {
        box future::ok::<_, Error>(())
    }
}