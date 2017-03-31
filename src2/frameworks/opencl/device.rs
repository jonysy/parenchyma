use error::Result;
use hardware::HardwareDevice;
use memory::{Alloc, BoxChunk, FlatBox};
use super::api::core;
use super::MemoryLock;

/// Represents an OpenCL device.
///
/// A compute device can contain multiple queues, can be multiple devices treated as a single 
/// compute device, along with other configurations. All objects (queue(s), device(s), 
/// buffer(s), etc.) must belong to the same context.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeDevice {
    /// The ID of the device.
    pub(super) device: core::Device,
    /// The context containing the `device` and `queue`.
    pub(super) context: core::Context,
    /// A command queue.
    ///
    /// A command queue is the mechanism for interaction with the device. The queue is used for 
    /// operations such as kernel launches and memory copies. At least one command queue per device
    /// is required. Queues are used by the host application to submit work to devices and 
    /// associated with devices within a context.
    ///
    /// __commands__:
    ///
    /// - memory copy or mapping
    /// - device code execution
    /// - synchronization point
    ///
    /// __modes__:
    ///
    /// - in-order
    /// - out-of-order
    ///
    /// ## TODO
    ///
    /// * Use events to synchronize
    pub(super) queue: core::CommandQueue,
}

impl HardwareDevice for ComputeDevice {
    // ..
}

impl Alloc<f32> for ComputeDevice {
    fn alloc_place(&self, data: Vec<f32>) -> Result<BoxChunk> {
        /// > `vec![x; n]`, `vec![a, b, c, d]`, and `Vec::with_capacity(n)`, will all produce a `Vec` 
        /// > with exactly the requested capacity. If `len()==capacity()`, (as is the case for 
        /// > the `vec!` macro), then a `Vec<T>` can be converted to and from a `Box<[T]>` without 
        /// > reallocating or moving the elements.

        let boxed = FlatBox::from(data.into_boxed_slice());
        let memory = MemoryLock::with(&self.context, self.queue.clone(), boxed)?;
        let location = self.clone();
        let chunk = Box::new((location, memory));

        Ok(chunk)
    }
}

impl Alloc<f64> for ComputeDevice { }