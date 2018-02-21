//! Provides a representation for one or many ready to use compute devices.

use std::any::{Any, TypeId};

use super::error::Result;
use super::memory::Memory;
use super::tensor::TensorShape;

/// An device capable of processing data.
///
/// A compute device can be a single device, or multiple devices treated as a single device.
///
/// ## Load Balancing Multiple Devices
///
/// todo..
pub trait ComputeDevice: Any + Allocate<f64> + Allocate<f32> { }

/// Implemented by allocators.
pub trait Allocate<T> {
    /// Allocates memory on the device.
    fn allocate(&self, shape: &TensorShape) -> Result<Box<Memory<T>>>;
}

impl ComputeDevice {
    /// Returns `true` if the boxed type is the same as `T`.
    #[inline]
    pub fn is<T>(&self) -> bool where T: ComputeDevice {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();
        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();
        // Compare both TypeIds on equality
        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[inline]
    pub fn downcast_ref<T>(&self) -> Option<&T> where T: ComputeDevice {
        if self.is::<T>() {
            unsafe {
                Some(&*(self as *const ComputeDevice as *const T))
            }
        } else {
            None
        }
    }
}