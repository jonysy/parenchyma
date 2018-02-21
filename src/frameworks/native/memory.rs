use ndarray::{Array, IxDyn};
use std::ops::{Deref, DerefMut};

// use super::super::super::{Device, Memory, TransferDirection};
// use super::super::super::error::Result;

use super::NativeDevice;
use super::super::super::compute_device::ComputeDevice;
use super::super::super::memory::Memory;

/// A newtype (with an internal type of an n-dimensional array) representing a native memory buffer.
///
/// note: named `Memory` for consistency across frameworks.
pub struct NativeMemory<T>(pub(in crate) Array<T, IxDyn>);

impl<T: 'static> Memory<T> for NativeMemory<T> {
    fn synchronized(&self, compute_device: &ComputeDevice) -> bool {
        compute_device.is::<NativeDevice>()
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