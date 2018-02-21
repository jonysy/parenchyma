use ndarray::Array;

use super::NativeMemory;
use super::super::super::compute_device::{Allocate, ComputeDevice};
use super::super::super::error::Result;
use super::super::super::memory::Memory;
use super::super::super::tensor::TensorShape;

/// The native device.
#[derive(Debug)]
pub struct NativeDevice;

impl ComputeDevice for NativeDevice { }

impl<T: 'static> Allocate<T> for NativeDevice {
    fn allocate(&self, shape: &TensorShape) -> Result<Box<Memory<T>>> {
        let mut v = Vec::with_capacity(shape.capacity());

        unsafe {
            v.set_len(shape.capacity());
        }

        let array = Array::from_shape_vec(shape.dimensions(), v).unwrap();
        let memory = NativeMemory(array);

        return Ok(Box::new(memory));
    }
}