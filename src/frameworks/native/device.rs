use {Alloc, ComputeDevice, Memory, Result, Shape, Synch, Viewable};
use super::NativeMemory;

/// The native device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDevice;

impl Viewable for NativeDevice {

    fn view(&self) -> ComputeDevice {
        ComputeDevice::Native(NativeDevice)
    }
}

impl<T> Alloc<T> for NativeDevice {


    fn alloc(&self, shape: &Shape) -> Result<Memory<T>> {
        // TODO

        let mut buffer = Vec::with_capacity(shape.capacity());

        unsafe {
            buffer.set_len(shape.capacity());
        }

        Ok(Memory::Native(
            NativeMemory::from_shape_vec(shape.dimensions(), buffer).unwrap()))
    }

    fn allocwrite(&self, shape: &Shape, data: Vec<T>) -> Result<Memory<T>> {
        // TODO

        Ok(Memory::Native(
            NativeMemory::from_shape_vec(shape.dimensions(), data).unwrap()))
    }
}

impl<T> Synch<T> for NativeDevice where T: Clone {

    fn write(
        &self, 
        memory: &mut Memory<T>, 
        src_device: &ComputeDevice, 
        source: &Memory<T>) 
    -> Result {

        match *src_device {
            ComputeDevice::Native(_) => {
                let memory = unsafe { memory.as_mut_native_unchecked() };
                let source = unsafe { source.as_native_unchecked() };
                // > Array implements .clone_from() to reuse an array's existing allocation. 
                // > Semantically equivalent to *self = other.clone(), but potentially more efficient.
                Ok(memory.clone_from(source))
            },

            ComputeDevice::OpenCL(ref cl_device) => {
                cl_device.read(source, &mut ComputeDevice::Native(NativeDevice), memory)
            }
        }
    }

    fn read(
        &self, 
        memory: &Memory<T>, 
        dest_device: &mut ComputeDevice, 
        destination: &mut Memory<T>) 
    -> Result {

        match *dest_device {
            ComputeDevice::Native(_) => {
                let source = unsafe { memory.as_native_unchecked() };
                let destination = unsafe { destination.as_mut_native_unchecked() };
                // > Array implements .clone_from() to reuse an array's existing allocation. 
                // > Semantically equivalent to *self = other.clone(), but potentially more efficient.
                Ok(destination.clone_from(source))
            },

            ComputeDevice::OpenCL(ref mut cl_device) => {
                cl_device.write(destination, &ComputeDevice::Native(NativeDevice), memory)
            }
        }
    }
}