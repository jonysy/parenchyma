use frameworks::opencl::OpenClBuffer;
use super::utility::Array;

/// Represents a buffer object.
///
/// note: downcast methods are provided.
#[derive(Debug)]
pub enum Buffer<T> {
    // /// A CUDA memory object.
    // Cuda(..),

    /// The native memory representation.
    Native(Array<T>),

    /// An OpenCL buffer.
    OpenCl(OpenClBuffer),
}

impl<T> Buffer<T> {
    /// Returns a reference to the native memory representation.
    pub fn as_native(&self) -> Option<&Array<T>> {
        match *self {
            Buffer::Native(ref native) => Some(native),
            _ => None
        }
    }

    /// Returns a mutable reference to the native memory representation.
    pub fn as_mut_native(&mut self) -> Option<&mut Array<T>> {
        match *self {
            Buffer::Native(ref mut native) => Some(native),
            _ => None
        }
    }

    /// Returns the native memory representation, consuming the convertee.
    pub fn into_native(self) -> Option<Array<T>> {
        match self {
            Buffer::Native(native) => Some(native),
            _ => None
        }
    }

    /// Returns a reference to the OpenCL buffer.
    pub fn as_opencl(&self) -> Option<&OpenClBuffer> {
        match *self {
            Buffer::OpenCl(ref opencl) => Some(opencl),
            _ => None
        }
    }

    /// Returns a mutable reference to the OpenCL buffer.
    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenClBuffer> {
        match *self {
            Buffer::OpenCl(ref mut opencl) => Some(opencl),
            _ => None
        }
    }

    /// Returns the OpenCL buffer, consuming the convertee.
    pub fn into_opencl(self) -> Option<OpenClBuffer> {
        match self {
            Buffer::OpenCl(opencl) => Some(opencl),
            _ => None
        }
    }
}