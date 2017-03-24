use native::NativeMemory;
use opencl::OpenCLMemory;

/// Provides a representation for memory across different frameworks.
///
/// Memory is allocated by a device in a way that it is accessible for its computations.
///
/// Downcast methods are provided, but normally you will want to use a [`SharedTensor`] which 
/// handles synchronization of the latest memory copy to the required device.
///
/// [`SharedTensor`]: ./struct.SharedTensor.html
#[derive(Debug)]
pub enum Memory<T> {
    // /// A CUDA memory object.
    // Cuda(..),

    /// The native memory representation.
    Native(NativeMemory<T>),
    /// An OpenCL Memory.
    OpenCL(OpenCLMemory),
}

impl<T> Memory<T> {
    /// Returns a reference to the native memory representation.
    pub fn as_native(&self) -> Option<&NativeMemory<T>> {
        match *self {
            Memory::Native(ref native) => Some(native),
            _ => None
        }
    }

    /// Returns a mutable reference to the native memory representation.
    pub fn as_mut_native(&mut self) -> Option<&mut NativeMemory<T>> {
        match *self {
            Memory::Native(ref mut native) => Some(native),
            _ => None
        }
    }

    /// Returns a reference to the native memory representation.
    pub unsafe fn as_native_unchecked(&self) -> &NativeMemory<T> {
        match *self {
            Memory::Native(ref native) => native,
            _ => unreachable!()
        }
    }

    /// Returns a reference to the native memory representation.
    pub unsafe fn as_mut_native_unchecked(&mut self) -> &mut NativeMemory<T> {
        match *self {
            Memory::Native(ref mut native) => native,
            _ => unreachable!()
        }
    }

    /// Returns the native memory representation, consuming the convertee.
    pub fn into_native(self) -> Option<NativeMemory<T>> {
        match self {
            Memory::Native(native) => Some(native),
            _ => None
        }
    }

    /// Returns a reference to the OpenCL memory.
    pub fn as_opencl(&self) -> Option<&OpenCLMemory> {
        match *self {
            Memory::OpenCL(ref opencl) => Some(opencl),
            _ => None
        }
    }

    /// Returns a mutable reference to the OpenCL memory.
    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenCLMemory> {
        match *self {
            Memory::OpenCL(ref mut opencl) => Some(opencl),
            _ => None
        }
    }

    /// Returns a reference to the opencl memory representation.
    pub unsafe fn as_opencl_unchecked(&self) -> &OpenCLMemory {
        match *self {
            Memory::OpenCL(ref opencl) => opencl,
            _ => unreachable!()
        }
    }

    /// Returns the OpenCL memory, consuming the convertee.
    pub fn into_opencl(self) -> Option<OpenCLMemory> {
        match self {
            Memory::OpenCL(opencl) => Some(opencl),
            _ => None
        }
    }
}