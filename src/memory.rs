use frameworks::native::Memory as NativeMemory;
use frameworks::opencl::Memory as OpenCLMemory;

/// Represents a memory object.
///
/// note: downcast methods are provided.
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

    /// Returns the OpenCL memory, consuming the convertee.
    pub fn into_opencl(self) -> Option<OpenCLMemory> {
        match self {
            Memory::OpenCL(opencl) => Some(opencl),
            _ => None
        }
    }
}