use super::native::NativeMemory;
use super::opencl::OpenCLMemory;

pub enum MemoryView {
    Native(NativeMemory),
    OpenCL(OpenCLMemory),
}

impl MemoryView {

    pub fn as_native(&self) -> Option<&NativeMemory> {
        match *self {
            MemoryView::Native(ref native) => Some(native),
            _ => None,
        }
    }
    
    pub fn as_mut_native(&mut self) -> Option<&mut NativeMemory> {
        match *self {
            MemoryView::Native(ref mut native) => Some(native),
            _ => None,
        }
    }

    pub fn as_opencl(&self) -> Option<&OpenCLMemory> {
        match *self {
            MemoryView::OpenCL(ref opencl) => Some(opencl),
            _ => None,
        }
    }

    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenCLMemory> {
        match *self {
            MemoryView::OpenCL(ref mut opencl) => Some(opencl),
            _ => None,
        }
    }
}