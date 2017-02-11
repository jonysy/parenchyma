#[cfg(not(feature = "unstable_alloc"))]
pub use self::stable_alloc::allocate_boxed_slice;
#[cfg(feature = "unstable_alloc")]
pub use self::unstable_alloc::allocate_boxed_slice;

#[cfg(not(feature = "unstable_alloc"))]
mod stable_alloc;
#[cfg(feature = "unstable_alloc")]
mod unstable_alloc;

use std::{fmt, mem, slice};

/// A `Box` without any knowledge of its underlying type.
pub struct NativeMemory {
    /// The wrapped raw pointer
    raw: *mut [u8],
    len: usize
}

impl NativeMemory {
    pub fn allocate(size: usize) -> Self {
        let bx: Box<[u8]> = allocate_boxed_slice(size);
        NativeMemory::from(bx)
    }

    /// Access memory as a slice.
    pub fn as_slice<T>(&self) -> &[T] {

        unsafe {

            slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Access memory as a mutable slice.
    pub fn as_mut_slice<T>(&self) -> &mut [T] {

        unsafe {

            slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Returns memory size of the `FlatBox` (`NativeMemory`).
    pub fn size(&self) -> usize {
        self.len
    }
}

impl From<Box<[u8]>> for NativeMemory {

    fn from(b: Box<[u8]>) -> NativeMemory {

        NativeMemory { len: b.len(), raw: Box::into_raw(b) }
    }
}

impl Drop for NativeMemory {
    fn drop(&mut self) {

        unsafe {

            let _ = Box::from_raw(self.raw);
        }
    }
}

impl fmt::Debug for NativeMemory {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        write!(f, "FlatBox of length {}", self.len)
    }
}