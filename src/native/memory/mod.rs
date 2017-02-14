#[cfg(not(feature = "unstable_alloc"))]
pub use self::stable_alloc::allocate_boxed_slice;
#[cfg(feature = "unstable_alloc")]
pub use self::unstable_alloc::allocate_boxed_slice;

#[cfg(not(feature = "unstable_alloc"))]
mod stable_alloc;
#[cfg(feature = "unstable_alloc")]
mod unstable_alloc;

use std::{fmt, mem, slice};

/// A `Box` with no knowledge of its underlying type ("flat box").
pub struct NativeMemory {
    len: usize,
    raw_box: *mut [u8]
}

impl NativeMemory {
    /// Access memory as a shared slice `&[T]`, where `T` represents the element type.
    pub fn as_slice<T>(&self) -> &[T] {

        unsafe {
            slice::from_raw_parts_mut(self.raw_box as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Access memory as a mutable slice `&mut [T]`, where `T` represents the element type.
    pub fn as_mut_slice<T>(&self) -> &mut [T] {

        unsafe {
            slice::from_raw_parts_mut(self.raw_box as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Returns the memory size of the `NativeMemory` ("flat box").
    pub fn len(&self) -> usize {
        self.len
    }
}

impl From<Box<[u8]>> for NativeMemory {

    /// Create a "flat box" from `Box`, consuming it.
    fn from(b: Box<[u8]>) -> NativeMemory {

        NativeMemory {
            len: b.len(),
            raw_box: Box::into_raw(b)
        }
    }
}

impl Drop for NativeMemory {

    fn drop(&mut self) {

        unsafe {
            // > After calling this function, the raw pointer is owned by the resulting `Box`. 
            // > Specifically, the `Box` destructor will call the destructor of `T` and free the 
            // > allocated memory. Since the way `Box` allocates and releases memory is unspecified, 
            // > the only valid pointer to pass to this function is the one taken from another `Box` 
            // > via the `Box::into_raw` function.
            let _ = Box::from_raw(self.raw_box);
        }
    }
}

impl fmt::Debug for NativeMemory {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        write!(f, "`NativeMemory` ('flat box') of length {}", self.len)
    }
}