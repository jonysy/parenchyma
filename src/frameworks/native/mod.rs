//! Native backend support.

use std::{cmp, mem, slice};

/// Provides the native framework.
#[derive(Debug)]
pub struct Native;

/// The native context.
#[derive(Clone, Debug)]
pub struct NativeContext;

/// The native device.
#[derive(Clone, Debug)]
pub struct NativeDevice;

impl cmp::PartialEq for NativeDevice {

    fn eq(&self, _: &Self) -> bool { true }
}

impl cmp::Eq for NativeDevice { }

/// A `Box` with no knowledge of its underlying type ("flat box").
#[derive(Debug)]
pub struct NativeFlatBox {
    raw: *mut [u8],
    len: usize,
}

impl NativeFlatBox {

    /// Traditional allocation through the creation of a filled `Vec<u8>` of length `capacity`.
    #[cfg(not(feature = "unstable_alloc"))]
    pub fn alloc(capacity: usize) -> NativeFlatBox {
        let vec: Vec<u8> = vec![0; capacity];
        let b: Box<[u8]> = vec.into_boxed_slice();
        let len = b.len();
        let raw = Box::into_raw(b);

        NativeFlatBox { raw, len }
    }

    /// An alternative way to allocate memory, but requires [`RawVec`][RawVec] which is currently
    /// unstable (*[#27783]).
    ///
    /// [#27783]: https://github.com/rust-lang/rust/issues/27783
    /// [RawVec]: https://doc.rust-lang.org/alloc/raw_vec/struct.RawVec.html
    #[cfg(feature = "unstable_alloc")]
    pub fn alloc(capacity: usize) -> NativeFlatBox {
        use alloc::raw_vec::RawVec;

        unsafe {
            let raw_vec = RawVec::with_capacity(capacity);
            let b: Box<[u8]> = raw_vec.into_box();
            let len = b.len();
            let raw = Box::into_raw(b);

            NativeFlatBox { raw, len }
        }
    }

    /// Access memory as a shared slice `&[T]`, where `T` represents the element type.
    pub fn as_slice<T>(&self) -> &[T] {

        unsafe {
            slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Access memory as a mutable slice `&mut [T]`, where `T` represents the element type.
    pub fn as_mut_slice<T>(&self) -> &mut [T] {

        unsafe {
            slice::from_raw_parts_mut(self.raw as *mut T, self.len / mem::size_of::<T>())
        }
    }

    /// Returns the memory size of the `NativeMemory` ("flat box").
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for NativeFlatBox {

    fn drop(&mut self) {

        unsafe {
            // > After calling this function, the raw pointer is owned by the resulting `Box`. 
            // > Specifically, the `Box` destructor will call the destructor of `T` and free the 
            // > allocated memory. Since the way `Box` allocates and releases memory is unspecified, 
            // > the only valid pointer to pass to this function is the one taken from another `Box` 
            // > via the `Box::into_raw` function.
            let _ = Box::from_raw(self.raw);
        }
    }
}