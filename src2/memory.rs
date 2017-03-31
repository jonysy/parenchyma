//! Provides a representation for memory across different frameworks.

use alloc::raw_vec::RawVec;
use error::{Error, ErrorKind, Result};
use futures::{Async, Future};
use hardware::HardwareDevice;
use std::any::{Any, TypeId};
use std::fmt;
use std::{mem, slice};

/// The success type returned by allocator methods.
pub type BoxChunk = Box<Chunk>;

pub type BoxFuture<'a, T = ()> = Box<(Future<Item=T, Error=Error> + 'a)>;

/// An _allocation_ or a segment of allocated memory on a device.
///
/// **notes**:
///
/// * The word _chunk_ is used here (until a better term comes to mind (candidates: _allocation_, 
/// _partition_, etc.)) for the sake of simplification.
/// * Each framework handles memory allocation differently (e.g., OpenCL allocates memory _lazily_ 
/// and isn't associated with any device until it's used). 
/// * Downcast methods are provided, but normally you will want to use a [`SharedTensor`] which 
/// handles synchronization of the latest memory copy/partition/allocation to the required device.
///
/// [`SharedTensor`]: ./struct.SharedTensor.html
#[fundamental]
pub trait Chunk: Any + Synch<f32> {
    /// Returns the memory object and its location.
    fn this(&self) -> (&HardwareDevice, &Memory);

    fn located_on(&self, other: &HardwareDevice) -> bool {
        false
    }
}

impl Chunk {

    #[inline]
    pub fn is<I: Chunk>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<I>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    pub fn downcast_ref<I: Chunk>(&self) -> Option<&I> {
        if self.is::<I>() {
            unsafe {
                Some(&*(self as *const Chunk as *const I))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_mut<I: Chunk>(&mut self) -> Option<&mut I> {
        if self.is::<I>() {
            unsafe {
                Some(&mut *(self as *mut Chunk as *mut I))
            }
        } else {
            None
        }
    }
}

pub trait Memory: Any + fmt::Debug {

    // fn bytes<'a>(&'a self) -> &'a [u8] {

    //     unimplemented!()
    // }

    //fn async_view<'a>(&'a self) -> Box<Future<Item = &'a [T], Error = Error>>;

    // fn view<'a>(&self) -> &'a [T] {
    //     unimplemented!()
    // }

    // fn view_mut<'a>(&mut self) -> &'a mut [T] {
    //     unimplemented!()
    // }
}

impl Memory {

    #[inline]
    pub fn is<I: Memory>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<I>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    pub fn downcast_ref<I: Memory>(&self) -> Option<&I> {
        if self.is::<I>() {
            unsafe {
                Some(&*(self as *const Memory as *const I))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_mut<I: Memory>(&mut self) -> Option<&mut I> {
        if self.is::<I>() {
            unsafe {
                Some(&mut *(self as *mut Memory as *mut I))
            }
        } else {
            None
        }
    }
}

/// Provides synchronization methods.
///
/// _Synch_ shouldn't be confused with the marker type `Sync` found in the standard library. 
/// The less common abbreviation for _synchronize_ (the extra _h_) is used here to avoid confusion.
pub trait Synch<T> {

    /// Use the _syncable_ method to determine if there's a transfer route available or if 
    /// it's _transferable_ to the specified `source`.
    ///
    /// ## Example
    ///
    /// ```{.text}
    /// opencl device (context `a`) -> opencl device (context `b`) = true
    /// opencl device -> native/host = true
    /// opencl <-> cuda = false
    /// native/host -> native/host = true
    /// native/host -> cuda/opencl = false
    /// ```
    fn syncable(&self, &HardwareDevice) -> bool {
        false
    }

    /// Synchronizes memory from `source`.
    fn synchronize_in<'a>(&'a mut self, source: &'a mut Chunk) -> BoxFuture {
        Box::new(::futures::future::err(
            ErrorKind::NoAvailableSynchronizationRouteFound.into()))
    }

    /// Synchronizes memory to `destination`.
    fn synchronize_out<'a>(&'a mut self, destination: &'a mut Chunk) -> BoxFuture {
        Box::new(::futures::future::err(
            ErrorKind::NoAvailableSynchronizationRouteFound.into()))
    }
}

/// Allocator
pub trait Alloc<T> {

    /// Allocates memory on the device and then places `data` into it.
    fn alloc_place(&self, data: Vec<T>) -> Result<BoxChunk> { unimplemented!() }
}

/// A `Box` without any knowledge of its underlying type.
pub struct FlatBox {
    /// The size of the allocated memory in bytes.
    pub byte_size: usize,
    /// The raw pointer.
    pub pointer: *mut [u8],
}

impl FlatBox {

    /// Allocates memory
    pub fn new(capacity: usize) -> FlatBox {
        let raw = RawVec::with_capacity(capacity);
        let boxed_slice: Box<[u8]> = unsafe { raw.into_box() };
        FlatBox::from(boxed_slice)
    }

    /// Access memory as slice - the preferred way to access native memory.
    pub fn as_slice<T>(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts_mut(
                self.pointer as *mut T, self.byte_size / mem::size_of::<T>())
        }
    }

    /// Access memory as mutable slice - the preferred way to access native memory.
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(
                self.pointer as *mut T, self.byte_size / mem::size_of::<T>())
        }
    }

    /// Returns memory size of the `FlatBox`.
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl Drop for FlatBox {

    fn drop(&mut self) {
        // > Constructs a box from a raw pointer.
        // >
        // > After calling this function, the raw pointer is owned by the resulting `Box`. 
        // > Specifically, the `Box` destructor will call the destructor of `T` and free the 
        // > allocated memory. Since the way `Box` allocates and releases memory is 
        // > unspecified, the only valid pointer to pass to this function is the one taken from 
        // > another `Box` via the `Box::into_raw` function.
        // >
        // > This function is unsafe because improper use may lead to memory problems. For 
        // > example, a double-free may occur if the function is called twice on the same raw 
        // > pointer.
        let _ = unsafe { Box::from_raw(self.pointer) };
    }
}

impl<T> From<Box<[T]>> for FlatBox {

    fn from(bytes: Box<[T]>) -> FlatBox {
        let byte_size = bytes.len() * mem::size_of::<T>();
        // > Consumes the Box, returning the wrapped raw pointer.
        // >
        // > After calling this function, the caller is responsible for the memory previously 
        // > managed by the `Box`. In particular, the caller should properly destroy `T` and 
        // > release the memory. The proper way to do so is to convert the raw pointer back into 
        // > a `Box` with the `Box::from_raw` function.
        // >
        // > Note: this is an associated function, which means that you have to call it 
        // > as `Box::into_raw(b)` instead of `b.into_raw()`. This is so that there is no conflict 
        // > with a method on the inner type.
        let raw: *mut [T] = Box::into_raw(bytes);
        let pointer: *mut [u8] = raw as *mut [u8];

        FlatBox { byte_size, pointer }
    }
}

impl fmt::Debug for FlatBox {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "`FlatBox` containing {} bytes", self.byte_size)
    }
}