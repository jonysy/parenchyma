//! Provides a unified representation of memory across different frameworks.

use std::any::{Any, TypeId};
use super::compute_device::ComputeDevice;
use super::error::{ErrorKind, Result};

// TODO
// pub struct Stacked<'p, T> { data: T, marker: PhantomData<&'p mut &'a ()> }
// pub struct Boxed<T> { data: Box<T> }

/// The transfer direction
pub enum TransferDirection {
    /// Transfer data 
    TransferIn,
    /// Transfer data out (i.e., _dumps_ data)
    TransferOut,
}

/// The struct `Memory` has generic type parameters representing memory and its location as seen 
/// by the shared tensor.
///
/// **notes**:
///
/// * The words _buf_ and _memory_ are used here (until a better term comes to 
/// mind (candidates: _allocation_, _partition_, etc.)) for the sake of simplification.
/// * Each framework handles memory allocation differently (e.g., OpenCL allocates memory _lazily_ 
/// and isn't associated with any device within the context, even after it's used). 
/// * Downcast methods are provided, but normally you will want to use a [`SharedTensor`] which 
/// handles synchronization of the latest memory copy to the required device.
pub trait Memory<T>: Any {
    /// Specifies synchronization behavior for keeping data consistent across frameworks and contexts.
    ///
    /// **note**
    ///
    /// _Synch_ shouldn't be confused with the marker type `Sync` found in the standard library. 
    /// The less common abbreviation for _synchronize_ (the extra _h_) is used here to 
    /// avoid confusion.
    ///
    /// The `transfer` method handles the asynchronous data transfer behavior across 
    /// frameworks and contexts.
    ///
    // # TODO: Transfer Matrix/Routes
    //
    // Host-GPU: Host <-> GPU
    // GPU-GPU: GPU -> HOST -> GPU
    //
    // ```{.text}
    // opencl device (context `a`) -> opencl device (context `b`) = true
    // opencl device -> native/host = true
    // opencl <-> cuda = false
    // native/host -> native/host = true
    // native/host -> cuda/opencl = false
    // ```
    fn transfer(&mut self, TransferDirection, &mut Memory<T>) -> Result {
        Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
    }
    /// Determines whether or not the memory is allocated or pinned on the `backend`'s active device.
    ///
    /// # Arguments
    ///
    /// * `compute_device` - The computation device.
    ///
    /// **note**:
    ///
    /// Certain frameworks have a concept of _shared memory_, where the location of the 
    /// memory is omnipresent (in a very abstract sense) as long as the devices are within the same
    /// context.
    #[allow(unused_variables)]
    fn synchronized(&self, compute_device: &ComputeDevice) -> bool {
        return false;
    }
}

impl<T: 'static> Memory<T> {
    /// Returns `true` if the boxed type is the same as `T`.
    #[inline]
    pub fn is<M: Memory<T>>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<M>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[inline]
    pub fn downcast_ref<M: Memory<T>>(&self) -> Option<&M> {
        if self.is::<M>() {
            unsafe {
                Some(&*(self as *const Memory<T> as *const M))
            }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[inline]
    pub fn downcast_mut<M: Memory<T>>(&mut self) -> Option<&mut M> {
        if self.is::<M>() {
            unsafe {
                Some(&mut *(self as *mut Memory<T> as *mut M))
            }
        } else {
            None
        }
    }
}