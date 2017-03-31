use super::{Memory, Synch};

use device::ComputeDevice;
use std::any::{Any, TypeId};

/// An _allocation_ or a segment of allocated memory on a device.
///
/// **notes**:
///
/// * The word _chunk_ is used here (until a better term comes to mind (candidates: _allocation_, 
/// _partition_, etc.)) for the sake of simplification.
/// * Each framework handles memory allocation differently (e.g., OpenCL allocates memory _lazily_ 
/// and isn't associated with any device within the context, even after it's used). 
/// * Downcast methods are provided, but normally you will want to use a [`SharedTensor`] which 
/// handles synchronization of the latest memory copy to the required device.
///
/// [`SharedTensor`]: ./tensor/struct.SharedTensor.html
pub trait Chunk: Any {
    /// Returns the memory object and its location.
    fn this(&self) -> (&ComputeDevice, &Memory);


    /// Determines whether or not the `chunk` is allocated or pinned on the device.
    ///
    /// **note**:
    ///
    /// Certain frameworks have a concept of _shared memory_, where the location of the 
    /// memory is omnipresent (in a very abstract sense) as long as the devices are within the same
    /// context.
    fn located_on(&self, &ComputeDevice) -> bool;

    // /// Returns `true` if the memory is pinned.
    // fn pinned(&self) -> bool;
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