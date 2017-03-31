use memory::Chunk;
use std::any::{Any, TypeId};
use utility::typedef::BoxChunk;

/// An device capable of processing data.
///
/// A compute device can be a single device, or multiple devices treated as a single device.
///
/// ## Load Balancing Multiple Devices
///
/// ..
pub trait ComputeDevice: Any {

    // /// `alloc` allocates memory on the device and then places `data` into it.
    // fn alloc(&self, data: Box<[u8]>) -> Result<BoxChunk>;

    // fn pin(&self, pointer: )
}

impl ComputeDevice {

    #[inline]
    pub fn is<I: ComputeDevice>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<I>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    pub fn downcast_ref<I: ComputeDevice>(&self) -> Option<&I> {
        if self.is::<I>() {
            unsafe {
                Some(&*(self as *const ComputeDevice as *const I))
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn downcast_mut<I: ComputeDevice>(&mut self) -> Option<&mut I> {
        if self.is::<I>() {
            unsafe {
                Some(&mut *(self as *mut ComputeDevice as *mut I))
            }
        } else {
            None
        }
    }
}