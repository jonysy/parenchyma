use hardware::HardwareDevice;
use memory::{Chunk, FlatBox, Memory, Synch};
use std::ops::{Deref, DerefMut};
use super::NativeDevice;

/// A newtype representing a native memory buffer.
///
/// note: named `NativeMemory` for consistency across frameworks.
#[derive(Debug)]
pub struct NativeMemory(pub(super) FlatBox);

impl Deref for NativeMemory {

    type Target = FlatBox;

    fn deref(&self) -> &FlatBox {
        &self.0
    }
}

impl DerefMut for NativeMemory {

    fn deref_mut(&mut self) -> &mut FlatBox {
        &mut self.0
    }
}

pub type NativeChunk = (NativeDevice, NativeMemory);

impl Chunk for NativeChunk {

    fn this(&self) -> (&HardwareDevice, &Memory) {
        let &(ref h, ref m) = self;
        (h, m)
    }

    fn located_on(&self, other: &HardwareDevice) -> bool {
        other.is::<NativeDevice>()
    }
}

impl Memory for NativeMemory {

    // fn bytes<'a>(&'a self) -> &'a [u8] {
    //     unsafe {
    //         &*self.0.pointer
    //     }
    // }
}

impl<T> Synch<T> for NativeChunk { }