use error::{Error, ErrorKind, Result};
use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt::Debug;
use super::{NativeDevice, NativeMemory, Framework};

/// A device
pub trait Device: 'static + Clone + Debug + Eq + PartialEq + Sized {
    /// The framework associated with the Device.
    type F: Framework<Device = Self>;

    /// Allocates memory on a device.
    fn allocate_memory(&self, size: usize) 
        -> Result<<Self::F as Framework>::M, <Self::F as Framework>::E>;

    /// Synchronizes `memory` from `source`.
    fn synch_in(
        self:           &Self, 
        memory:         &mut <Self::F as Framework>::M, 
        source_device: &NativeDevice, 
        source_memory:  &NativeMemory) 
        -> Result<(), <Self::F as Framework>::E>;

    /// Synchronizes `memory` to `destination`.
    fn synch_out(
        self:           &Self, 
        memory:         &<Self::F as Framework>::M, 
        destn_device:  &NativeDevice, 
        destn_memory:   &mut NativeMemory) 
        -> Result<(), <Self::F as Framework>::E>;
}

/// General categories for devices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceKind {
    /// Accelerators
    Accelerator,
    /// CPU devices (host processors)
    Cpu,
    /// GPU devices
    Gpu,
    /// Used for anything else
    Other(Cow<'static, str>),
}

// ===============

#[doc(hidden)]
pub enum Action<'s> {
    Write {
        memory: &'s mut Any, 
        source_device: &'s ObjSafeDev, 
        source_memory: &'s Any 
    },

    Read { 
        memory: &'s Any,  
        destn_device: &'s ObjSafeDev,  
        destn_memory: &'s mut Any 
    },
}

/// An object-safe version of `Device`.
///
/// Rules for object safety:
///
/// * [RFC 255](https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md)
/// * [RFC 428](https://github.com/rust-lang/rfcs/issues/428)
/// * [RFC 546](https://github.com/rust-lang/rfcs/blob/master/text/0546-Self-not-sized-by-default.md)
#[doc(hidden)]
pub trait ObjSafeDev: Any + Debug {

    fn alloc(&self, size: usize) -> Result<Box<Any>>;

    fn synch(&self, action: Action) -> Result;
}

#[doc(hidden)]
impl<V> ObjSafeDev for V where V: 'static + Device {

    fn alloc(&self, size: usize) -> Result<Box<Any>> {

        let box_memory = |m| Box::new(m) as Box<Any>;
        Device::allocate_memory(self, size).map(box_memory).map_err(Error::from_framework::<V::F>)
    }

    fn synch(&self, action: Action) -> Result {

        match action {
            Action::Write { memory: m, source_device: sctx, source_memory: sm } => {

                match (m.downcast_mut(), sctx.downcast_ref(), sm.downcast_ref()) {
                    (Some(memory), Some(source_device), Some(source_memory)) => {

                        self.synch_in(memory, source_device, source_memory)
                            .map_err(Error::from_framework::<V::F>)
                    },

                    _ => {
                        Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
                    }
                }
            },

            Action::Read { memory: m, destn_device: dctx, destn_memory: dm } => {
                match (m.downcast_ref(), dctx.downcast_ref(), dm.downcast_mut()) {
                    (Some(memory), Some(destn_device), Some(destn_memory)) => {

                        self.synch_out(memory, destn_device, destn_memory)
                            .map_err(Error::from_framework::<V::F>)
                    },

                    _ => {
                        Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
                    }
                }
            }
        }
    }
}

#[doc(hidden)]
impl ObjSafeDev {

    /// Returns true if the boxed type is the same as `T`.
    pub fn is<V: ObjSafeDev>(&self) -> bool {
        let t = TypeId::of::<V>();

        let boxed = self.get_type_id();

        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub fn downcast_ref<V: ObjSafeDev>(&self) -> Option<&V> {
        if self.is::<V>() {
            unsafe {
                Some(&*(self as *const ObjSafeDev as *const V))
            }
        } else {
            None
        }
    }
}