use error::{Error, ErrorKind, Result};
use std::any::{Any, TypeId};
use std::fmt::Debug;
use super::{NativeContext, NativeMemory, Framework};

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
pub trait Context: 'static + Clone + Debug + Eq + PartialEq + Sized {

    /// The framework associated with the context.
    type F: Framework<Context = Self>;

    /// Constructs a context from a selection of devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - a list of devices.
    fn new(devices: <Self::F as Framework>::D) 
        -> Result<Self, <Self::F as Framework>::E>;

    /// Allocates memory on a device.
    fn allocate_memory(&self, size: usize) 
        -> Result<<Self::F as Framework>::M, <Self::F as Framework>::E>;

    /// Synchronizes `memory` from `source`.
    fn synch_in(
        self:           &Self, 
        memory:         &mut <Self::F as Framework>::M, 
        source_context: &NativeContext, 
        source_memory:  &NativeMemory) 
        -> Result<(), <Self::F as Framework>::E>;

    /// Synchronizes `memory` to `destination`.
    fn synch_out(
        self:           &Self, 
        memory:         &<Self::F as Framework>::M, 
        destn_context:  &NativeContext, 
        destn_memory:   &mut NativeMemory) 
        -> Result<(), <Self::F as Framework>::E>;
}

#[doc(hidden)]
pub enum Action<'s> {
    Write {
        memory: &'s mut Any, 
        source_context: &'s ObjSafeCtx, 
        source_memory: &'s Any 
    },

    Read { 
        memory: &'s Any,  
        destn_context: &'s ObjSafeCtx,  
        destn_memory: &'s mut Any 
    },
}

/// An object-safe version of `Context`.
///
/// Rules for object safety:
///
/// * [RFC 255](https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md)
/// * [RFC 428](https://github.com/rust-lang/rfcs/issues/428)
/// * [RFC 546](https://github.com/rust-lang/rfcs/blob/master/text/0546-Self-not-sized-by-default.md)
#[doc(hidden)]
pub trait ObjSafeCtx: Any + Debug {

    fn alloc(&self, size: usize) -> Result<Box<Any>>;

    fn synch(&self, action: Action) -> Result;
}

#[doc(hidden)]
impl<V> ObjSafeCtx for V where V: 'static + Context {

    fn alloc(&self, size: usize) -> Result<Box<Any>> {

        let box_memory = |m| Box::new(m) as Box<Any>;
        Context::allocate_memory(self, size).map(box_memory).map_err(Error::from_framework::<V::F>)
    }

    fn synch(&self, action: Action) -> Result {

        match action {
            Action::Write { memory: m, source_context: sctx, source_memory: sm } => {

                match (m.downcast_mut(), sctx.downcast_ref(), sm.downcast_ref()) {
                    (Some(memory), Some(source_context), Some(source_memory)) => {

                        self.synch_in(memory, source_context, source_memory)
                            .map_err(Error::from_framework::<V::F>)
                    },

                    _ => {
                        Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
                    }
                }
            },

            Action::Read { memory: m, destn_context: dctx, destn_memory: dm } => {
                match (m.downcast_ref(), dctx.downcast_ref(), dm.downcast_mut()) {
                    (Some(memory), Some(destn_context), Some(destn_memory)) => {

                        self.synch_out(memory, destn_context, destn_memory)
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
impl ObjSafeCtx {

    /// Returns true if the boxed type is the same as `T`.
    pub fn is<V: ObjSafeCtx>(&self) -> bool {
        let t = TypeId::of::<V>();

        let boxed = self.get_type_id();

        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub fn downcast_ref<V: ObjSafeCtx>(&self) -> Option<&V> {
        if self.is::<V>() {
            unsafe {
                Some(&*(self as *const ObjSafeCtx as *const V))
            }
        } else {
            None
        }
    }
}