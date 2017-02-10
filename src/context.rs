use std::any::{Any, TypeId};
use std::hash::Hash;

use error::{Error, ErrorKind, Result};
use super::{NativeContext, NativeMemory, Framework};
use super::shared::Location;

pub trait Context: 'static + Clone + Eq + Hash + PartialEq + Sized {
    /// The framework associated with this context.
    type F: Framework<Context = Self>;

    /// Constructs a context from a selection of devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - takes a list of devices.
    fn new(devices: Vec<<Self::F as Framework>::D>) 
        -> Result<Self, <Self::F as Framework>::E>;

    /// Allocates memory
    fn allocate_memory(&self, size: usize) 
        -> Result<<Self::F as Framework>::M, <Self::F as Framework>::E>;

    fn synch_in(
        self:           &Self, 
        memory:         &mut <Self::F as Framework>::M, 
        source_context: &NativeContext, 
        source_memory:  &NativeMemory) 
        -> Result;

    fn synch_out(
        self:           &Self, 
        memory:         &<Self::F as Framework>::M, 
        destn_context:  &NativeContext, 
        destn_memory:   &mut NativeMemory) 
        -> Result;
}

#[doc(hidden)]
pub enum Action<'s> {
    Write {
        memory: &'s mut Any, 
        source_context: &'s ContextView, 
        source_memory: &'s Any 
    },

    Read { 
        memory: &'s Any,  
        destn_context: &'s ContextView,  
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
pub trait ContextView: Any {

    fn alloc(&self, size: usize) -> Result<Box<Any>>;

    fn synch(&self, action: Action) -> Result;
}

#[doc(hidden)]
impl<V> ContextView for V where V: 'static + Context {

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
impl ContextView {

    /// Returns true if the boxed type is the same as `T`.
    pub fn is<V: ContextView>(&self) -> bool {
        let t = TypeId::of::<V>();

        let boxed = self.get_type_id();

        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub fn downcast_ref<V: ContextView>(&self) -> Option<&V> {
        if self.is::<V>() {
            unsafe {
                Some(&*(self as *const ContextView as *const V))
            }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub fn downcast_mut<V: ContextView>(&mut self) -> Option<&mut V> {
        if self.is::<V>() {
            unsafe {
                Some(&mut *(self as *mut ContextView as *mut V))
            }
        } else {
            None
        }
    }
}