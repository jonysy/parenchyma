use std::any::{Any, TypeId};
use std::hash::Hash;

use error::{Error, Result};
use super::Framework;

pub trait Context: Eq + Hash + PartialEq + Sized {
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

    // fn sync_in(
    //  &self, 
    //  my_memory: &mut <Self::Framework as Framework>::Memory, 
    //  src_context: &NativeContext, 
    //  src_memory: &NativeMemory
    // ) -> Result;

    // fn sync_out(
    //  &self, 
    //  my_memory: &<Self::Framework as Framework>::Memory, 
    //  dst_device: &NativeContext, 
    //  dst_memory: &mut NativeMemory
    // ) -> Result;
}

// ==============
// ==============

pub(super) enum Synch<'s> {
    In {
        memory: &'s mut Any, 
        source_context: &'s ContextView, 
        source_memory: &'s Any 
    },

    Out { 
        memory: &'s Any, 
        source_context: &'s ContextView, 
        source_memory: &'s mut Any 
    },
}

/// An object-safe version of `Context`.
///
/// Rules for object safety:
///
/// * [RFC 255](https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md)
/// * [RFC 428](https://github.com/rust-lang/rfcs/issues/428)
/// * [RFC 546](https://github.com/rust-lang/rfcs/blob/master/text/0546-Self-not-sized-by-default.md)
pub(super) trait ContextView: Any {

    fn alloc(&self, size: usize) -> Result<Box<Any>>;

    fn synchronize(&self, synch: Synch) -> Result;
}

impl<V> ContextView for V where V: 'static + Context {

    fn alloc(&self, size: usize) -> Result<Box<Any>> {

        let box_memory = |m| Box::new(m) as Box<Any>;
        Context::allocate_memory(self, size).map(box_memory).map_err(Error::from_framework::<V::F>)
    }

    fn synchronize(&self, synch: Synch) -> Result {

        // match synch {
        //  Synch::In { memory, source_context, source_memory } => {
        //      match memory.downcast_ref::<V>() {
        //          Some(memory) => {
        //              let cx = source_context.downcast_ref().unwrap();
        //              let m = source_memory.downcast_ref().unwrap();
        //              self.sync_in(memory, cx, m)
        //          },

        //          _ => {
        //              Err(ErrorKind::NoAvailableSynchronizationRouteFound.into())
        //          }
        //  },

        //  Synch::Out { memory, source_context, source_memory } => {
        //      self.sync_out(memory, source_context, source_memory)
        //  }
        // }

        unimplemented!()
    }
}

impl ContextView {

    /// Returns true if the boxed type is the same as `T`.
    pub(super) fn is<V: ContextView>(&self) -> bool {
        let t = TypeId::of::<V>();

        let boxed = self.get_type_id();

        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub(super) fn downcast_ref<V: ContextView>(&self) -> Option<&V> {
        if self.is::<V>() {
            unsafe {
                Some(&*(self as *const ContextView as *const V))
            }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the boxed value if it is of type `T`, or `None` if it isn't.
    pub(super) fn downcast_mut<V: ContextView>(&mut self) -> Option<&mut V> {
        if self.is::<V>() {
            unsafe {
                Some(&mut *(self as *mut ContextView as *mut V))
            }
        } else {
            None
        }
    }
}