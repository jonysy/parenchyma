use std::fmt::Debug;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
///
/// note: `Clone` breaks object safety, so clone is implemented using `CloneableBoxedContext`.
///
/// TODO:
///
/// * `CloneableBoxedContext` may not be necessary.
pub trait Context: CloneableBoxedContext + Debug {

}

// =======

#[doc(hidden)]
pub trait CloneableBoxedContext {

    fn clone_box(&self) -> Box<Context>;
}

impl<T> CloneableBoxedContext for T where T: 'static + Context + Clone {

    fn clone_box(&self) -> Box<Context> { Box::new(self.clone()) }
}

impl Clone for Box<Context> {

    fn clone(&self) -> Box<Context> { self.clone_box() }
}