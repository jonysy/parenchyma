use std::fmt::Debug;
use super::Framework;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
pub trait Context: 'static + Clone + Debug {

    /// The framework associated with the context.
    type F: Framework<Context = Self>;

    /// Constructs a context from a selection of devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - a list of devices.
    fn new(devices: Vec<<Self::F as Framework>::Device>) 
        -> Result<Self, <Self::F as Framework>::E>;

    /// Returns the devices encapsulated by the context.
    fn devices(&self) -> &[<Self::F as Framework>::Device];
}