use error::{Error, ErrorKind, Result};
use framework::Framework;
use hardware::{Hardware, HardwareDevice};
use std::fmt::Debug;

/// Contexts are the heart of both OpenCL and CUDA applications. Contexts provide a container for
/// objects such as memory, command-queues, programs/modules and kernels.
///
/// You can create a context encapsulating a selection of hardware via a [`Backend`].
///
/// [`Backend`]: ./struct.Backend.html
pub trait Context: Debug {
    /// Returns all _activatable_ hardware provided to the context.
    fn selection(&self) -> &[Hardware];

    /// Returns the _active_ device.
    fn active(&self) -> &(HardwareDevice + 'static);

    /// Set the device at the specified `index` as the active device.
    ///
    /// Only one device can be the _active_ device - the device in which operations are executed -
    /// if used through the context.
    fn activate(&mut self, index: usize) -> Result;

    /// Select the first device that meets the specified requirements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parenchyma::{Backend, HardwareKind, Native};
    ///
    /// let mut native: Backend = Backend::new::<Native>().unwrap();
    /// assert!(native.select(|hardware| hardware.kind == HardwareKind::CPU).is_ok());
    /// ```
    fn select(&mut self, pred: &Fn(&Hardware) -> bool) -> Result {

        let nth = self.selection().iter().enumerate().filter(|&(_, h)| pred(h)).map(|(i, _)| i).nth(0);

        match nth {
            Some(n) => self.activate(n),
            _ => {
                let message = "There are no devices matching the specified criteria.";
                Err(Error::new(ErrorKind::Other, message))
            }
        }
    }
}

/// Provides configuration for a context.
pub struct ContextConfig<'a, F: 'a> {
    /// The framework associated with the context.
    pub framework: &'a F,
    /// A selection of hardware.
    pub selection: Vec<Hardware>,
}