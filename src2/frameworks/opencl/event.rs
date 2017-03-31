use error::Error;
use futures::{Async, Future};

/// Represents an OpenCL event.
///
/// Most OpenCL operations happen asynchronously on an OpenCL device. Multiple OpenCL operations 
/// can be ordered and synchronized by way of the event objects yielded by the operations. An event
/// object can be used as an input to other operations which will wait until the event has 
/// finished executing to run.
pub type Event = super::api::core::Event;

impl Future for Event {

    type Item = ();

    type Error = Error;

    fn poll(&mut self) -> Result<Async<Self::Item>, Error> {
        if self.is_complete()? {
            Ok(Async::Ready(()))
        } else {
            Ok(Async::NotReady)
        }
    }
}