/// Represents an OpenCL event.
///
/// Most OpenCL operations happen asynchronously on an OpenCL device. Multiple OpenCL operations 
/// can be ordered and synchronized by way of the event objects yielded by the operations. An event
/// object can be used as an input to other operations which will wait until the event has 
/// finished executing to run.
#[derive(Debug)]
pub struct OpenCLEvent(() /* TODO */);