// TODO
/// Represents an OpenCL event.
///
/// Most OpenCL operations happen asynchronously on the OpenCL Device. To provide the possibility 
/// to order and synchronize multiple operations, the execution of an operation yields a event 
/// object. This event can be used as an input to other operations which will wait until this event 
/// has finished executing to run.
#[derive(Clone, Debug)]
pub struct OpenCLEvent;