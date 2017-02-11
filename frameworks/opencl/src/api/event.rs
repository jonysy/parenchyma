use opencl_sys;

pub struct EventPtr(pub(super) opencl_sys::cl_event);