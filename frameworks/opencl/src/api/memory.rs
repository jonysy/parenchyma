use opencl_sys;
use error::{ErrorKind, Result};

// fn clCreateBuffer(
//     context: cl_context,
//     flags: cl_mem_flags,
//     size: libc::size_t,
//     host_ptr: *mut raw::c_void,
//     errcode_ret: *mut cl_int
// ) -> cl_mem;

#[derive(Debug)]
pub struct MemoryPtr(opencl_sys::cl_mem);

impl MemoryPtr {

}