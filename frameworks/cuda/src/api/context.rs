use std::mem;
use super::error::{Error, ErrorKind, Result};
use super::sys;

#[derive(Debug)]
pub struct ContextHandle(pub(super) sys::CUcontext);

impl ContextHandle {

    /// Destroy a CUDA Context. <sup>*</sup>There's no need to manually call this method __unless 
    /// you know what you're doing__. `destroy` is automatically called when the context goes
    /// out of scope.
    pub fn destroy(self) -> Result {
        unsafe {
            match sys::cuCtxDestroy_v2(self.0) {
                sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(()),

                e @ _ => 
                    Err(Error::from(e.into(): ErrorKind))
            }
        }
    }
}

impl Drop for ContextHandle {

    fn drop(&mut self) {

        unsafe {
            let mut p = mem::uninitialized();
            mem::swap(self, &mut p);
            let _ = p.destroy();
        }
    }
}

impl PartialEq<ContextHandle> for ContextHandle {

    fn eq(&self, other: &ContextHandle) -> bool {

        self.0 == other.0
    }
}