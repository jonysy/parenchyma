use cuda_sys;
use error::{Error, ErrorKind, Result};
use std::mem;

#[derive(Debug)]
pub struct CudaContextHandle(pub(super) cuda_sys::CUcontext);

impl CudaContextHandle {

    /// Destroy a CUDA Context. <sup>*</sup>There's no need to manually call this method __unless 
    /// you know what you're doing__. `destroy` is automatically called when the context goes
    /// out of scope.
    pub fn destroy(self) -> Result {
        unsafe {
            match cuda_sys::cuCtxDestroy_v2(self.0) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(()),

                e @ _ => 
                    Err(Error::from(e.into(): ErrorKind))
            }
        }
    }
}

impl Drop for CudaContextHandle {

    fn drop(&mut self) {

        unsafe {
            let mut p = mem::uninitialized();
            mem::swap(self, &mut p);
            let _ = p.destroy();
        }
    }
}