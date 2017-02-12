use cuda_sys;
use error::{Error, ErrorKind, Result};

pub struct Memory(pub(super) u64 /* dptr */);

impl Memory {

    /// Frees the memory space pointed to by dptr.
    fn mem_free(&self) -> Result {

        unsafe {

            match cuda_sys::cuMemFree_v2(self.0) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(()),

                e @ _ =>
                    Err(Error::from(e.into(): ErrorKind)),
            }
        }
    }
}

impl Drop for Memory {

    fn drop(&mut self) {

        self.mem_free().unwrap()
    }
}