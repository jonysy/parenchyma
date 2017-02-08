use cuda::{self, CudaContextHandle, CudaDriver};
use parenchyma;
use parenchyma::error::Result;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
pub use super::{Cuda, Device, Memory};

#[derive(Debug)]
pub struct Context {
    id: Rc<CudaContextHandle>,
    selected_devices: Vec<Device>,
}

impl parenchyma::Context for Context {
    type Framework = Cuda;

    /// Creates a new CUDA context for computation.
    fn new(devices: Vec<Device>) -> Result<Self> {

        let len = devices.len();

        match len {
            1 => {
                // let f = cuda::CudaContextFlag::CU_CTX_SCHED_BLOCKING_SYNC;
                // let h = devices[0].handle;
                // let context = CudaDriver::create_context(f, h)?;

                // Ok(Context { id: Rc::new(context), selected_devices: devices })

                unimplemented!()
            },
            _ => {

                unimplemented!()
            }
        }
    }

    /// Allocates memory
    fn allocate_memory(&self, size: usize) -> Result<Memory> {

        unimplemented!()
    }
}

impl PartialEq for Context {

    fn eq(&self, other: &Self) -> bool {

        unimplemented!()
    }
}

impl Eq for Context { }

impl Hash for Context {

    fn hash<H>(&self, state: &mut H) where H: Hasher {
        
        unimplemented!()
    }
}