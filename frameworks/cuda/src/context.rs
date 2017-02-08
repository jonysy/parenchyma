use cuda::{driver, ContextHandle};
use parenchyma::Context;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
pub use super::{Cuda, CudaDevice, CudaError, CudaMemory};

#[derive(Debug)]
pub struct CudaContext {
    id: Rc<ContextHandle>,
    selected_devices: Vec<CudaDevice>,
}

impl Context for CudaContext {
    type F = Cuda;

    /// Creates a new CUDA context for computation.
    fn new(devices: Vec<CudaDevice>) -> Result<Self, CudaError> {

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
    fn allocate_memory(&self, _: usize) -> Result<CudaMemory, CudaError> {

        unimplemented!()
    }
}

impl PartialEq for CudaContext {

    fn eq(&self, _: &Self) -> bool {

        unimplemented!()
    }
}

impl Eq for CudaContext { }

impl Hash for CudaContext {

    fn hash<H>(&self, _: &mut H) where H: Hasher {
        
        unimplemented!()
    }
}