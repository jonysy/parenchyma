use api::{driver, ContextHandle};
use parenchyma::{Context, NativeContext, NativeMemory};
use std::rc::Rc;
pub use super::{Cuda, CudaDevice, CudaMemory, Result};

#[derive(Clone, Debug)]
pub struct CudaContext {
    id: Rc<ContextHandle>,
    selected_device: CudaDevice,
}

impl Context for CudaContext {
    type F = Cuda;

    /// Creates a new CUDA context for computation.
    fn new(device: CudaDevice) -> Result<Self> {
        use api::ContextFlag::CU_CTX_SCHED_BLOCKING_SYNC;

        let context = driver::create_context(CU_CTX_SCHED_BLOCKING_SYNC, &device.handle)?;

        Ok(CudaContext { id: Rc::new(context), selected_device: device })
    }

    /// Allocates memory
    fn allocate_memory(&self, size: usize) -> Result<CudaMemory> {

        let dptr = driver::mem_alloc(size)?;

        Ok(CudaMemory { dptr: dptr })
    }

    fn synch_in(&self, destn: &mut CudaMemory, _: &NativeContext, src: &NativeMemory) -> Result {

        driver::mem_cpy_h_to_d(&destn.dptr, src.as_slice().as_ptr(), src.len())
    }

    fn synch_out(&self, src: &CudaMemory, _: &NativeContext, destn: &mut NativeMemory) -> Result {

        driver::mem_cpy_d_to_h(destn.as_mut_slice().as_mut_ptr(), &src.dptr, destn.len())
    }
}

impl PartialEq for CudaContext {

    fn eq(&self, other: &Self) -> bool {

        self.id == other.id
    }
}

impl Eq for CudaContext { }