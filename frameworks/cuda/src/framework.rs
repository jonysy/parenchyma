use cuda::driver;
use cuda::error::{Error, ErrorKind};
use parenchyma::{Framework, Processor};
use super::{CudaContext, CudaDevice, CudaError, CudaMemory};

pub struct Cuda {
    available_devices: Vec<CudaDevice>,
}

impl Framework for Cuda {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "CUDA";

    /// The context representation.
    type Context = CudaContext;

    /// The device representation.
    type D = CudaDevice;

    type E = CudaError;

    /// The memory representation.
    type M = CudaMemory;

    /// Initializes the framework.
    fn new() -> Result<Cuda, CudaError> {
        let _ = driver::init()?;
        let ndevices = driver::ndevices()?;

        if ndevices == 0 {
            return Err((ErrorKind::NoDevice.into(): Error).into());
        }

        let mut devices = Vec::with_capacity(ndevices as usize);

        let range = (0..ndevices);

        for n in range {
            let h = driver::device(n as u32)?;

            devices.push(CudaDevice {
                name: h.name()?,
                multiprocessors: h.multiprocessor_count()?,
                processor: Processor::Gpu,
                handle: h,
            });
        }

        Ok(Cuda { available_devices: devices })
    }

    fn default_selection(&self) -> Vec<CudaDevice> {
        vec![self.available_devices[0].clone()]
    }
}