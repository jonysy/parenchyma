use cuda::CudaDriver;
use cuda::error::{Error, ErrorKind};
use parenchyma::{Framework, Processor};
use super::{Context, Device, Memory};

pub struct Cuda {
    devices: Vec<Device>,
}

impl Framework for Cuda {
    /// The name of the framework.
    const FRAMEWORK_NAME: &'static str = "CUDA";

    // type Err = 

    /// The context representation.
    type Context = Context;

    /// The device representation.
    type Device = Device;

    /// The memory representation.
    type Memory = Memory;

    /// Initializes the framework.
    fn new() -> Cuda {
        let cuda = || -> Result<Cuda, Error> {
            let _ = CudaDriver::init()?;

            let ndevices = CudaDriver::ndevices()?;

            if ndevices == 0 {
                return Err(ErrorKind::NoDevice.into());
            }

            let mut devices = vec![];

            for n in 0..ndevices {
                let dev = CudaDriver::device(n)?;

                let name = dev.name()?;

                devices.push(Device {
                    name: dev.name()?,
                    multiprocessors: dev.multiprocessor_count()?,
                    processor: Processor::Gpu,
                    handle: dev,
                });
            }

            Ok(Cuda {
                devices: devices,
            })
        };

        match cuda() {
            Ok(cuda) => cuda,
            Err(error) => panic!("{}", error)
        }
    }
}