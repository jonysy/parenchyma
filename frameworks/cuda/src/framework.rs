use cuda::CudaDriver;
use cuda::error::Error;
use super::Device;

pub struct Cuda {
    devices: Vec<Device>,
}

impl /*Framework for*/ Cuda {

    pub fn new() -> Cuda {
        let cuda = || -> Result<Cuda, Error> {
            let driver = CudaDriver::init()?;

            for n in 0..driver.ndevices()? {
                let dev = driver.device(n)?;

                let name = dev.name()?;
            }

            unimplemented!()
        };

        match cuda() {
            Ok(cuda) => cuda,
            Err(error) => panic!("{}", error)
        }
    }
}