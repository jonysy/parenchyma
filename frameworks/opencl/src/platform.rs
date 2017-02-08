use super::{cl, core};
use super::OpenCLDevice;

#[derive(Debug)]
pub struct OpenCLPlatform {
    id: core::PlatformId,
    pub name: String,
    pub available_devices: Vec<OpenCLDevice>,
}

impl<'p> From<&'p cl::Platform> for OpenCLPlatform {

    fn from(cl_platform: &cl::Platform) -> Self {

        let id = *cl_platform.as_core();
        let name = cl_platform.name();
        let available_devices = {
            let list = cl::Device::list_all(&cl_platform);
            list.unwrap().iter().map(From::from).collect()
        };

        OpenCLPlatform {
            id: id,
            name: name,
            available_devices: available_devices,
        }
    }
}