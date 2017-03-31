use framework::{Framework, FrameworkExt};
use hardware::{Hardware, HardwareType};
use super::Context;
use super::api::{core, import};

/// Provides the OpenCL framework.
///
/// # Flow
///
/// Since multiple platforms can exist, the first available platform is selected during 
/// the initialization. A list of available devices are then provided for your choosing. Then,
/// the provided selection of devices are used to create a context, with a command queue for each
/// device. At this stage, a program(s) is compiled. A (host) program is essentially a collection 
/// of kernels. A kernel is the smallest unit of execution.
///
/// In OpenCL, the host code can read in a kernel binary (i.e., compiled off-line) or a kernel 
/// source file (i.e., compile on-line). More information on on-line/off-line compilation can be
/// found [here][1]. Kernels are expensive to start, so they're typically used to do a large amount
/// of work. Memory allocated on an OpenCL device can be used when executing kernels, and then 
/// transfered back.
///
/// Work-groups, a collection of work-items, are assigned to execute on compute-units. A work-item
/// is an instance of a kernel as runtime. That kernel instance is at a point in an index, which 
/// can be thought of as a grid and the work-groups which contain the work-items can be thought of 
/// as sub-grids within the grid. The work-groups can be defined explicitly or implicitly by 
/// simply specifying the number of work-items, both dealing with data parallelism. In terms of task
/// parallelism, kernels are executed independent of an index space.
/// It should also be noted that there are [built-in scalar data types][2] along with
/// [built-in functions][3].
///
/// [1]: https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/online-offline-compilation/
/// [2]: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
/// [3]: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/mathFunctions.html
#[derive(Debug)]
pub struct OpenCL {
    /// A list of available devices for the first platform found.
    ///
    /// Platforms are defined by the implementation. Platforms enables the host to interact with 
    /// OpenCL-capable devices.
    pub available_hardware: Vec<Hardware>,
}

impl Framework for OpenCL {

    const ID: &'static str = "OpenCL";

    fn hardware(&self) -> &[Hardware] {
        &self.available_hardware
    }
}

impl FrameworkExt for OpenCL {

    type Context = Context;
}

impl Default for OpenCL {

    fn default() -> OpenCL {
        let available_hardware = {
            core::platforms().unwrap().remove(0).devices().unwrap().into_iter()
                //.skip(|dev_handle| dev_handle.available())
                .map(|device|
                    Hardware {
                        id: *device.ptr() as usize,
                        framework: OpenCL::ID,
                        processor: match device.type_().unwrap() {
                            import::CL_DEVICE_TYPE_CPU => HardwareType::CPU,
                            import::CL_DEVICE_TYPE_GPU => HardwareType::GPU,
                            import::CL_DEVICE_TYPE_ACCELERATOR => HardwareType::Accelerator,
                            _ => HardwareType::Other,
                        },
                        name: device.name().unwrap(),
                        compute_units: device.max_compute_units().unwrap() as usize,
                    })
                .collect()
        };
        
        OpenCL { available_hardware }
    }
}