use ocl;
use ocl::Platform as Implementation;
use ocl::enums::{DeviceInfo, DeviceInfoResult};
use ocl::flags::{DEVICE_TYPE_ACCELERATOR, DEVICE_TYPE_CPU, DEVICE_TYPE_GPU};

use super::OpenCLContext;
use super::super::super::error::Result;
use super::super::super::framework::{Framework, FrameworkCtor};
use super::super::super::hardware::{Hardware, HardwareKind};

/// Provides the Open CL framework.
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
    available_hardware: Vec<Hardware>,
    /// The specific Open CL implementation (e.g., AMD APP, NVIDIA or Intel Open CL)
    ///
    /// Platforms are defined by the implementation. Platforms enables the host to interact with 
    /// OpenCL-capable devices.
    pub(in frameworks::open_cl) implementation: Implementation,
}

impl OpenCL {
    pub(in frameworks::open_cl) const ID: &'static str = "Open CL";
}

impl Framework for OpenCL {
    fn name(&self) -> &'static str {
        return OpenCL::ID;
    }

    fn hardware(&self) -> &[Hardware] {
        &self.available_hardware
    }
}

impl<P> FrameworkCtor<P> for OpenCL where P: 'static {
    type Context = OpenCLContext<P>;

    fn new() -> Result<Self> {
        let ignore_env_var = false;
        let implementation = Implementation::first(ignore_env_var)?;
        let devices = ocl::Device::list_all(implementation)?;

        let available_hardware = {

            devices.iter().enumerate()

            .filter(|&(_, d)| {
                use ocl::enums::DeviceInfo::{MaxComputeUnits, Type};
                use ocl::enums::DeviceInfoResult::Error;

                let _1 = d.is_available().unwrap_or(false);
                let _2 = match d.info(Type) { Error(_) => false, _ => true };
                let _3 = match d.info(MaxComputeUnits) { Error(_) => false, _ => true };

                _1 || _2 || _3
            })

            .map(|(i, d)| {
                
                let kind = {
                    match d.info(DeviceInfo::Type) {
                        DeviceInfoResult::Type(t) => match t {
                            DEVICE_TYPE_ACCELERATOR => HardwareKind::Accelerator,
                            DEVICE_TYPE_CPU => HardwareKind::CPU,
                            DEVICE_TYPE_GPU => HardwareKind::GPU,
                            _ => HardwareKind::Unknown,
                        },
                        _ => unreachable!(),
                    }
                };

                let compute_units = {
                    match d.info(DeviceInfo::MaxComputeUnits) {
                        DeviceInfoResult::MaxComputeUnits(n) => n as usize,
                        _ => unreachable!(),
                    }
                };

                Hardware {
                    id: i,
                    framework: OpenCL::ID,
                    kind,
                    name: d.name(),
                    compute_units,
                }
            })

            .collect::<Vec<Hardware>>()
        };

        Ok(OpenCL {  available_hardware, implementation })
    }
}