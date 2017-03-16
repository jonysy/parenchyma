use {BoxContext, Build, Context, Error, ErrorKind, ExtensionPackage, Framework};
use {Hardware, HardwareKind, Result};
use std::marker::Unsize;
use super::{OpenCLContext, OpenCLDevice};
use super::super::{foreign, high};
use utility::{TryDefault, Uninitialized};

const OPEN_CL: &'static str = "OpenCL";

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
    /// raw device pointers
    cache: Vec<high::Device>,
    /// A list of available devices for the first platform found.
    ///
    /// Platforms are defined by the implementation. Platforms enables the host to interact with 
    /// OpenCL-capable devices.
    pub available_hardware: Vec<Hardware>,
}

impl Framework for OpenCL {

    const FRAMEWORK_NAME: &'static str = OPEN_CL;

    fn available_hardware(&self) -> Vec<Hardware> {
        self.available_hardware.clone()
    }
}

impl<X> BoxContext<X> for OpenCL 
    where X: ExtensionPackage + Build<OpenCLContext<Uninitialized>>,
          OpenCLContext<X>: Unsize<X::Extension> 
          {

    fn enclose(&self, hw_selection: Vec<Hardware>) -> Result<Box<Context<Package = X>>> {
        let indices: Vec<usize> = hw_selection.iter().map(|hardware| hardware.id).collect();

        let devices: Vec<high::Device> = self.cache
            .iter()
            .enumerate()
            .filter(|&(idx, _)| indices.contains(&idx))
            .map(|(_, device)| device.clone())
            .collect();

        // create a context for the devices
        let hl_context = high::Context::new(&devices)?;

        let mut selection = vec![];

        for raw_device in devices.into_iter() {

            // create a command queue (with profiling enabled, needed for timing kernels)
            let queue = high::Queue::new(&hl_context, &raw_device, foreign::CL_QUEUE_PROFILING_ENABLE)?;

            selection.push(OpenCLDevice {
                device: raw_device,
                context: hl_context.clone(),
                queue,
            });
        }

        let mut context: OpenCLContext<Uninitialized> = OpenCLContext {
            context: hl_context,
            package: (),
            selection: selection,
            active: 0,
        };

        let package = X::build(&mut context)?;

        Ok(Box::new(OpenCLContext {
            context: context.context,
            package: package,
            selection: context.selection,
            active: context.active,
        }))
    }
}

impl TryDefault for OpenCL {

    type Err = Error;

    fn try_default() -> Result<OpenCL> {

        let mut cache = vec![];

        let available_hardware = {
            high::platforms()?.remove(0).devices()?.into_iter()
                //.skip(|dev_handle| dev_handle.available())
                .enumerate()
                .map(|(index, dev_handle)| {
                    let hardware = Hardware {
                        id: index,
                        framework: OPEN_CL,
                        kind: match dev_handle.type_().unwrap() {
                            foreign::CL_DEVICE_TYPE_CPU => HardwareKind::Central,
                            foreign::CL_DEVICE_TYPE_GPU => HardwareKind::Graphics,
                            foreign::CL_DEVICE_TYPE_ACCELERATOR => HardwareKind::Accelerator,
                            _ => HardwareKind::Other,
                        },
                        name: dev_handle.name().unwrap(),
                        compute_units: dev_handle.max_compute_units().unwrap() as usize,
                    };

                    cache.push(dev_handle);

                    hardware
                })
                .collect()
        };

        Ok(OpenCL { cache, available_hardware })
    }
}

impl From<high::Error> for Error {

    fn from(e: high::Error) -> Error {
        Error::new(ErrorKind::Framework { name: OPEN_CL }, e)
    }
}