//! OpenCL backend support.
//!
//! ## Scalar Data Types
//!
//! TODO
//!
//! Built-in scalar data types:
//! https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
//!
//! ## Terminology
//!
//! Work-group: A collection of work items + has a unique work-group ID. work-groups are assigned
//! to execute on compute-units
//!
//! Work-item: An instance of a kernel at run time + has a unique ID within the work-group
//!
//! TODO
//!
//! ## Flow
//!
//! - Initialize the framework
//! - Select the desired platform
//! - Select the desired devices from the platform
//! - Create a context 
//! - create a command queue per device
//! - Compile programs
//!     - A program is essentially a collection of kernels.
//! - Create a kernel from the successfully compiled program
//!     - A kernel is the smallest unit of execution. Kernels are expensive to start so they're
//!     typically used to do a large amount of work.
//! - Specify arguments to the kernel
//! - Allocate memory on devices
//! - Transfer data to devices
//! - Execute
//! - Transfer results back
//! - Free memory on devices
//!
//! ## Events (TODO)
//!
//! Most OpenCL operations happen asynchronously on the OpenCL Device. To provide the possibility 
//! to order and synchronize multiple operations, the execution of an operation yields a event 
//! object. This event can be used as an input to other operations which will wait until this event 
//! has finished executing to run.

pub mod hl;
pub mod sh;

// use super::super::Framework;

/// Provides the OpenCL framework.
#[derive(Debug)]
pub struct OpenCL {
    // /// A list of available devices for the first platform found.
    // pub available_devices: Vec<OpenClDevice>,
}

// impl OpenCL {

//     /// Attempts to initialize the framework.
//     pub fn new() -> Result<OpenCL> {

//         unimplemented!()
//     }
// }