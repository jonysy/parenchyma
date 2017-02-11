use std::{error, fmt, result};

pub type Result<T = ()> = result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    payload: Option<Box<error::Error + Send + Sync>>,
}

#[derive(Debug, Copy, Clone)]
pub enum ErrorKind {
    /// No OpenCL devices that matched device_type were found.
    DeviceNotFound,
    /// A device is currently not available even though the device was returned by clGetDeviceIDs.
    DeviceNotAvailable,

    CompilerNotAvailable,
    MemObjectAllocationFailure,
    OutOfResources,
    OutOfHostMemory,
    ProfilingInfoNotAvailable,
    MemCopyOverlap,
    ImageFormatMismatch,
    ImageFormatNotSupported,
    BuildProgramFailure,
    MapFailure,
    MisalignedSubBufferOffset,
    ExecStatusErrorForEventsInWaitList,
    InvalidValue,
    InvalidDeviceType,
    InvalidPlatform,
    InvalidDevice,
    InvalidContext,
    InvalidQueueProperties,
    InvalidCommandQueue,
    InvalidHostPtr,
    InvalidMemObject,
    InvalidImageFormatDescriptor,
    InvalidImageSize,
    InvalidSampler,
    InvalidBinary,
    InvalidBuildOptions,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefinition,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidKernelArgs,
    InvalidWorkDimension,
    InvalidWorkGroupSize,
    InvalidWorkItemSize,
    InvalidGlobalOffset,
    InvalidEventWaitList,
    InvalidEvent,
    InvalidOperation,
    InvalidGlObject,
    InvalidBufferSize,
    InvalidMipLevel,
    InvalidGlobalWorkSize,
    InvalidProperty,
    PlatformNotFoundKhr,
}

impl From<::opencl_sys::CLStatus> for ErrorKind {

    fn from(cl_status: ::opencl_sys::CLStatus) -> ErrorKind {
        use self::ErrorKind::*;
        use opencl_sys::CLStatus::*;

        match cl_status {
            CL_SUCCESS => unreachable!(),
            CL_DEVICE_NOT_FOUND => DeviceNotFound,
            CL_DEVICE_NOT_AVAILABLE => DeviceNotAvailable,
            CL_COMPILER_NOT_AVAILABLE => CompilerNotAvailable,
            CL_MEM_OBJECT_ALLOCATION_FAILURE => MemObjectAllocationFailure,
            CL_OUT_OF_RESOURCES => OutOfResources,
            CL_OUT_OF_HOST_MEMORY => OutOfHostMemory,
            CL_PROFILING_INFO_NOT_AVAILABLE => ProfilingInfoNotAvailable,
            CL_MEM_COPY_OVERLAP => MemCopyOverlap,
            CL_IMAGE_FORMAT_MISMATCH => ImageFormatMismatch,
            CL_IMAGE_FORMAT_NOT_SUPPORTED => ImageFormatNotSupported,
            CL_BUILD_PROGRAM_FAILURE => BuildProgramFailure,
            CL_MAP_FAILURE => MapFailure,
            CL_MISALIGNED_SUB_BUFFER_OFFSET => MisalignedSubBufferOffset,
            CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => ExecStatusErrorForEventsInWaitList,
            CL_INVALID_VALUE => InvalidValue,
            CL_INVALID_DEVICE_TYPE => InvalidDeviceType,
            CL_INVALID_PLATFORM => InvalidPlatform,
            CL_INVALID_DEVICE => InvalidDevice,
            CL_INVALID_CONTEXT => InvalidContext,
            CL_INVALID_QUEUE_PROPERTIES => InvalidQueueProperties,
            CL_INVALID_COMMAND_QUEUE => InvalidCommandQueue,
            CL_INVALID_HOST_PTR => InvalidHostPtr,
            CL_INVALID_MEM_OBJECT => InvalidMemObject,
            CL_INVALID_IMAGE_FORMAT_DESCRIPTOR => InvalidImageFormatDescriptor,
            CL_INVALID_IMAGE_SIZE => InvalidImageSize,
            CL_INVALID_SAMPLER => InvalidSampler,
            CL_INVALID_BINARY => InvalidBinary,
            CL_INVALID_BUILD_OPTIONS => InvalidBuildOptions,
            CL_INVALID_PROGRAM => InvalidProgram,
            CL_INVALID_PROGRAM_EXECUTABLE => InvalidProgramExecutable,
            CL_INVALID_KERNEL_NAME => InvalidKernelName,
            CL_INVALID_KERNEL_DEFINITION => InvalidKernelDefinition,
            CL_INVALID_KERNEL => InvalidKernel,
            CL_INVALID_ARG_INDEX => InvalidArgIndex,
            CL_INVALID_ARG_VALUE => InvalidArgValue,
            CL_INVALID_ARG_SIZE => InvalidArgSize,
            CL_INVALID_KERNEL_ARGS => InvalidKernelArgs,
            CL_INVALID_WORK_DIMENSION => InvalidWorkDimension,
            CL_INVALID_WORK_GROUP_SIZE => InvalidWorkGroupSize,
            CL_INVALID_WORK_ITEM_SIZE => InvalidWorkItemSize,
            CL_INVALID_GLOBAL_OFFSET => InvalidGlobalOffset,
            CL_INVALID_EVENT_WAIT_LIST => InvalidEventWaitList,
            CL_INVALID_EVENT => InvalidEvent,
            CL_INVALID_OPERATION => InvalidOperation,
            CL_INVALID_GL_OBJECT => InvalidGlObject,
            CL_INVALID_BUFFER_SIZE => InvalidBufferSize,
            CL_INVALID_MIP_LEVEL => InvalidMipLevel,
            CL_INVALID_GLOBAL_WORK_SIZE => InvalidGlobalWorkSize,
            CL_INVALID_PROPERTY => InvalidProperty,
            CL_PLATFORM_NOT_FOUND_KHR => PlatformNotFoundKhr,
        }
    }
}

impl ErrorKind {

    fn as_str(&self) -> &'static str {

        unimplemented!()
    }
}

impl From<ErrorKind> for Error {

    fn from(kind: ErrorKind) -> Error {
        Error::_new(kind, None)
    }
}

impl Error {

    /// Creates a new error from a known kind of error as well as an arbitrary error payload.
    pub fn new<K, E>(kind: K, payload: E) -> Error 
        where K: Into<ErrorKind>, 
              E: Into<Box<error::Error + Send + Sync>>
    {

        Self::_new(kind.into(), Some(payload.into()))
    }

    // "De-generization" technique..
    fn _new(kind: ErrorKind, payload: Option<Box<error::Error + Send + Sync>>) -> Error {

        Error {
            kind: kind,
            payload: payload
        }
    }

    pub fn get_ref(&self) -> Option<&(error::Error + Send + Sync + 'static)> {
        use std::ops::Deref;

        match self.payload {
            Some(ref payload) => Some(payload.deref()),
            _ => None
        }
    }

    /// Returns the corresponding `ErrorKind` for this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

        write!(fmt, "{}", self.kind.as_str())
    }
}

impl error::Error for Error {

    fn description(&self) -> &str {

        if let Some(ref payload) = self.payload {
            payload.description()
        } else {
            self.kind.as_str()
        }
    }

    fn cause(&self) -> Option<&error::Error> {

        match self.payload {
            Some(ref payload) => {
                payload.cause()
            },
            _ => {
                None
            }
        }
    }
}