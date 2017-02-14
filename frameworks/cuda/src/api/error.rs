use std::{error, fmt, result};
use super::sys;

pub type Result<T = ()> = result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    payload: Option<Box<error::Error + Send + Sync>>,
}

#[derive(Debug, Copy, Clone)]
pub enum ErrorKind {
    /// This indicates that one or more of the parameters passed to the API call is not within 
    /// an acceptable range of values.
    InvalidValue,
    /// The API call failed because it was unable to allocate enough memory to perform the 
    /// requested operation.
    OutOfMemory,
    /// This indicates that the CUDA driver has not been initialized with cuInit() or that 
    /// initialization has failed.
    NotInitialized,
    /// This indicates that the CUDA driver is in the process of shutting down.
    Deinitialized,
    /// This indicates profiler is not initialized for this run. This can happen when the 
    /// application is running with external profiling tools like visual profiler.
    ProfilerDisabled,
    /// [Deprecated]
    /// This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt 
    /// to enable/disable the profiling via cuProfilerStart or cuProfilerStop 
    /// without initialization.
    ProfilerNotInitialized,
    /// [Deprecated]
    /// This error return is deprecated as of CUDA 5.0. It is no longer an error to 
    /// call cuProfilerStart() when profiling is already enabled.
    ProfilerAlreadyStarted,
    /// [Deprecated]
    /// This error return is deprecated as of CUDA 5.0. It is no longer an error to 
    /// call cuProfilerStop() when profiling is already disabled.
    ProfilerAlreadyStopped,
    /// This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
    NoDevice,
    /// This indicates that the device ordinal supplied by the user does not correspond to 
    /// a valid CUDA device.
    InvalidDevice,
    /// This indicates that the device kernel image is invalid. This can also indicate an 
    /// invalid CUDA module.
    InvalidImage,
    /// This most frequently indicates that there is no context bound to the current thread. This 
    /// can also be returned if the context passed to an API call is not a valid 
    /// handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be 
    /// returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). 
    /// See cuCtxGetApiVersion() for more details.
    InvalidContext,
    /// [Deprecated]
    /// This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to 
    /// push the active context via cuCtxPushCurrent().
    ///
    /// This indicated that the context being supplied as a parameter to the API call was 
    /// already the active context.
    ContextAlreadyCurrent,
    /// This indicates that a map or register operation has failed.
    MapFailed,
    /// This indicates that an unmap or unregister operation has failed.
    UnmapFailed,
    /// This indicates that the specified array is currently mapped and thus cannot be destroyed.
    ArrayIsMapped,
    /// This indicates that the resource is already mapped.
    AlreadyMapped,
    /// This indicates that there is no kernel image available that is suitable for the 
    /// device. This can occur when a user specifies code generation options for a particular 
    /// CUDA source file that do not include the corresponding device configuration.
    NoBinaryForGpu,
    /// This indicates that a resource has already been acquired.
    AlreadyAcquired,
    /// This indicates that a resource is not mapped.
    NotMapped,
    /// This indicates that a mapped resource is not available for access as an array.
    NotMappedAsArray,
    /// This indicates that a mapped resource is not available for access as a pointer.
    NotMappedAsPointer,
    /// This indicates that an uncorrectable ECC error was detected during execution.
    EccUncorrectable,
    /// This indicates that the CUlimit passed to the API call is not supported by the active device.
    UnsupportedLimit,
    /// This indicates that the CUcontext passed to the API call can only be bound to a 
    /// single CPU thread at a time but is already bound to a CPU thread.
    ContextAlreadyInUse,
    /// This indicates that peer access is not supported across the given devices.
    PeerAccessUnsupported,
    /// This indicates that a PTX JIT compilation failed.
    InvalidPtx,
    /// This indicates an error with OpenGL or DirectX context.
    InvalidGraphicsContext,
    /// This indicates that an uncorrectable NVLink error was detected during the execution.
    NvlinkUncorrectable,
    /// This indicates that the device kernel source is invalid.
    InvalidSource,
    /// This indicates that the file specified was not found.
    FileNotFound,
    /// This indicates that a link to a shared object failed to resolve.
    SharedObjectSymbolNotFound,
    /// This indicates that initialization of a shared object failed.
    SharedObjectInitFailed,
    /// This indicates that an OS call failed.
    OperatingSystem,
    /// This indicates that a resource handle passed to the API call was not valid. Resource 
    /// handles are opaque types like CUstream and CUevent.
    InvalidHandle,
    /// This indicates that a named symbol was not found. Examples of symbols are global/constant 
    /// variable names, texture names, and surface names.
    NotFound,
    /// This indicates that asynchronous operations issued previously have not completed yet. This 
    /// result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which 
    /// indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
    NotReady,
    /// While executing a kernel, the device encountered a load or store instruction on an invalid 
    /// memory address. The context cannot be used, so it must be destroyed (and a new one should 
    /// be created). All existing device memory allocations from this context are invalid and 
    /// must be reconstructed if the program is to continue using CUDA.
    IllegalAddress,
    /// This indicates that a launch did not occur because it did not have appropriate resources. 
    /// This error usually indicates that the user has attempted to pass too many arguments to the 
    /// device kernel, or the kernel launch specifies too many threads for the kernel's register 
    /// count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is 
    /// expected) is equivalent to passing too many arguments and can also result in this error.
    LaunchOutOfResources,
    /// This indicates that the device kernel took too long to execute. This can only occur if 
    /// timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT 
    /// for more information. The context cannot be used (and must be destroyed similar to 
    /// CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from this context are 
    /// invalid and must be reconstructed if the program is to continue using CUDA.
    LaunchTimeout,
    /// This error indicates a kernel launch that uses an incompatible texturing mode.
    LaunchIncompatibleTexturing,
    /// This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer 
    /// access to a context which has already had peer access to it enabled.
    PeerAccessAlreadyEnabled,
    /// This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which 
    /// has not been enabled yet via cuCtxEnablePeerAccess().
    PeerAccessNotEnabled,
    /// This error indicates that the primary context for the specified device has already been initialized.
    PrimaryContextActive,
    /// This error indicates that the context current to the calling thread has been destroyed 
    /// using cuCtxDestroy, or is a primary context which has not yet been initialized.
    ContextIsDestroyed,
    /// A device-side assert triggered during kernel execution. The context cannot be used 
    /// anymore, and must be destroyed. All existing device memory allocations from this context 
    /// are invalid and must be reconstructed if the program is to continue using CUDA.
    Assert,
    /// This error indicates that the hardware resources required to enable peer access have 
    /// been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess().
    TooManyPeers,
    /// This error indicates that the memory range passed to cuMemHostRegister() has already 
    /// been registered.
    HostMemoryAlreadyRegistered,
    /// This error indicates that the pointer passed to cuMemHostUnregister() does not correspond 
    /// to any currently registered memory region.
    HostMemoryNotRegistered,
    /// While executing a kernel, the device encountered a stack error. This can be due to stack 
    /// corruption or exceeding the stack size limit. The context cannot be used, so it must be 
    /// destroyed (and a new one should be created). All existing device memory allocations from 
    /// this context are invalid and must be reconstructed if the program is to continue using CUDA.
    HardwareStackError,
    /// While executing a kernel, the device encountered an illegal instruction. The context cannot 
    /// be used, so it must be destroyed (and a new one should be created). All existing device 
    /// memory allocations from this context are invalid and must be reconstructed if the program 
    /// is to continue using CUDA.
    IllegalInstruction,
    /// While executing a kernel, the device encountered a load or store instruction on a memory 
    /// address which is not aligned. The context cannot be used, so it must be destroyed (and a 
    /// new one should be created). All existing device memory allocations from this context are 
    /// invalid and must be reconstructed if the program is to continue using CUDA.
    MisalignedAddress,
    /// While executing a kernel, the device encountered an instruction which can only operate on 
    /// memory locations in certain address spaces (global, shared, or local), but was supplied a 
    /// memory address not belonging to an allowed address space. The context cannot be used, so 
    /// it must be destroyed (and a new one should be created). All existing device memory 
    /// allocations from this context are invalid and must be reconstructed if the program is 
    /// to continue using CUDA.
    InvalidAddressSpace,
    /// While executing a kernel, the device program counter wrapped its address space. The context 
    /// cannot be used, so it must be destroyed (and a new one should be created). All existing 
    /// device memory allocations from this context are invalid and must be reconstructed if the 
    /// program is to continue using CUDA.
    InvalidPc,
    /// An exception occurred on the device while executing a kernel. Common causes include 
    /// dereferencing an invalid device pointer and accessing out of bounds shared memory. The 
    /// context cannot be used, so it must be destroyed (and a new one should be created). All 
    /// existing device memory allocations from this context are invalid and must be reconstructed 
    /// if the program is to continue using CUDA.
    LaunchFailed,
    /// This error indicates that the attempted operation is not permitted.
    NotPermitted,
    /// This error indicates that the attempted operation is not supported on the current 
    /// system or device.
    NotSupported,
    /// This indicates that an unknown internal error has occurred.
    Unknown,
}

impl From<sys::cudaError_enum> for ErrorKind {

    fn from(cuda_error_enum: sys::cudaError_enum) -> ErrorKind {
        use super::sys::cudaError_enum::*;
        use self::ErrorKind::*;

        match cuda_error_enum {
            CUDA_SUCCESS => {
                unreachable!()
            },
            CUDA_ERROR_INVALID_VALUE => {
                InvalidValue
            },
            CUDA_ERROR_OUT_OF_MEMORY => {
                OutOfMemory
            },
            CUDA_ERROR_NOT_INITIALIZED => {
                NotInitialized
            },
            CUDA_ERROR_DEINITIALIZED => {
                Deinitialized
            },
            CUDA_ERROR_PROFILER_DISABLED => {
                ProfilerDisabled
            },
            CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
                ProfilerNotInitialized
            },
            CUDA_ERROR_PROFILER_ALREADY_STARTED => {
                ProfilerAlreadyStarted
            },
            CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
                ProfilerAlreadyStopped
            },
            CUDA_ERROR_NO_DEVICE => {
                NoDevice
            },
            CUDA_ERROR_INVALID_DEVICE => {
                InvalidDevice
            },
            CUDA_ERROR_INVALID_IMAGE => {
                InvalidImage
            },
            CUDA_ERROR_INVALID_CONTEXT => {
                InvalidContext
            },
            CUDA_ERROR_CONTEXT_ALREADY_CURRENT => {
                ContextAlreadyCurrent
            },
            CUDA_ERROR_MAP_FAILED => {
                MapFailed
            },
            CUDA_ERROR_UNMAP_FAILED => {
                UnmapFailed
            },
            CUDA_ERROR_ARRAY_IS_MAPPED => {
                ArrayIsMapped
            },
            CUDA_ERROR_ALREADY_MAPPED => {
                AlreadyMapped
            },
            CUDA_ERROR_NO_BINARY_FOR_GPU => {
                NoBinaryForGpu
            },
            CUDA_ERROR_ALREADY_ACQUIRED => {
                AlreadyAcquired
            },
            CUDA_ERROR_NOT_MAPPED => {
                NotMapped
            },
            CUDA_ERROR_NOT_MAPPED_AS_ARRAY => {
                NotMappedAsArray
            },
            CUDA_ERROR_NOT_MAPPED_AS_POINTER => {
                NotMappedAsPointer
            },
            CUDA_ERROR_ECC_UNCORRECTABLE => {
                EccUncorrectable
            },
            CUDA_ERROR_UNSUPPORTED_LIMIT => {
                UnsupportedLimit
            },
            CUDA_ERROR_CONTEXT_ALREADY_IN_USE => {
                ContextAlreadyInUse
            },
            CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => {
                PeerAccessUnsupported
            },
            CUDA_ERROR_INVALID_PTX => {
                InvalidPtx
            },
            CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
                InvalidGraphicsContext
            },
            CUDA_ERROR_NVLINK_UNCORRECTABLE => {
                NvlinkUncorrectable
            },
            CUDA_ERROR_INVALID_SOURCE => {
                InvalidSource
            },
            CUDA_ERROR_FILE_NOT_FOUND => {
                FileNotFound
            },
            CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                SharedObjectSymbolNotFound
            },
            CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                SharedObjectInitFailed
            },
            CUDA_ERROR_OPERATING_SYSTEM => {
                OperatingSystem
            },
            CUDA_ERROR_INVALID_HANDLE => {
                InvalidHandle
            },
            CUDA_ERROR_NOT_FOUND => {
                NotFound
            },
            CUDA_ERROR_NOT_READY => {
                NotReady
            },
            CUDA_ERROR_ILLEGAL_ADDRESS => {
                IllegalAddress
            },
            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => {
                LaunchOutOfResources
            },
            CUDA_ERROR_LAUNCH_TIMEOUT => {
                LaunchTimeout
            },
            CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                LaunchIncompatibleTexturing
            },
            CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                PeerAccessAlreadyEnabled
            },
            CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => {
                PeerAccessNotEnabled
            },
            CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => {
                PrimaryContextActive
            },
            CUDA_ERROR_CONTEXT_IS_DESTROYED => {
                ContextIsDestroyed
            },
            CUDA_ERROR_ASSERT => {
                Assert
            },
            CUDA_ERROR_TOO_MANY_PEERS => {
                TooManyPeers
            },
            CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                HostMemoryAlreadyRegistered
            },
            CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                HostMemoryNotRegistered
            },
            CUDA_ERROR_HARDWARE_STACK_ERROR => {
                HardwareStackError
            },
            CUDA_ERROR_ILLEGAL_INSTRUCTION => {
                IllegalInstruction
            },
            CUDA_ERROR_MISALIGNED_ADDRESS => {
                MisalignedAddress
            },
            CUDA_ERROR_INVALID_ADDRESS_SPACE => {
                InvalidAddressSpace
            },
            CUDA_ERROR_INVALID_PC => {
                InvalidPc
            },
            CUDA_ERROR_LAUNCH_FAILED => {
                LaunchFailed
            },
            CUDA_ERROR_NOT_PERMITTED => {
                NotPermitted
            },
            CUDA_ERROR_NOT_SUPPORTED => {
                NotSupported
            },
            CUDA_ERROR_UNKNOWN => {
                Unknown
            },
        }
    }
}

impl ErrorKind {

    fn as_str(&self) -> &'static str {

        use self::ErrorKind::*;

        match *self {
            InvalidValue => "one or more of the parameters passed to the API call is not within an acceptable range of values",
            OutOfMemory => "unable to allocate enough memory to perform the requested operation",
            NotInitialized => "either the CUDA driver has not been initialized or the initialization has failed",
            Deinitialized => "the CUDA driver is in the process of shutting down",
            ProfilerDisabled => "profiler is not initialized for this run",
            ProfilerNotInitialized => "[deprecated]",
            ProfilerAlreadyStarted => "[deprecated]",
            ProfilerAlreadyStopped => "[deprecated]",
            NoDevice => "no CUDA-capable devices were detected by the installed CUDA driver.",
            InvalidDevice => "the device ordinal supplied by the user does not correspond to a valid CUDA device",
            InvalidImage => "the device kernel image or Cuda module is invalid",
            InvalidContext => "invalid context",
            ContextAlreadyCurrent => "[deprecated]",
            MapFailed => "a map or register operation has failed",
            UnmapFailed => "an unmap or unregister operation has failed",
            ArrayIsMapped => "the specified array is currently mapped and thus cannot be destroyed",
            AlreadyMapped => "the resource is already mapped",
            NoBinaryForGpu => "there is no kernel image available that is suitable for the device",
            AlreadyAcquired => "resource has already been acquired",
            NotMapped => "resource is not mapped",
            NotMappedAsArray => "mapped resource is not available for access as an array",
            NotMappedAsPointer => "mapped resource is not available for access as a pointer",
            EccUncorrectable => "an uncorrectable ECC error was detected during execution",
            UnsupportedLimit => "the CUlimit passed to the API call is not supported by the active device",
            ContextAlreadyInUse => "the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread",
            PeerAccessUnsupported => "peer access is not supported across the given devices",
            InvalidPtx => "a PTX JIT compilation failed",
            InvalidGraphicsContext => "an OpenGL or DirectX context error occurred",
            NvlinkUncorrectable => "an uncorrectable NVLink error was detected during the execution",
            InvalidSource => "the device kernel source is invalid",
            FileNotFound => "the file specified was not found",
            SharedObjectSymbolNotFound => "a link to a shared object failed to resolve",
            SharedObjectInitFailed => "initialization of a shared object failed",
            OperatingSystem => "an OS call failed",
            InvalidHandle => "a resource handle passed to the API call was not valid",
            NotFound => "named symbol was not found",
            NotReady => "asynchronous operations issued previously have not completed yet",
            IllegalAddress => "while executing a kernel, the device encountered a load or store instruction on an invalid memory address",
            LaunchOutOfResources => "launch did not occur because it did not have appropriate resources",
            LaunchTimeout => "the device kernel took too long to execute",
            LaunchIncompatibleTexturing => "kernel launch uses an incompatible texturing mode",
            PeerAccessAlreadyEnabled => "a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled",
            PeerAccessNotEnabled => "cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess()",
            PrimaryContextActive => "the primary context for the specified device has already been initialized",
            ContextIsDestroyed => "calling thread context has been destroyed or is a primary context which has not yet been initialized",
            Assert => "a device-side assert triggered during kernel execution",
            TooManyPeers => "the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess()",
            HostMemoryAlreadyRegistered => "the memory range passed to cuMemHostRegister() has already been registered",
            HostMemoryNotRegistered => "the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region",
            HardwareStackError => "while executing a kernel, the device encountered a stack error",
            IllegalInstruction => "while executing a kernel, the device encountered an illegal instruction",
            MisalignedAddress => "misaligned address",
            InvalidAddressSpace => "invalid address space",
            InvalidPc => "while executing a kernel, the device program counter wrapped its address space",
            LaunchFailed => "exception occurred on the device while executing a kernel",
            NotPermitted => "the attempted operation is not permitted",
            NotSupported => "the attempted operation is not supported on the current system or device",
            Unknown => "an unknown internal error has occurred",
        }
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