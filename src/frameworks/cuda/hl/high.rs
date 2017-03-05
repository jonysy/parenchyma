use std::{cmp, ffi, ops};

use super::Result;
use super::utility;
use super::super::sh;

/// Initialize the CUDA driver API.
///
/// Initializes the driver API and must be called before any other function from the 
/// driver API. Currently, the `flags` parameter must be 0. If cuInit() has not been called, 
/// any function from the driver API will return `NotInitialized`.
///
/// # Parameters
///
/// * `flags` - Initialization flag for CUDA.
///
/// # Returns
///
/// Returns `()` if initialization was successful, otherwise returns an error kind of 
/// `InvalidValue` or `InvalidDevice`.
///
/// # Note
///
/// Note that this function may also return error codes from previous, asynchronous launches.
pub fn init() -> Result {
    unsafe {
        let ret_value = sh::cuInit(0);
        let err_message = "An error occurred while attempting to initialize the CUDA driver API";
        return utility::check_with(ret_value, err_message, || {})
    }
}

/// Returns the number of compute-capable devices.
///
/// Returns the number of devices with compute capability greater than or equal to 1.0 that 
/// are available for execution. If there is no such device, returns 0.
pub fn ndevices() -> Result<u32> {
    unsafe {
        let mut ndevices = 0;
        let ret_value = sh::cuDeviceGetCount(&mut ndevices);
        return utility::check(ret_value, || cmp::max(ndevices, 0) as u32)
    }
}

// /// Copies from host memory to device memory. dstDevice and srcHost are the base addresses of the 
// /// destination and source, respectively. ByteCount specifies the number of bytes to copy. Note 
// /// that this function is synchronous.
// pub fn mem_cpy_h_to_d(dst_device: &Memory, src_host: *const c_void, byte_count: usize) -> Result {

//     unsafe {

//         match sys::cuMemcpyHtoD_v2(dst_device.0, src_host, byte_count) {
//             sys::cudaError_enum::CUDA_SUCCESS => 
//                 Ok(()),

//             e @ _ =>
//                 Err(Error::from(e.into(): ErrorKind)),
//         }
//     }
// }

// /// Copies from device to host memory. dstHost and srcDevice specify the base pointers of the 
// /// destination and source, respectively. ByteCount specifies the number of bytes to copy. Note 
// /// that this function is synchronous.
// pub fn mem_cpy_d_to_h(dst_host: *mut c_void, src_device: &Memory, byte_count: usize) -> Result {

//     unsafe {

//         match sys::cuMemcpyDtoH_v2(dst_host, src_device.0, byte_count) {
//             sys::cudaError_enum::CUDA_SUCCESS => 
//                 Ok(()),

//             e @ _ =>
//                 Err(Error::from(e.into(): ErrorKind)),   
//         }
//     }
// }

#[derive(Debug)]
pub struct Context(sh::CUcontext);

impl Context {

    // /// Create a CUDA context.
    // ///
    // /// # Parameters
    // ///
    // /// * `f` - Context creation flags
    // pub fn new(f: ContextFlag, dev: &Device) -> Result<Context> {
    //     unsafe {
    //         let mut ctx = ptr::null_mut();

    //         match sys::cuCtxCreate_v2(&mut ctx, f as u32, dev.0) {
    //             sys::cudaError_enum::CUDA_SUCCESS => 
    //                 Ok(Context(ctx)),

    //             e @ _ =>
    //                 Err(Error::from(e.into(): ErrorKind)),
    //         }
    //     }
    // }

    /// Destroy a CUDA Context. <sup>*</sup>There's no need to manually call this method __unless 
    /// you know what you're doing__. `destroy` is automatically called when the context goes
    /// out of scope.
    pub fn destroy(&self) -> Result {
        unsafe {
            let ret_value = sh::cuCtxDestroy_v2(self.0);
            return utility::check(ret_value, || {});
        }
    }
}

impl cmp::PartialEq<Context> for Context {

    fn eq(&self, other: &Context) -> bool {

        self.0 == other.0
    }
}

impl ops::Drop for Context {

    fn drop(&mut self) {

        self.destroy().expect("failed to destroy CUDA context");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Device(sh::CUdevice);

impl Device {

    /// Returns a handle to a compute device.
    ///
    /// # Parameters
    ///
    /// * `ordinal` - Device number to get handle for.
    pub fn device(n: u32) -> Result<Device> {
        unsafe {
            let mut device_handle = 0;
            let ret_value = sh::cuDeviceGet(&mut device_handle, n as i32);
            return utility::check(ret_value, || Device(device_handle));
        }
    }

    /// Returns an identifier string for the device.
    pub fn name(&self) -> Result<String> {
        unsafe {
            const LENGTH: i32 = 1024;

            let mut array = [0; LENGTH as usize];
            let name_pointer = array.as_mut_ptr();

            let ret_value = sh::cuDeviceGetName(name_pointer, LENGTH, self.0);

            return utility::check(ret_value, || {
                let c_string = ffi::CString::from_raw(name_pointer);
                let name = c_string.into_string()
                    .expect("an unexpected error occurred while retrieving the name of a device");
                name
            });
        }
    }

    // ========= Query CUDA device attributes/properties

    /// Returns information about the device.
    pub fn attribute(&self, attribute: sh::CUdevice_attribute) -> Result<i32> {
        unsafe {
            // returned device attribute value
            let mut pi = 0;

            let ret_value = sh::cuDeviceGetAttribute(&mut pi, attribute, self.0);
            return utility::check(ret_value, || pi)
        }
    }

    /// Maximum number of threads per block.
    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Maximum x-dimension of a block.
    pub fn max_block_dim_x(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    }

    /// Maximum y-dimension of a block.
    pub fn max_block_dim_y(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    }

    /// Maximum z-dimension of a block.
    pub fn max_block_dim_z(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    }

    /// Maximum x-dimension of a grid.
    pub fn max_grid_dim_x(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    }

    /// Maximum y-dimension of a grid.
    pub fn max_grid_dim_y(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    }

    /// Maximum z-dimension of a grid.
    pub fn max_grid_dim_z(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    }

    /// Maximum amount of shared memory available to a thread block in bytes.
    pub fn max_shared_memory_per_block(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    }

    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes.
    pub fn total_constant_memory(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    }

    /// Warp size in threads.
    pub fn warp_size(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    }

    /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions 
    /// allocated through ::cuMemAllocPitch().
    pub fn max_pitch(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
    }

    /// Maximum 1D texture width.
    pub fn maximum_texture1d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    }

    /// Maximum width for a 1D texture bound to linear memory.
    pub fn maximum_texture1d_linear_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)
    }

    /// Maximum mipmapped 1D texture width.
    pub fn maximum_texture1d_mipmapped_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH)
    }

    /// Maximum 2D texture width.
    pub fn maximum_texture2d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    }

    /// Maximum 2D texture height.
    pub fn maximum_texture2d_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    }

    /// Maximum width for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)
    }

    /// Maximum height for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)
    }

    /// Maximum pitch in bytes for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_pitch(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)
    }

    /// Maximum mipmapped 2D texture width.
    pub fn maximum_texture2d_mipmapped_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH)
    }

    /// Maximum mipmapped 2D texture height.
    pub fn maximum_texture2d_mipmapped_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT)
    }

    /// Maximum 3D texture width.
    pub fn maximum_texture3d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    }

    /// Maximum 3D texture height.
    pub fn maximum_texture3d_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    }

    /// Maximum 3D texture depth.
    pub fn maximum_texture3d_depth(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    }

    /// Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_width_alternate(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE)
    }

    /// Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_height_alternate(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE)
    }

    /// Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_depth_alternate(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE)
    }

    /// Maximum cubemap texture width or height.
    pub fn maximum_texturecubemap_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)
    }

    /// Maximum 1D layered texture width.
    pub fn maximum_texture1d_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)
    }

    /// Maximum layers in a 1D layered texture.
    pub fn maximum_texture1d_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS)
    }

    /// Maximum 2D layered texture width.
    pub fn maximum_texture2d_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)
    }

    /// Maximum 2D layered texture height.
    pub fn maximum_texture2d_layered_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT)
    }

    /// Maximum layers in a 2D layered texture.
    pub fn maximum_texture2d_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS)
    }

    /// Maximum cubemap layered texture width or height.
    pub fn maximum_texturecubemap_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH)
    }

    /// Maximum layers in a cubemap layered texture.
    pub fn maximum_texturecubemap_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS)
    }

    /// Maximum 1D surface width.
    pub fn maximum_surface1d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)
    }

    /// Maximum 2D surface width.
    pub fn maximum_surface2d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)
    }

    /// Maximum 2D surface height.
    pub fn maximum_surface2d_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)
    }

    /// Maximum 3D surface width.
    pub fn maximum_surface3d_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)
    }

    /// Maximum 3D surface height.
    pub fn maximum_surface3d_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)
    }

    /// Maximum 3D surface depth.
    pub fn maximum_surface3d_depth(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)
    }

    /// Maximum 1D layered surface width.
    pub fn maximum_surface1d_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)
    }

    /// Maximum layers in a 1D layered surface.
    pub fn maximum_surface1d_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS)
    }

    /// Maximum 2D layered surface width.
    pub fn maximum_surface2d_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)
    }

    /// Maximum 2D layered surface height.
    pub fn maximum_surface2d_layered_height(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT)
    }

    /// Maximum layers in a 2D layered surface.
    pub fn maximum_surface2d_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS)
    }

    /// Maximum cubemap surface width.
    pub fn maximum_surfacecubemap_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)
    }

    /// Maximum cubemap layered surface width.
    pub fn maximum_surfacecubemap_layered_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH)
    }

    /// Maximum layers in a cubemap layered surface.
    pub fn maximum_surfacecubemap_layered_layers(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS)
    }

    /// Maximum number of 32-bit registers available to a thread block.
    pub fn max_registers_per_block(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    }

    /// The typical clock frequency in kilohertz.
    pub fn clock_rate(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    }

    /// Alignment requirement; texture base addresses aligned to ::textureAlign bytes do not need 
    /// an offset applied to texture fetches.
    pub fn texture_alignment(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
    }

    /// Pitch alignment requirement for 2D texture references bound to pitched memory.
    pub fn texture_pitch_alignment(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)
    }

    /// 1 if the device can concurrently copy memory between host and device while executing a 
    /// kernel, or 0 if not;
    pub fn gpu_overlap(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP)
    }

    /// Number of multiprocessors on the device
    pub fn multiprocessor_count(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }

    /// 1 if there is a run time limit for kernels executed on the device, or 0 if not
    pub fn kernel_exec_timeout(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
    }

    /// 1 if the device is integrated with the memory subsystem, or 0 if not
    pub fn integrated(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
    }

    /// 1 if the device can map host memory into the CUDA address space, or 0 if not
    pub fn can_map_host_memory(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)
    }

    /// Compute mode that device is currently in. Available modes are as follows:
    ///
    /// `default` - Device is not restricted and can have multiple CUDA contexts present at a single time.
    /// `prohibited` - Device is prohibited from creating new CUDA contexts
    /// `exclusive_process` - Device can have only one context used by a single process at a time
    pub fn compute_mode(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
    }

    /// 1 if the device supports executing multiple kernels within the same context 
    /// simultaneously, or 0 if not. It is not guaranteed that multiple kernels will be resident
    /// on the device concurrently so this feature should not be relied upon for correctness.
    pub fn concurrent_kernels(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    }

    /// 1 if error correction is enabled on the device, 0 if error correction is disabled or not 
    /// supported by the device
    pub fn ecc_enabled(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
    }

    /// PCI bus identifier of the device
    pub fn pci_bus_id(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    }

    /// PCI device (also known as slot) identifier of the device
    pub fn pci_device_id(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    }

    /// 1 if the device is using a TCC driver. TCC is only available on Tesla hardware running 
    /// Windows Vista or later
    pub fn tcc_driver(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER)
    }

    /// Peak memory clock frequency in kilohertz
    pub fn memory_clock_rate(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    }

    /// Global memory bus width in bits
    pub fn global_memory_bus_width(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    }

    /// Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
    pub fn l2_cache_size(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    }

    /// Maximum resident threads per multiprocessor
    pub fn max_threads_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    }

    /// 1 if the device shares a unified address space with the host, or 0 if not
    pub fn unified_addressing(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
    }

    /// Major compute capability version number
    pub fn compute_capability_major(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    }

    /// Minor compute capability version number
    pub fn compute_capability_minor(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    }

    /// 1 if device supports caching globals in L1 cache, 0 if caching globals in L1 cache is not 
    /// supported by the device
    pub fn global_l1_cache_supported(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)
    }

    /// 1 if device supports caching locals in L1 cache, 0 if caching locals in L1 cache is not 
    /// supported by the device
    pub fn local_l1_cache_supported(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)
    }

    /// Maximum amount of shared memory available to a multiprocessor in bytes; this amount is 
    /// shared by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_shared_memory_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    }

    /// Maximum number of 32-bit registers available to a multiprocessor; this number is shared 
    /// by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_registers_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    }

    /// 1 if device supports allocating managed memory on this system, 0 if allocating managed 
    /// memory is not supported by the device on this system
    pub fn managed_memory(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
    }

    /// 1 if device is on a multi-GPU board, 0 if not
    pub fn multi_gpu_board(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD)
    }

    /// Unique identifier for a group of devices associated with the same board. Devices on the 
    /// same multi-GPU board will share the same identifier
    pub fn multi_gpu_board_group_id(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)
    }

    /// 1 if Link between the device and the host supports native atomic operations
    pub fn host_native_atomic_supported(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)
    }

    /// Ratio of single precision performance (in floating-point operations per second) to double 
    /// precision performance.
    pub fn single_to_double_precision_perf_ratio(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
    }

    /// Device supports coherently accessing "pageable" memory without calling cudaHostRegister on it
    pub fn pageable_memory_access(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)
    }

    /// Device can coherently access managed memory concurrently with the CPU
    pub fn concurrent_managed_access(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)
    }

    /// Device supports Compute Preemption
    pub fn compute_preemption_supported(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
    }

    /// Device can access host registered memory at the same virtual address as the CPU
    pub fn can_use_host_pointer_for_registered_mem(&self) -> Result<i32> {
        self.attribute(sh::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)
    }
}

#[derive(Debug)]
pub struct Memory(sh::CUdeviceptr);

impl Memory {
    // /// Allocates byte_size bytes of linear memory on the device and returns in `*dptr` a pointer 
    // /// to the allocated memory. The allocated memory is suitably aligned for any kind of 
    // /// variable. The memory is not cleared. If bytesize is 0, cuMemAlloc() 
    // /// returns CUDA_ERROR_INVALID_VALUE.
    // pub fn alloc(byte_size: usize) -> Result<Memory> {

    //     unsafe {
    //         let mut dptr = 0u64;

    //         match sys::cuMemAlloc_v2(&mut dptr, byte_size) {
    //             sys::cudaError_enum::CUDA_SUCCESS => 
    //                 Ok(Memory(dptr)),

    //             e @ _ =>
    //                 Err(Error::from(e.into(): ErrorKind)),
    //         }
    //     }
    // }

    /// Frees the memory space pointed to by dptr.
    fn free(&self) -> Result {
        unsafe {
            let ret_value = sh::cuMemFree_v2(self.0);
            return utility::check(ret_value, || {})
        }
    }
}

impl ops::Drop for Memory {

    fn drop(&mut self) {

        self.free().expect("failed to free CUDA memory")
    }
}