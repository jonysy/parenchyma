use cuda_sys;
use error::{Error, ErrorKind, Result};
use std::ffi::CString;
use super::CudaAttribute;

pub struct CudaDeviceHandle(pub(super) i32);

impl CudaDeviceHandle {

    /// Returns an identifier string for the device.
    pub fn name(&self) -> Result<String> {
        unsafe {
            const LENGTH: i32 = 1024;

            let mut name = [0; LENGTH as usize];

            match cuda_sys::cuDeviceGetName(name.as_mut_ptr(), LENGTH, self.0) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => {
                    let c_string = CString::from_raw(name.as_mut_ptr());

                    let st = c_string.into_string().map_err(|e| Error::new(ErrorKind::Unknown, e))?;

                    Ok(st)
                },

                e @ _ => 
                    Err(Error::from(e.into(): ErrorKind))
            }
        }
    }

    // ========= Query CUDA device attributes/properties

    /// Returns information about the device.
    pub fn attribute(&self, attribute: CudaAttribute) -> Result<i32> {
        let mut pi = 0;

        unsafe {
            match cuda_sys::cuDeviceGetAttribute(&mut pi, attribute, self.0) {
                cuda_sys::cudaError_enum::CUDA_SUCCESS => 
                    Ok(pi),

                e @ _ =>
                    Err(Error::from(e.into(): ErrorKind))
            }
        }
    }

    /// Maximum number of threads per block.
    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Maximum x-dimension of a block.
    pub fn max_block_dim_x(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    }

    /// Maximum y-dimension of a block.
    pub fn max_block_dim_y(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    }

    /// Maximum z-dimension of a block.
    pub fn max_block_dim_z(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    }

    /// Maximum x-dimension of a grid.
    pub fn max_grid_dim_x(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    }

    /// Maximum y-dimension of a grid.
    pub fn max_grid_dim_y(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    }

    /// Maximum z-dimension of a grid.
    pub fn max_grid_dim_z(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    }

    /// Maximum amount of shared memory available to a thread block in bytes.
    pub fn max_shared_memory_per_block(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    }

    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes.
    pub fn total_constant_memory(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    }

    /// Warp size in threads.
    pub fn warp_size(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    }

    /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions 
    /// allocated through ::cuMemAllocPitch().
    pub fn max_pitch(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
    }

    /// Maximum 1D texture width.
    pub fn maximum_texture1d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    }

    /// Maximum width for a 1D texture bound to linear memory.
    pub fn maximum_texture1d_linear_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)
    }

    /// Maximum mipmapped 1D texture width.
    pub fn maximum_texture1d_mipmapped_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH)
    }

    /// Maximum 2D texture width.
    pub fn maximum_texture2d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    }

    /// Maximum 2D texture height.
    pub fn maximum_texture2d_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    }

    /// Maximum width for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)
    }

    /// Maximum height for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)
    }

    /// Maximum pitch in bytes for a 2D texture bound to linear memory.
    pub fn maximum_texture2d_linear_pitch(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)
    }

    /// Maximum mipmapped 2D texture width.
    pub fn maximum_texture2d_mipmapped_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH)
    }

    /// Maximum mipmapped 2D texture height.
    pub fn maximum_texture2d_mipmapped_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT)
    }

    /// Maximum 3D texture width.
    pub fn maximum_texture3d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    }

    /// Maximum 3D texture height.
    pub fn maximum_texture3d_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    }

    /// Maximum 3D texture depth.
    pub fn maximum_texture3d_depth(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    }

    /// Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_width_alternate(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE)
    }

    /// Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_height_alternate(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE)
    }

    /// Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported.
    pub fn maximum_texture3d_depth_alternate(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE)
    }

    /// Maximum cubemap texture width or height.
    pub fn maximum_texturecubemap_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)
    }

    /// Maximum 1D layered texture width.
    pub fn maximum_texture1d_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)
    }

    /// Maximum layers in a 1D layered texture.
    pub fn maximum_texture1d_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS)
    }

    /// Maximum 2D layered texture width.
    pub fn maximum_texture2d_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)
    }

    /// Maximum 2D layered texture height.
    pub fn maximum_texture2d_layered_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT)
    }

    /// Maximum layers in a 2D layered texture.
    pub fn maximum_texture2d_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS)
    }

    /// Maximum cubemap layered texture width or height.
    pub fn maximum_texturecubemap_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH)
    }

    /// Maximum layers in a cubemap layered texture.
    pub fn maximum_texturecubemap_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS)
    }

    /// Maximum 1D surface width.
    pub fn maximum_surface1d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)
    }

    /// Maximum 2D surface width.
    pub fn maximum_surface2d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)
    }

    /// Maximum 2D surface height.
    pub fn maximum_surface2d_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)
    }

    /// Maximum 3D surface width.
    pub fn maximum_surface3d_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)
    }

    /// Maximum 3D surface height.
    pub fn maximum_surface3d_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)
    }

    /// Maximum 3D surface depth.
    pub fn maximum_surface3d_depth(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)
    }

    /// Maximum 1D layered surface width.
    pub fn maximum_surface1d_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)
    }

    /// Maximum layers in a 1D layered surface.
    pub fn maximum_surface1d_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS)
    }

    /// Maximum 2D layered surface width.
    pub fn maximum_surface2d_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)
    }

    /// Maximum 2D layered surface height.
    pub fn maximum_surface2d_layered_height(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT)
    }

    /// Maximum layers in a 2D layered surface.
    pub fn maximum_surface2d_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS)
    }

    /// Maximum cubemap surface width.
    pub fn maximum_surfacecubemap_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)
    }

    /// Maximum cubemap layered surface width.
    pub fn maximum_surfacecubemap_layered_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH)
    }

    /// Maximum layers in a cubemap layered surface.
    pub fn maximum_surfacecubemap_layered_layers(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS)
    }

    /// Maximum number of 32-bit registers available to a thread block.
    pub fn max_registers_per_block(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    }

    /// The typical clock frequency in kilohertz.
    pub fn clock_rate(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    }

    /// Alignment requirement; texture base addresses aligned to ::textureAlign bytes do not need 
    /// an offset applied to texture fetches.
    pub fn texture_alignment(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
    }

    /// Pitch alignment requirement for 2D texture references bound to pitched memory.
    pub fn texture_pitch_alignment(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)
    }

    /// 1 if the device can concurrently copy memory between host and device while executing a 
    /// kernel, or 0 if not;
    pub fn gpu_overlap(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP)
    }

    /// Number of multiprocessors on the device
    pub fn multiprocessor_count(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }

    /// 1 if there is a run time limit for kernels executed on the device, or 0 if not
    pub fn kernel_exec_timeout(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
    }

    /// 1 if the device is integrated with the memory subsystem, or 0 if not
    pub fn integrated(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
    }

    /// 1 if the device can map host memory into the CUDA address space, or 0 if not
    pub fn can_map_host_memory(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)
    }

    /// Compute mode that device is currently in. Available modes are as follows:
    ///
    /// `default` - Device is not restricted and can have multiple CUDA contexts present at a single time.
    /// `prohibited` - Device is prohibited from creating new CUDA contexts
    /// `exclusive_process` - Device can have only one context used by a single process at a time
    pub fn compute_mode(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)
    }

    /// 1 if the device supports executing multiple kernels within the same context 
    /// simultaneously, or 0 if not. It is not guaranteed that multiple kernels will be resident
    /// on the device concurrently so this feature should not be relied upon for correctness.
    pub fn concurrent_kernels(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
    }

    /// 1 if error correction is enabled on the device, 0 if error correction is disabled or not 
    /// supported by the device
    pub fn ecc_enabled(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
    }

    /// PCI bus identifier of the device
    pub fn pci_bus_id(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    }

    /// PCI device (also known as slot) identifier of the device
    pub fn pci_device_id(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    }

    /// 1 if the device is using a TCC driver. TCC is only available on Tesla hardware running 
    /// Windows Vista or later
    pub fn tcc_driver(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER)
    }

    /// Peak memory clock frequency in kilohertz
    pub fn memory_clock_rate(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    }

    /// Global memory bus width in bits
    pub fn global_memory_bus_width(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    }

    /// Size of L2 cache in bytes. 0 if the device doesn't have L2 cache
    pub fn l2_cache_size(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    }

    /// Maximum resident threads per multiprocessor
    pub fn max_threads_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    }

    /// 1 if the device shares a unified address space with the host, or 0 if not
    pub fn unified_addressing(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
    }

    /// Major compute capability version number
    pub fn compute_capability_major(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    }

    /// Minor compute capability version number
    pub fn compute_capability_minor(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    }

    /// 1 if device supports caching globals in L1 cache, 0 if caching globals in L1 cache is not 
    /// supported by the device
    pub fn global_l1_cache_supported(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)
    }

    /// 1 if device supports caching locals in L1 cache, 0 if caching locals in L1 cache is not 
    /// supported by the device
    pub fn local_l1_cache_supported(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)
    }

    /// Maximum amount of shared memory available to a multiprocessor in bytes; this amount is 
    /// shared by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_shared_memory_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    }

    /// Maximum number of 32-bit registers available to a multiprocessor; this number is shared 
    /// by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_registers_per_multiprocessor(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    }

    /// 1 if device supports allocating managed memory on this system, 0 if allocating managed 
    /// memory is not supported by the device on this system
    pub fn managed_memory(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)
    }

    /// 1 if device is on a multi-GPU board, 0 if not
    pub fn multi_gpu_board(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD)
    }

    /// Unique identifier for a group of devices associated with the same board. Devices on the 
    /// same multi-GPU board will share the same identifier
    pub fn multi_gpu_board_group_id(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)
    }

    /// 1 if Link between the device and the host supports native atomic operations
    pub fn host_native_atomic_supported(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)
    }

    /// Ratio of single precision performance (in floating-point operations per second) to double 
    /// precision performance.
    pub fn single_to_double_precision_perf_ratio(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
    }

    /// Device supports coherently accessing "pageable" memory without calling cudaHostRegister on it
    pub fn pageable_memory_access(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)
    }

    /// Device can coherently access managed memory concurrently with the CPU
    pub fn concurrent_managed_access(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)
    }

    /// Device supports Compute Preemption
    pub fn compute_preemption_supported(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
    }

    /// Device can access host registered memory at the same virtual address as the CPU
    pub fn can_use_host_pointer_for_registered_mem(&self) -> Result<i32> {
        self.attribute(CudaAttribute::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)
    }
}