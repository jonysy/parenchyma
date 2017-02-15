use std::ops::Deref;
use std::os::raw::c_void;
use std::{mem, ptr};
use super::error::Result;
use super::sys;

#[derive(Clone, Debug)]
pub struct Device(sys::cl_device_id);

impl Device {
    fn info_size(&self, p: u32) -> Result<usize> {

        unsafe {

            let mut size = 0;
            
            result!(sys::clGetDeviceInfo(self.0, p, 0, ptr::null_mut(), &mut size) => Ok(size))
        }
    }

    fn info<F1, F2, T>(&self, p: u32, f1: F1, f2: F2) -> Result<T>
        where F1: Fn(usize) -> T, 
              F2: Fn(&mut T) -> *mut c_void
    {

        unsafe {

            let size = self.info_size(p)?;

            let mut b = f1(size);
            
            result!(sys::clGetDeviceInfo(self.0, p, size, f2(&mut b), ptr::null_mut()) => Ok(b))
        }
    }

    /// The default compute device address space size specified as an unsigned integer value 
    /// in bits. Currently supported values are 32 or 64 bits.
    pub fn address_bits(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_ADDRESS_BITS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Is CL_TRUE if the device is available and CL_FALSE if the device is not available.
    pub fn available(&self) -> Result<bool> {

        let p = sys::CL_DEVICE_AVAILABLE;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res != 0)
    }

    /// Is CL_FALSE if the implementation does not have a compiler available to compile the 
    /// program source. Is CL_TRUE if the compiler is available. This can be CL_FALSE for the 
    /// embedded platform profile only.
    pub fn compiler_available(&self) -> Result<bool> {

        let p = sys::CL_DEVICE_COMPILER_AVAILABLE;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res != 0)
    }

//    /// Describes the OPTIONAL double precision floating-point capability of the OpenCL device. 
//    /// This is a bit-field that describes one or more of the following values:
//    ///
//    /// CL_FP_DENORM - denorms are supported.
//    /// CL_FP_INF_NAN - INF and NaNs are supported.
//    /// CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
//    /// CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
//    /// CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
//    /// CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
//    ///
//    /// The mandated minimum double precision floating-point capability is CL_FP_FMA | 
//    /// CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | 
//    /// CL_FP_DENORM.
//    pub fn double_fp_config(&self) -> Result {
//
//        unimplemented!()
//    }

    /// Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise.
    pub fn endian_little(&self) -> Result<bool> {

        let p = sys::CL_DEVICE_ENDIAN_LITTLE;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res != 0)
    }

    /// Is CL_TRUE if the device implements error correction for the memories, caches, registers 
    /// etc. in the device. Is CL_FALSE if the device does not implement error correction. This can 
    /// be a requirement for certain clients of OpenCL.
    pub fn error_correction_support(&self) -> Result<bool> {

        let p = sys::CL_DEVICE_ERROR_CORRECTION_SUPPORT;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res != 0)
    }

//    /// Describes the execution capabilities of the device. This is a bit-field that describes one 
//    /// or more of the following values:
//    ///
//    /// CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels.
//    ///
//    /// CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels.
//    ///
//    /// The mandated minimum capability is CL_EXEC_KERNEL.
//    pub fn execution_capabilities(&self) -> Result {
//
//        unimplemented!()
//    }

    /// Returns a space separated list of extension names (the extension names themselves do not 
    /// contain any spaces). The list of extension names returned currently can include one or more 
    /// of the following approved extension names:
    ///
    /// cl_khr_fp64
    /// cl_khr_select_fprounding_mode
    /// cl_khr_global_int32_base_atomics
    /// cl_khr_global_int32_extended_atomics
    /// cl_khr_local_int32_base_atomics
    /// cl_khr_local_int32_extended_atomics
    /// cl_khr_int64_base_atomics
    /// cl_khr_int64_extended_atomics
    /// cl_khr_3d_image_writes
    /// cl_khr_byte_addressable_store
    /// cl_khr_fp16
    pub fn extensions(&self) -> Result<Vec<String>> {

        let p = sys::CL_DEVICE_EXTENSIONS;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap()).map(|st| {
            st.split_whitespace().map(|s| s.into()).collect()
        })
    }

    /// Size of global memory cache in bytes.
    pub fn global_mem_cache_size(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

//    /// Type of global memory cache supported. Valid values are: CL_NONE, CL_READ_ONLY_CACHE, 
//    /// and CL_READ_WRITE_CACHE.
//    pub fn global_mem_cache_type(&self) -> Result {
//
//        unimplemented!()
//    }

    /// Size of global memory cache line in bytes.
    pub fn global_mem_cacheline_size(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Size of global memory cache line in bytes.
    pub fn global_mem_size(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_GLOBAL_MEM_SIZE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

//    /// Describes the OPTIONAL half precision floating-point capability of the OpenCL device. This 
//    /// is a bit-field that describes one or more of the following values:
//    ///
//    /// CL_FP_DENORM - denorms are supported.
//    /// CL_FP_INF_NAN - INF and NaNs are supported.
//    /// CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
//    /// CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
//    /// CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
//    /// CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
//    /// The required minimum half precision floating-point capability as implemented by this 
//    /// extension is CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN.
//    pub fn half_fp_config(&self) -> Result {
//
//        unimplemented!()
//    }

    /// Is CL_TRUE if images are supported by the OpenCL device and CL_FALSE otherwise.
    pub fn image_support(&self) -> Result<bool> {

        let p = sys::CL_DEVICE_IMAGE_SUPPORT;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res != 0)
    }

    /// Max height of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image2d_max_height(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_IMAGE2D_MAX_HEIGHT;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Max width of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image2d_max_width(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_IMAGE2D_MAX_WIDTH;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Max depth of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_depth(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_IMAGE3D_MAX_DEPTH;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Max height of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_height(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_IMAGE3D_MAX_HEIGHT;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Max width of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_width(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_IMAGE3D_MAX_WIDTH;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Size of local memory arena in bytes. The minimum value is 16 KB.
    pub fn local_mem_size(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_LOCAL_MEM_SIZE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

//    /// Type of local memory supported. This can be set to CL_LOCAL implying dedicated local memory 
//    /// storage such as SRAM, or CL_GLOBAL.
//    pub fn local_mem_type(&self) -> Result {
//
//        unimplemented!()
//    }

    /// Size of local memory arena in bytes. The minimum value is 16 KB.
    pub fn max_clock_frequency(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_CLOCK_FREQUENCY;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// The number of parallel compute cores on the OpenCL device. The minimum value is 1.
    pub fn max_compute_units(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_COMPUTE_UNITS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Max number of arguments declared with the __constant qualifier in a kernel. The minimum 
    /// value is 8.
    pub fn max_constant_args(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_CONSTANT_ARGS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Max size in bytes of a constant buffer allocation. The minimum value is 64 KB.
    pub fn max_constant_buffer_size(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

    /// Max size of memory object allocation in bytes. The minimum value is 
    /// max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024)
    pub fn max_mem_alloc_size(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_MAX_MEM_ALLOC_SIZE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

    /// Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 256.
    pub fn max_parameter_size(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_MAX_PARAMETER_SIZE;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Max number of simultaneous image objects that can be read by a kernel. The minimum value 
    /// is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_read_image_args(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_READ_IMAGE_ARGS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Maximum number of samplers that can be used in a kernel. The minimum value is 16 
    /// if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_samplers(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_SAMPLERS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Maximum number of work-items in a work-group executing a kernel using the data parallel 
    /// execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 1.
    pub fn max_work_group_size(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_MAX_WORK_GROUP_SIZE;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

    /// Maximum dimensions that specify the global and local work-item IDs used by the data 
    /// parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 3.
    pub fn max_work_item_dimensions(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Maximum number of work-items that can be specified in each dimension of the work-group 
    /// to clEnqueueNDRangeKernel.
    ///
    /// Returns n size_t entries, where n is the value returned by the query 
    /// for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. The minimum value is (1, 1, 1).
    pub fn max_work_item_sizes(&self) -> Result<Vec<usize>> {

        let p = sys::CL_DEVICE_MAX_WORK_ITEM_SIZES;
        let ve = |size| vec![1usize; size / mem::size_of::<usize>()];

        let res = self.info(p, ve, |b| b.as_mut_ptr() as _)?;

        Ok(res)
    }

    /// Max number of simultaneous image objects that can be written to by a kernel. The minimum 
    /// value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_write_image_args(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Describes the alignment in bits of the base address of any allocated memory object.
    pub fn mem_base_addr_align(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MEM_BASE_ADDR_ALIGN;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// The smallest alignment in bytes which can be used for any data type.
    pub fn min_data_type_align_size(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// Device name string.
    pub fn name(&self) -> Result<String> {

        let p = sys::CL_DEVICE_NAME;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap())
    }

//    /// The platform associated with this device.
//    pub fn platform(&self) -> Result {
//
//        let _ = sys::CL_DEVICE_PLATFORM;
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_char(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
//        
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_short(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
//        
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_int(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
//        
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_long(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
//        
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_float(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
//        
//        unimplemented!()
//    }

//    /// Preferred native vector width size for built-in scalar types that can be put into 
//    /// vectors. The vector width is defined as the number of scalar elements that can be stored 
//    /// in the vector.
//    ///
//    /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 
//    /// must return 0.
//    pub fn preferred_vector_width_double(&self) -> Result<u32> {
//
//        let _ = sys::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
//        
//        unimplemented!()
//    }

    /// OpenCL profile string. Returns the profile name supported by the device (see note). The 
    /// profile name returned can be one of the following strings:
    ///
    /// FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as 
    /// part of the core specification and does not require any extensions to be supported).
    ///
    /// EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
    pub fn profile(&self) -> Result<String> {

        let p = sys::CL_DEVICE_PROFILE;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// The smallest alignment in bytes which can be used for any data type.
    pub fn profiling_timer_resolution(&self) -> Result<usize> {

        let p = sys::CL_DEVICE_PROFILING_TIMER_RESOLUTION;
        let res = self.info(p, |_| 0usize, |b| b as *mut usize as _)?;

        Ok(res)
    }

//    /// Describes the command-queue properties supported by the device. This is a bit-field that 
//    /// describes one or more of the following values:
//    ///
//    /// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
//    ///
//    /// CL_QUEUE_PROFILING_ENABLE
//    ///
//    /// These properties are described in the table for clCreateCommandQueue. The mandated minimum 
//    /// capability is CL_QUEUE_PROFILING_ENABLE.
//    pub fn queue_properties(&self) -> Result {
//
//        unimplemented!()
//    }

//    /// Describes single precision floating-point capability of the device. This is a bit-field 
//    /// that describes one or more of the following values:
//    ///
//    /// CL_FP_DENORM - denorms are supported
//    ///
//    /// CL_FP_INF_NAN - INF and quiet NaNs are supported
//    ///
//    /// CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported
//    ///
//    /// CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported
//    ///
//    /// CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported
//    ///
//    /// CL_FP_FMA - IEEE754-2008 fused multiply-add is supported
//    ///
//    /// The mandated minimum floating-point capability is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN.
//    pub fn single_fp_config(&self) -> Result {
//
//        unimplemented!()
//    }

    /// The OpenCL device type. Currently supported values are one of or a combination 
    /// of: CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, 
    /// or CL_DEVICE_TYPE_DEFAULT.
    pub fn type_(&self) -> Result<u64> {

        let p = sys::CL_DEVICE_TYPE;
        let res = self.info(p, |_| 0u64, |b| b as *mut u64 as _)?;

        Ok(res)
    }

    /// Vendor name string.
    pub fn vendor(&self) -> Result<String> {

        let p = sys::CL_DEVICE_VENDOR;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// A unique device vendor identifier. An example of a unique device identifier could be 
    /// the PCIe ID.
    pub fn vendor_id(&self) -> Result<u32> {

        let p = sys::CL_DEVICE_VENDOR_ID;
        let res = self.info(p, |_| 0u32, |b| b as *mut u32 as _)?;

        Ok(res)
    }

    /// OpenCL version string. Returns the OpenCL version supported by the device. This version 
    /// string has the following format:
    ///
    /// OpenCL<space><major_version.minor_version><space><vendor-specific information>
    ///
    /// The major_version.minor_version value returned will be 1.0.
    pub fn version(&self) -> Result<String> {

        let p = sys::CL_DEVICE_VERSION;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// OpenCL software driver version string in the form major_number.minor_number.
    pub fn driver_version(&self) -> Result<String> {

        let p = sys::CL_DRIVER_VERSION;
        let res = self.info(p, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);

        res.map(|b| String::from_utf8(b).unwrap())
    }
}

impl Deref for Device {
    
    type Target = sys::cl_device_id;
    
    fn deref(&self) -> &Self::Target {
        
        &self.0
    }
}

impl From<sys::cl_device_id> for Device {
    
    fn from(cl_device_id: sys::cl_device_id) -> Self {
        
        Device(cl_device_id)
    }
}

impl Into<sys::cl_device_id> for Device {
    
    fn into(self) -> sys::cl_device_id {
        self.0
    }
}

impl PartialEq<Device> for Device {

    fn eq(&self, other: &Device) -> bool {

        self.0 == other.0
    }
}