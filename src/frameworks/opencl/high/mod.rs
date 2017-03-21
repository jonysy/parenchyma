//! Wrapper module for OpenCL
#![allow(missing_docs, unused_qualifications)]

pub use self::error::{Error, ErrorKind, Result};
pub use self::functions::{nplatforms, platforms};

mod error;
mod functions;
mod utility;

use std::{clone, cmp, ffi, mem, ops, ptr};
use std::os::raw::c_void;
use super::foreign;

#[derive(Debug)]
pub struct Buffer(foreign::cl_mem);

impl Buffer {
    /// Increments the memory object reference count.
    pub fn retain(&self) -> Result {
        unsafe {
            let ret_value = foreign::clRetainMemObject(self.0);

            return utility::check(ret_value, || {});
        }
    }

    /// Decrements the memory object reference count.
    pub fn release(&self) -> Result {
        unsafe {
            let ret_value = foreign::clReleaseMemObject(self.0);
            
            return utility::check(ret_value, || {});
        }
    }
}

impl clone::Clone for Buffer {
    fn clone(&self) -> Self {
        self.retain().expect("unable to retain `Buffer`");
        Buffer(self.0)
    }
}

impl ops::Drop for Buffer {
    fn drop(&mut self) {
        self.release().expect("unable to release `Buffer`")
    }
}

#[derive(Debug)]
pub struct Context(foreign::cl_context);

impl Context {
    pub fn ptr(&self) -> &foreign::cl_context {
        &self.0
    }

    /// Creates an OpenCL context.
    ///
    /// An OpenCL context is created with one or more devices. Contexts are used by the OpenCL 
    /// runtime for managing objects such as command-queues, memory, program and kernel objects 
    /// and for executing kernels on one or more devices specified in the context.
    pub fn new(devices: &[Device]) -> Result<Self> {
        unsafe {
            // An appropriate error code. If errcode_ret is NULL, no error code is returned.
            let mut errcode_ret: i32 = 0;

            // Specifies a list of context property names and their corresponding values. Each 
            // property name is immediately followed by the corresponding desired value. The list 
            // is terminated with 0. properties can be NULL in which case the platform that is 
            // selected is implementation-defined. The list of supported properties is described 
            // in the table below.
            //
            // ```
            // cl_context_properties enum   | Property value | Description
            // -----------------------------------------------------------
            // CL_CONTEXT_PLATFORM          | cl_platform_id | Specifies the platform to use.
            // ```
            //
            // TODO
            let properties: *const isize = ptr::null();

            // The number of devices specified in the devices argument.
            let number_of_devices = devices.len() as u32;
            // A pointer to a list of unique devices returned by clGetDeviceIDs for a platform.
            let raw_devices: Vec<*mut c_void> = devices.iter().map(|d| d.0).collect();
            let raw_devices_ptr = raw_devices.as_ptr();

            // A callback function that can be registered by the application. This callback function 
            // will be used by the OpenCL implementation to report information on errors that occur 
            // in this context. This callback function may be called asynchronously by the OpenCL 
            // implementation. It is the application's responsibility to ensure that the callback 
            // function is thread-safe. If pfn_notify is NULL, no callback function is registered. 
            // The parameters to this callback function are:
            //
            // `errinfo` is a pointer to an error string.
            //
            // `private_info` and `cb` represent a pointer to binary data that is returned by 
            // the OpenCL implementation that can be used to log additional information helpful in 
            // debugging the error.
            //
            // `user_data` is a pointer to user supplied data.
            //
            // TODO
            let pfn_notify: extern fn(*const i8, *const c_void, usize, *mut c_void) 
                = mem::transmute(ptr::null::<fn()>());

            // Passed as the `user_data` argument when pfn_notify is called. user_data can be NULL.
            //
            // TODO
            let user_data: *mut c_void = ptr::null_mut();

            let cl_context = foreign::clCreateContext(
                properties, 
                number_of_devices, 
                raw_devices_ptr, 
                pfn_notify, 
                user_data, 
                &mut errcode_ret
            );

            let ret_value = foreign::CLStatus::new(errcode_ret)
                .expect("failed to convert `i32` to `CLStatus`");

            return utility::check(ret_value, || Context(cl_context));
        }
    }

    /// Creates a buffer object with a size of `size`.
    ///
    /// # Arguments
    ///
    /// * `flag` - A bit-field that is used to specify allocation and usage information such as the 
    /// memory arena that should be used to allocate the buffer object and how it will be used.
    /// https://streamcomputing.eu/blog/2013-02-03
    ///
    /// * `size` - The size in bytes of the buffer memory object to be allocated.
    ///
    /// * `host_pointer` - A pointer to the buffer data that may already be allocated by the 
    /// application. The size of the buffer that host_ptr points to must be greater than or equal 
    /// to the size bytes.
    pub fn create_buffer<F, H>(&self, f: F, size: usize, h: H) -> Result<Buffer> 
        where F: Into<Option<foreign::cl_bitfield>>,
              H: Into<Option<*mut c_void>>,
    {
        unsafe {
            let mut errcode_ret: i32 = 0;
            let flags = f.into().unwrap_or(foreign::CL_MEM_READ_WRITE);
            let host_pointer = h.into().unwrap_or(ptr::null_mut());

            let mem = foreign::clCreateBuffer(self.0, flags, size, host_pointer, &mut errcode_ret);

            let ret_value = foreign::CLStatus::new(errcode_ret)
                .expect("failed to convert `i32` to `CLStatus`");

            return utility::check(ret_value, || Buffer(mem));
        }
    }

    /// Creates a program object for a context, and loads the source code specified by the text 
    /// strings in the strings array into the program object.
    pub fn create_program_with_source<I>(&self, strings: &[I]) -> Result<Program> 
        where I: AsRef<str> {

        unsafe {
            let mut errcode = 0i32;

            let n = strings.len() as u32;
            let lengths: Vec<usize> = strings.iter().map(|s| s.as_ref().len() as usize).collect();
            let lens_ptr = lengths.as_ptr();
            // https://doc.rust-lang.org/std/ffi/struct.CString.html#method.as_ptr
            let cstrings: Vec<ffi::CString> = strings.iter().map(|s| {
                ffi::CString::new(s.as_ref()).unwrap()
            }).collect();

            let ptrs: Vec<*const i8> = cstrings.iter().map(|s| s.as_ptr()).collect();
            let ptr = ptrs.as_ptr();

            let cl_program = foreign::clCreateProgramWithSource(self.0, n, ptr, lens_ptr, &mut errcode);

            let ret_value = foreign::CLStatus::new(errcode).expect("failed to convert i32 to CLStatus");

            return utility::check(ret_value, || Program(cl_program));
        }
    }

    /// Increment the context reference count.
    fn retain(&self) -> Result {
        unsafe {
            let ret_value = foreign::clRetainContext(self.0);
            return utility::check(ret_value, || {});
        }
    }

    /// Decrement the context reference count.
    fn release(&self) -> Result {
        unsafe {
            let ret_value = foreign::clReleaseContext(self.0);
            return utility::check(ret_value, || {});
        }
    }
}

impl clone::Clone for Context {

    fn clone(&self) -> Self {
        self.retain().expect("unable to retain `Context`");
        Context(self.0)
    }
}

impl cmp::Eq for Context { }

impl cmp::PartialEq<Context> for Context {

    fn eq(&self, other: &Context) -> bool {

        self.0 == other.0
    }
}

impl ops::Drop for Context {

    fn drop(&mut self) {
        self.release().expect("unable to release `Context`");
    }
}

/// Newtype with an internal type of `cl_device_id`.
#[derive(Clone, Debug)]
pub struct Device(foreign::cl_device_id);

impl Device {
    pub fn ptr(&self) -> &foreign::cl_device_id {
        &self.0
    }

    /// The default compute device address space size specified as an unsigned integer value 
    /// in bits. Currently supported values are 32 or 64 bits.
    pub fn address_bits(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_ADDRESS_BITS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Is CL_TRUE if the device is available and CL_FALSE if the device is not available.
    pub fn available(&self) -> Result<bool> {
        let parameter = foreign::CL_DEVICE_AVAILABLE;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res != 0)
    }

    /// Is CL_FALSE if the implementation does not have a compiler available to compile the 
    /// program source. Is CL_TRUE if the compiler is available. This can be CL_FALSE for the 
    /// embedded platform profile only.
    pub fn compiler_available(&self) -> Result<bool> {
        let parameter = foreign::CL_DEVICE_COMPILER_AVAILABLE;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
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
        let parameter = foreign::CL_DEVICE_ENDIAN_LITTLE;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res != 0)
    }

    /// Is CL_TRUE if the device implements error correction for the memories, caches, registers 
    /// etc. in the device. Is CL_FALSE if the device does not implement error correction. This can 
    /// be a requirement for certain clients of OpenCL.
    pub fn error_correction_support(&self) -> Result<bool> {
        let parameter = foreign::CL_DEVICE_ERROR_CORRECTION_SUPPORT;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
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
        let parameter = foreign::CL_DEVICE_EXTENSIONS;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).expect("UTF8 string")).map(|st| {
            st.split_whitespace().map(|s| s.into()).collect()
        })
    }

    /// Size of global memory cache in bytes.
    pub fn global_mem_cache_size(&self) -> Result<u64> {
        let parameter = foreign::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
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
        let parameter = foreign::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Size of global memory cache line in bytes.
    pub fn global_mem_size(&self) -> Result<u64> {
        let parameter = foreign::CL_DEVICE_GLOBAL_MEM_SIZE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
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
        let parameter = foreign::CL_DEVICE_IMAGE_SUPPORT;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res != 0)
    }

    /// Max height of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image2d_max_height(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_IMAGE2D_MAX_HEIGHT;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Max width of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image2d_max_width(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_IMAGE2D_MAX_WIDTH;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Max depth of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_depth(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_IMAGE3D_MAX_DEPTH;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Max height of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_height(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_IMAGE3D_MAX_HEIGHT;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Max width of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT 
    /// is CL_TRUE.
    pub fn image3d_max_width(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_IMAGE3D_MAX_WIDTH;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Size of local memory arena in bytes. The minimum value is 16 KB.
    pub fn local_mem_size(&self) -> Result<u64> {
        let parameter = foreign::CL_DEVICE_LOCAL_MEM_SIZE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
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
        let parameter = foreign::CL_DEVICE_MAX_CLOCK_FREQUENCY;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// The number of parallel compute cores on the OpenCL device. The minimum value is 1.
    pub fn max_compute_units(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_COMPUTE_UNITS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Max number of arguments declared with the __constant qualifier in a kernel. The minimum 
    /// value is 8.
    pub fn max_constant_args(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_CONSTANT_ARGS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Max size in bytes of a constant buffer allocation. The minimum value is 64 KB.
    pub fn max_constant_buffer_size(&self) -> Result<u64> {
        let parameter = foreign::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
        Ok(res)
    }

    /// Max size of memory object allocation in bytes. The minimum value is 
    /// max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024)
    pub fn max_mem_alloc_size(&self) -> Result<u64> {
        let parameter = foreign::CL_DEVICE_MAX_MEM_ALLOC_SIZE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
        Ok(res)
    }

    /// Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 256.
    pub fn max_parameter_size(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_MAX_PARAMETER_SIZE;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Max number of simultaneous image objects that can be read by a kernel. The minimum value 
    /// is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_read_image_args(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_READ_IMAGE_ARGS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Maximum number of samplers that can be used in a kernel. The minimum value is 16 
    /// if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_samplers(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_SAMPLERS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Maximum number of work-items in a work-group executing a kernel using the data parallel 
    /// execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 1.
    pub fn max_work_group_size(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_MAX_WORK_GROUP_SIZE;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
        Ok(res)
    }

    /// Maximum dimensions that specify the global and local work-item IDs used by the data 
    /// parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 3.
    pub fn max_work_item_dimensions(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Maximum number of work-items that can be specified in each dimension of the work-group 
    /// to clEnqueueNDRangeKernel.
    ///
    /// Returns n size_t entries, where n is the value returned by the query 
    /// for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. The minimum value is (1, 1, 1).
    pub fn max_work_item_sizes(&self) -> Result<Vec<usize>> {
        let parameter = foreign::CL_DEVICE_MAX_WORK_ITEM_SIZES;
        let ve = |size| vec![1usize; size / mem::size_of::<usize>()];
        let res = self.info(parameter, ve, |b| b.as_mut_ptr() as _)?;

        Ok(res)
    }

    /// Max number of simultaneous image objects that can be written to by a kernel. The minimum 
    /// value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
    pub fn max_write_image_args(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Describes the alignment in bits of the base address of any allocated memory object.
    pub fn mem_base_addr_align(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MEM_BASE_ADDR_ALIGN;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// The smallest alignment in bytes which can be used for any data type.
    pub fn min_data_type_align_size(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// Device name string.
    pub fn name(&self) -> Result<String> {
        let parameter = foreign::CL_DEVICE_NAME;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).unwrap())
    }

//    /// The platform associated with this device.
//    pub fn platform(&self) -> Result {
//
//        let _ = foreign::CL_DEVICE_PLATFORM;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
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
//        let _ = foreign::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
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
        let parameter = foreign::CL_DEVICE_PROFILE;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// The smallest alignment in bytes which can be used for any data type.
    pub fn profiling_timer_resolution(&self) -> Result<usize> {
        let parameter = foreign::CL_DEVICE_PROFILING_TIMER_RESOLUTION;
        let res = self.info(parameter, |_| 0usize, |b| b as *mut usize as _)?;
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
        let parameter = foreign::CL_DEVICE_TYPE;
        let res = self.info(parameter, |_| 0u64, |b| b as *mut u64 as _)?;
        Ok(res)
    }

    /// Vendor name string.
    pub fn vendor(&self) -> Result<String> {
        let parameter = foreign::CL_DEVICE_VENDOR;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// A unique device vendor identifier. An example of a unique device identifier could be 
    /// the PCIe ID.
    pub fn vendor_id(&self) -> Result<u32> {
        let parameter = foreign::CL_DEVICE_VENDOR_ID;
        let res = self.info(parameter, |_| 0u32, |b| b as *mut u32 as _)?;
        Ok(res)
    }

    /// OpenCL version string. Returns the OpenCL version supported by the device. This version 
    /// string has the following format:
    ///
    /// OpenCL<space><major_version.minor_version><space><vendor-specific information>
    ///
    /// The major_version.minor_version value returned will be 1.0.
    pub fn version(&self) -> Result<String> {
        let parameter = foreign::CL_DEVICE_VERSION;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// OpenCL software driver version string in the form major_number.minor_number.
    pub fn driver_version(&self) -> Result<String> {
        let parameter = foreign::CL_DRIVER_VERSION;
        let res = self.info(parameter, |size| vec![0u8; size], |b| b.as_mut_ptr() as _);
        res.map(|b| String::from_utf8(b).unwrap())
    }

    /// Returns the size of `parameter`.
    fn info_size(&self, parameter: u32) -> Result<usize> {
        unsafe {
            let mut size = 0;
            let ret_value = foreign::clGetDeviceInfo(self.0, parameter, 0, ptr::null_mut(), &mut size);

            return utility::check(ret_value, || size);
        }
    }

    fn info<F1, F2, T>(&self, p: u32, f1: F1, f2: F2) -> Result<T>
        where F1: Fn(usize) -> T, 
              F2: Fn(&mut T) -> *mut c_void {
        unsafe {

            let size = self.info_size(p)?;
            let mut ret = f1(size);
            
            let ret_value = foreign::clGetDeviceInfo(self.0, p, size, f2(&mut ret), ptr::null_mut());

            return utility::check(ret_value, || ret);
        }
    }
}

impl cmp::Eq for Device { }

impl cmp::PartialEq<Device> for Device {

    fn eq(&self, other: &Device) -> bool {

        self.0 == other.0
    }
}

// TODO use newtype: https://github.com/rust-lang/rust/issues/32146
pub type Event = foreign::cl_event;

#[derive(Debug)]
pub struct Kernel(foreign::cl_kernel);

pub trait KernelArg {
    fn size() -> usize;
    fn pointer(&self) -> *mut c_void;
}

impl KernelArg for Buffer {
    fn size() -> usize { mem::size_of::<foreign::cl_mem>() }
    fn pointer(&self) -> foreign::cl_mem { unsafe { mem::transmute(self) } }
}

impl KernelArg for i32 {
    fn size() -> usize { mem::size_of::<i32>() }
    fn pointer(&self) -> foreign::cl_mem { unsafe { self as *const i32 as foreign::cl_mem } }
}

impl Kernel {

    /// Used to set the argument value for a specific argument of a kernel.
    ///
    /// # Arguments
    ///
    /// * `index` - The argument index. Arguments to the kernel are referred by indices that go 
    /// from 0 for the leftmost argument to n - 1, where n is the total number of arguments 
    /// declared by a kernel.
    ///
    /// * `size` - Specifies the size of the argument value. If the argument is a memory object, the 
    /// size is the size of the buffer or image object type. For arguments declared with 
    /// the __local qualifier, the size specified will be the size in bytes of the buffer that 
    /// must be allocated for the __local argument. If the argument is of 
    /// type sampler_t, the arg_size value must be equal to sizeof(cl_sampler). For all other 
    /// arguments, the size will be the size of argument type.
    ///
    /// * `value` - A pointer to data that should be used as the argument value for argument 
    /// specified by arg_index. The argument data pointed to by arg_value is copied and 
    /// the arg_value pointer can therefore be reused by the application after clSetKernelArg 
    /// returns. The argument value specified is the value used by all API calls that enqueue 
    /// kernel (clEnqueueNDRangeKernel and clEnqueueTask) until the argument value is changed by 
    /// a call to clSetKernelArg for kernel.
    ///
    /// If the argument is a memory object (buffer or image), the arg_value entry will be a pointer 
    /// to the appropriate buffer or image object. The memory object must be created with the 
    /// context associated with the kernel object. A NULL value can also be specified if the 
    /// argument is a buffer object in which case a NULL value will be used as the value for the 
    /// argument declared as a pointer to __global or __constant memory in the kernel. If the 
    /// argument is declared with the __local qualifier, the arg_value entry must be NULL. If the 
    /// argument is of type sampler_t, the arg_value entry must be a pointer to the sampler 
    /// object. For all other kernel arguments, the arg_value entry must be a pointer to the actual 
    /// data to be used as argument value.
    pub fn set_arg<A>(&self, position: u32, buf: &A) -> Result where A: KernelArg {
        unsafe {
            let size = A::size();
            let ptr = buf.pointer();
            let ret_value = foreign::clSetKernelArg(self.0, position, size, ptr);
            return utility::check(ret_value, || {});
        }
    }

    /// Increment the kernel reference count.
    fn retain(&self) -> Result {
        unsafe {
            let ret_value = foreign::clRetainKernel(self.0);
            return utility::check(ret_value, || {});
        }
    }

    /// Decrement the kernel reference count.
    fn release(&self) -> Result {
        unsafe {
            let ret_value = foreign::clReleaseKernel(self.0);
            return utility::check(ret_value, || {});
        }
    }
}

impl clone::Clone for Kernel {

    fn clone(&self) -> Self {
        self.retain().expect("unable to retain `Kernel`");
        Kernel(self.0)
    }
}

impl ops::Drop for Kernel {

    fn drop(&mut self) {
        self.release().expect("unable to release `Kernel`");
    }
}

/// Newtype with an internal type of `cl_platform_id`.
#[derive(Clone, Debug)]
pub struct Platform(foreign::cl_platform_id);

impl Platform {

    /// OpenCL profile string. Returns the profile name supported by the implementation. The 
    /// profile name returned can be one of the following strings:
    ///
    /// * `FULL_PROFILE` - if the implementation supports the OpenCL specification (functionality 
    /// defined as part of the core specification and does not require any extensions to be supported).
    ///
    /// * `EMBEDDED_PROFILE` - if the implementation supports the OpenCL embedded profile. The 
    /// embedded profile is defined to be a subset for each version of OpenCL.
    pub fn profile(&self) -> Result<String> {
        self.info(foreign::CL_PLATFORM_PROFILE)
    }

    /// Returns the platform name.
    pub fn name(&self) -> Result<String> {
        self.info(foreign::CL_PLATFORM_NAME)
    }

    /// Returns the platform vendor.
    pub fn vendor(&self) -> Result<String> {
        self.info(foreign::CL_PLATFORM_VENDOR)
    }

    /// Returns a space-separated list of extension names (the extension names themselves do 
    /// not contain any spaces) supported by the platform. Extensions defined here must be 
    /// supported by all devices associated with this platform.
    pub fn extensions(&self) -> Result<Vec<String>> {

        let closure = |st: String| {
            st.split_whitespace().map(|s| s.into()).collect()
        };

        self.info(foreign::CL_PLATFORM_EXTENSIONS).map(closure)
    }

    pub fn ndevices_by_type(&self, t: u64) -> Result<u32> {
        unsafe {
            let mut ndevices = 0;
            let ret_value = foreign::clGetDeviceIDs(self.0, t, 0, ptr::null_mut(), &mut ndevices);
            return utility::check(ret_value, || ndevices);
        }
    }

    pub fn devices_by_type(&self, t: u64) -> Result<Vec<Device>> {
        unsafe {
            let ndevices = self.ndevices_by_type(t)?;
            let mut vec_id = vec![0 as foreign::cl_device_id; ndevices as usize];
            let n = ptr::null_mut();

            let ret_value = foreign::clGetDeviceIDs(self.0, t, ndevices, vec_id.as_mut_ptr(), n);
            utility::check(ret_value, || vec_id.iter().map(|&id| Device(id)).collect())
        }
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        self.devices_by_type(foreign::CL_DEVICE_TYPE_ALL)
    }

    /// Returns the size of `parameter`.
    fn info_size(&self, parameter: u32) -> Result<usize> {
        unsafe {
            let mut size = 0;
            let ret_value = foreign::clGetPlatformInfo(self.0, parameter, 0, ptr::null_mut(), &mut size);

            return utility::check(ret_value, || size);
        }
    }

    fn info(&self, parameter: u32) -> Result<String> {
        unsafe {
            let size = self.info_size(parameter)?;
            let mut bytes = vec![0u8; size];
            let ret_value = foreign::clGetPlatformInfo(
                self.0, 
                parameter, 
                size, 
                bytes.as_mut_ptr() as *mut c_void, 
                ptr::null_mut()
            );

            return utility::check(ret_value, || String::from_utf8(bytes).expect("UTF8 string"));
        }
    }
}

#[derive(Debug)]
pub struct Program(foreign::cl_program);

impl Program {

    /// Builds (compiles and links) a program executable from the program source or binary.
    ///
    /// # Arguments
    ///
    /// * `devices` - The program executable is built for devices specified in this list for 
    /// which a source or binary has been loaded.
    ///
    /// * `options` - A pointer to a string that describes the build options to be used for 
    /// building the program executable.
    pub fn build<T>(&self, devices: &[Device], opt: T) -> Result where T: Into<Option<String>> {
        unsafe {
            let num_devices = devices.len() as u32;
            let raw_devices: Vec<foreign::cl_device_id> = devices.iter().map(|d| d.0).collect();
            let raw_devices_ptr = raw_devices.as_ptr();

            let options = match opt.into() {
                Some(..) => unimplemented!(), // TODO

                _ => ptr::null()
            };

            let pfn_notify = mem::transmute(ptr::null::<fn()>());
            let user_data = ptr::null_mut();

            let ret_value = foreign::clBuildProgram(
                self.0, 
                num_devices, 
                raw_devices_ptr,
                options, 
                pfn_notify, 
                user_data
            );

            return utility::check(ret_value, || {});
        }
    }

    /// Creates a kernel object.
    pub fn create_kernel<T>(&self, name: T) -> Result<Kernel> where T: AsRef<str> {
        unsafe {
            let mut errcode = 0i32;
            let cstring = ffi::CString::new(name.as_ref()).unwrap();
            let ptr = cstring.as_ptr();
            let cl_kernel = foreign::clCreateKernel(self.0, ptr, &mut errcode);
            let ret_value = foreign::CLStatus::new(errcode).expect("failed to convert i32 to CLStatus");

            return utility::check(ret_value, || Kernel(cl_kernel));
        }
    }

    /// Increment the context reference count.
    fn retain(&self) -> Result {
        unsafe {
            let ret_value = foreign::clRetainProgram(self.0);
            return utility::check(ret_value, || {});
        }
    }

    /// Decrement the context reference count.
    fn release(&self) -> Result {
        unsafe {
            let ret_value = foreign::clReleaseProgram(self.0);
            return utility::check(ret_value, || {});
        }
    }
}

impl clone::Clone for Program {

    fn clone(&self) -> Self {
        self.retain().expect("unable to retain `Program`");
        Program(self.0)
    }
}

impl ops::Drop for Program {

    fn drop(&mut self) {
        self.release().expect("unable to release `Program`");
    }
}

#[derive(Debug)]
pub struct Queue(foreign::cl_command_queue);

impl Queue {

    /// Create a command-queue on a specific device.
    pub fn new(context: &Context, device: &Device, properties: u64) -> Result<Self> {
        unsafe {
            let mut errcode_ret: i32 = 0;
            let cl_command_queue = foreign::clCreateCommandQueue(context.0, device.0, 
                properties, &mut errcode_ret);
            let ret_value = foreign::CLStatus::new(errcode_ret).expect("failed to convert i32 to CLStatus");

            return utility::check(ret_value, || Queue(cl_command_queue));
        }
    }

    /// Enqueue commands to write to a buffer object from host memory.
    ///
    /// # Arguments
    ///
    /// `command_queue`   - The command-queue in which the write command will be queued. command_queue 
    ///                     and buffer must be created with the same OpenCL context.
    /// `buffer`          - Refers to a valid buffer object.
    /// `blocking_write`  - Indicates if the write operations are blocking or nonblocking.
    ///                     If blocking_write is CL_TRUE, the OpenCL implementation copies the data 
    ///                     referred to by ptr and enqueues the write operation in the command-queue. 
    ///                     The memory pointed to by ptr can be reused by the application after 
    ///                     the clEnqueueWriteBuffer call returns.
    ///                     If blocking_write is CL_FALSE, the OpenCL implementation will use ptr to 
    ///                     perform a nonblocking write. As the write is non-blocking the implementation 
    ///                     can return immediately. The memory pointed to by ptr cannot be reused by 
    ///                     the application after the call returns. The event argument returns an event 
    ///                     object which can be used to query the execution status of the write command. 
    ///                     When the write command has completed, the memory pointed to by ptr can then 
    ///                     be reused by the application.
    /// `offset`          - The offset in bytes in the buffer object to write to.
    /// `cb`              - The size in bytes of data being written.
    /// `ptr`             - The pointer to buffer in host memory where data is to be written from.
    /// `event_wait_list` - event_wait_list and num_events_in_wait_list specify events that need to 
    ///                     complete before this particular command can be executed. If event_wait_list 
    ///                     is NULL, then this particular command does not wait on any event to complete. 
    ///                     If event_wait_list is NULL, num_events_in_wait_list must be 0. If 
    ///                     event_wait_list is not NULL, the list of events pointed to by 
    ///                     event_wait_list must be valid and num_events_in_wait_list must be greater 
    ///                     than 0. The events specified in event_wait_list act as synchronization points. 
    ///                     The context associated with events in event_wait_list and command_queue 
    ///                     must be the same.
    ///
    /// # Returns
    ///
    /// Returns an event object that identifies this particular write command and can be used to query 
    /// or queue a wait for this particular command to complete. event can be NULL in which case it 
    /// will not be possible for the application to query the status of this command or queue a wait 
    /// for this command to complete. 
    pub fn enqueue_write_buffer(
        &self, 
        buffer:          &Buffer, 
        blocking_write:  bool, 
        offset:          usize,
        cb:              usize, 
        ptr:             *const c_void,
        event_wait_list: &[Event]) 
        -> Result<Event> {

        unsafe {

            let num_events_in_wait_list = event_wait_list.len() as u32;
            let events = 
                if num_events_in_wait_list > 0 { event_wait_list.as_ptr() } else { ptr::null() };

            let mut new_event = 0 as foreign::cl_event;

            let blocking_write_u32 = if blocking_write { 1 } else { 0 };

            let ret_value = foreign::clEnqueueWriteBuffer(
                self.0, 
                buffer.0, 
                blocking_write_u32,
                offset,
                cb,
                ptr,
                num_events_in_wait_list,
                events,
                &mut new_event
            );

            return utility::check(ret_value, || new_event);
        }
    }

    /// Enqueue commands to read from a buffer object to host memory.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Refers to a valid buffer object.
    /// 
    /// * `blocking_read` - Indicates if the read operations are blocking or non-blocking. 
    /// If blocking_read is CL_TRUE i.e. the read command is blocking, clEnqueueReadBuffer does not 
    /// return until the buffer data has been read and copied into memory pointed to by ptr.
    ///
    /// If blocking_read is CL_FALSE i.e. the read command is non-blocking, clEnqueueReadBuffer 
    /// queues a non-blocking read command and returns. The contents of the buffer that ptr points 
    /// to cannot be used until the read command has completed. The event argument returns an event 
    /// object which can be used to query the execution status of the read command. When the read 
    /// command has completed, the contents of the buffer that ptr points to can be used by the 
    /// application.
    ///
    /// * `offset` - The offset in bytes in the buffer object to read from.
    ///
    /// * `cb` - The size in bytes of data being read.
    ///
    /// * `ptr` - The pointer to buffer in host memory where data is to be read into.
    ///
    /// * `event_wait_list` - specify events that need to complete before this particular command 
    /// can be executed.
    pub fn enqueue_read_buffer(
        &self,
        buffer:          &Buffer,
        blocking_read:   bool,
        offset:          usize,
        cb:              usize,
        ptr:             *mut c_void,
        event_wait_list: &[Event])
        -> Result<Event> {

        unsafe {
            let num_events_in_wait_list = event_wait_list.len() as u32;
            let events = 
                if num_events_in_wait_list > 0 { event_wait_list.as_ptr() } else { ptr::null() };

            let mut new_event = 0 as foreign::cl_event;

            let blocking_read_u32 = if blocking_read { 1 } else { 0 };

            let ret_value = foreign::clEnqueueReadBuffer(
                self.0, 
                buffer.0, 
                blocking_read_u32,
                offset,
                cb,
                ptr,
                num_events_in_wait_list,
                events,
                &mut new_event
            );

            return utility::check(ret_value, || new_event);
        }
    }

    /// Enqueues a command to execute a kernel on a device.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A valid kernel object. The OpenCL context associated with `kernel` 
    /// and `command_queue` must be the same.
    ///
    /// * `work_dim` - The number of dimensions used to specify the global work-items 
    /// and work-items in the work-group. `work_dim` must be greater than zero and less than or 
    /// equal to three.
    ///
    /// * `global_work_size` - Points to an array of work_dim unsigned values that describe the 
    /// number of global work-items in work_dim dimensions that will execute the kernel function. 
    /// The total number of global work-items is computed 
    /// as global_work_size[0] *...* global_work_size[work_dim - 1].
    ///
    /// The values specified in global_work_size cannot exceed the range given by 
    /// the sizeof(size_t) for the device on which the kernel execution will be enqueued. The 
    /// sizeof(size_t) for a device can be determined using CL_DEVICE_ADDRESS_BITS in the table of 
    /// OpenCL Device Queries for clGetDeviceInfo. If, for example, CL_DEVICE_ADDRESS_BITS = 32, i.e. 
    /// the device uses a 32-bit address space, size_t is a 32-bit unsigned integer 
    /// and global_work_size values must be in the range 1 .. 2^32 - 1. Values outside this range 
    /// return a CL_OUT_OF_RESOURCES error.
    ///
    /// * `local_work_size` - Points to an array of work_dim unsigned values that describe the 
    /// number of work-items that make up a work-group (also referred to as the size of 
    /// the work-group) that will execute the kernel specified by kernel. The total number 
    /// of work-items in a work-group is computed 
    /// as local_work_size[0] *... * local_work_size[work_dim - 1]. The total number of work-items 
    /// in the work-group must be less than or equal to the CL_DEVICE_MAX_WORK_GROUP_SIZE value 
    /// specified in table of OpenCL Device Queries for clGetDeviceInfo and the number of 
    /// work-items specified in local_work_size[0],... local_work_size[work_dim - 1] must be less 
    /// than or equal to the corresponding values specified 
    /// by CL_DEVICE_MAX_WORK_ITEM_SIZES[0],.... CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]. The 
    /// explicitly specified local_work_size will be used to determine how to break the global 
    /// work-items specified by global_work_size into appropriate work-group instances. 
    /// If local_work_size is specified, the values specified 
    /// in global_work_size[0],... global_work_size[work_dim - 1] must be evenly divisible by 
    /// the corresponding values specified in local_work_size[0],... local_work_size[work_dim - 1].
    ///
    /// The work-group size to be used for kernel can also be specified in the program source 
    /// using the __attribute__((reqd_work_group_size(X, Y, Z)))qualifier. In this case the size of 
    /// work group specified by local_work_size must match the value specified by 
    /// the reqd_work_group_size __attribute__ qualifier.
    ///
    /// local_work_size can also be a NULL value in which case the OpenCL implementation will 
    /// determine how to be break the global work-items into appropriate work-group instances.
    pub fn enqueue_nd_range_kernel(
        &self, 
        kernel: &Kernel,
        global_work_size: &[usize],
        local_work_size: &[usize],
        event_wait_list: &[Event]) -> Result<Event> {

        unsafe {
            let work_dim = global_work_size.len() as u32;

            // `global_work_offset` must currently be a NULL value. In a future revision 
            // of OpenCL, global_work_offset can be used to specify an array of work_dim unsigned 
            // values that describe the offset used to calculate the global ID of a work-item 
            // instead of having the global IDs always start at offset (0, 0,... 0).
            let global_work_offset = ptr::null();

            let num_events_in_wait_list = event_wait_list.len() as u32;
            let events = 
                if num_events_in_wait_list > 0 { event_wait_list.as_ptr() } else { ptr::null() };

            let mut new_event = 0 as foreign::cl_event;

            // == ptrs
            let global_work_size_ptr = 
                if global_work_size.len() > 0 { global_work_size.as_ptr() } else { ptr::null() };

            let local_work_size_ptr =
                if local_work_size.len() > 0 { local_work_size.as_ptr() } else { ptr::null() };

            let ret_value = foreign::clEnqueueNDRangeKernel(
                self.0,
                kernel.0,
                work_dim,
                global_work_offset,
                global_work_size_ptr,
                local_work_size_ptr,
                num_events_in_wait_list,
                events,
                &mut new_event
            );

            return utility::check(ret_value, || Event::from(new_event));
        }
    }

    /// Increments the command_queue reference count.
    fn retain(&self) -> Result {
        unsafe {
            let ret_value = foreign::clRetainCommandQueue(self.0);
            return utility::check(ret_value, || {});
        }
    }

    /// Decrements the command_queue reference count.
    fn release(&self) -> Result {
        unsafe {
            let ret_value = foreign::clReleaseCommandQueue(self.0);
            return utility::check(ret_value, || {});
        }
    }
}

impl clone::Clone for Queue {

    fn clone(&self) -> Self {

        self.retain().expect("unable to retain `Queue`");

        Queue(self.0)
    }
}

impl cmp::Eq for Queue { }

impl cmp::PartialEq<Queue> for Queue {

    fn eq(&self, other: &Queue) -> bool {

        self.0 == other.0
    }
}

impl ops::Drop for Queue {

    fn drop(&mut self) {

        self.release().expect("unable to release `Queue`");
    }
}