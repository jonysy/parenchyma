use std::os::raw::c_void;
use super::error::Result;
use super::sys;
use super::Memory;

#[derive(Debug)]
pub struct Kernel(pub(super) sys::cl_kernel);

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
    pub fn set_arg(&self, index: u32, size: usize, value: &Memory) -> Result {

        unsafe {

            result!(sys::clSetKernelArg(self.0, index, ::std::mem::size_of::<sys::cl_mem>(), value.0))
        }
    }

    /// Increment the kernel reference count.
    fn retain(&self) -> Result {

        unsafe {
            
            result!(sys::clRetainKernel(self.0))
        }
    }

    /// Decrement the kernel reference count.
    fn release(&self) -> Result {

        unsafe {
            
            result!(sys::clReleaseKernel(self.0))
        }
    }
}

impl Clone for Kernel {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Kernel(self.0)
    }
}

impl Drop for Kernel {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}

impl From<sys::cl_kernel> for Kernel {
    
    fn from(cl_kernel: sys::cl_kernel) -> Self {
        
        Kernel(cl_kernel)
    }
}