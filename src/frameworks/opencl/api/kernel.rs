use super::error::Result;
use super::sys;

#[derive(Debug)]
pub struct Kernel(sys::cl_kernel);

impl Kernel {

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