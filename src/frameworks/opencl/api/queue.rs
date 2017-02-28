use std::ops::Deref;
use std::os::raw::c_void;
use std::ptr;
use super::error::Result;
use super::{Context, Device, Event, Kernel, Memory, sys};

#[derive(Debug)]
pub struct Queue(sys::cl_command_queue);

impl Queue {

    /// Create a command-queue on a specific device.
    pub fn new(context: &Context, device: &Device, properties: u64) -> Result<Self> {

        unsafe {
            
            let mut errcode_ret = 0i32;

            let cl_command_queue = sys::clCreateCommandQueue(
                **context, 
                **device, 
                properties, 
                &mut errcode_ret
            );

            let ret = sys::CLStatus::new(errcode_ret)
                .expect("failed to convert i32 to CLStatus");

            result!(ret => Ok(Queue(cl_command_queue)))
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
    pub fn write_buffer(
        &self, 
        buffer:          &Memory, 
        blocking_write:  bool, 
        offset:          usize,
        cb:              usize, 
        ptr:             *const c_void,
        event_wait_list: &[Event]) 
        -> Result<Event> {

        unsafe {

            let num_events_in_wait_list = event_wait_list.len() as u32;

            // TODO https://github.com/rust-lang/rust/issues/32146
            let list = if num_events_in_wait_list == 0 {
                ptr::null()
            } else {
                event_wait_list.as_ptr() as *const sys::cl_event
            };

            let mut new_event = 0 as sys::cl_event;

            let blocking_write_u32 = if blocking_write { 1 } else { 0 };

            let result = sys::clEnqueueWriteBuffer(
                self.0, 
                buffer.0, 
                blocking_write_u32,
                offset,
                cb,
                ptr,
                num_events_in_wait_list,
                list,
                &mut new_event
            );

            result!(result => Ok(Event::from(new_event)))
        }
    }

    /// Enqueue commands to read from a buffer object to host memory.
    pub fn read_buffer(
        &self,
        buffer:          &Memory,
        blocking_read:   bool,
        offset:          usize,
        cb:              usize,
        ptr:             *mut c_void,
        event_wait_list: &[Event])
        -> Result<Event> {


        unsafe {

            let num_events_in_wait_list = event_wait_list.len() as u32;

            // TODO https://github.com/rust-lang/rust/issues/32146
            let list = if num_events_in_wait_list == 0 {
                ptr::null()
            } else {
                event_wait_list.as_ptr() as *const sys::cl_event
            };

            let mut new_event = 0 as sys::cl_event;

            let blocking_read_u32 = if blocking_read { 1 } else { 0 };

            let result = sys::clEnqueueReadBuffer(
                self.0, 
                buffer.0, 
                blocking_read_u32,
                offset,
                cb,
                ptr,
                num_events_in_wait_list,
                list,
                &mut new_event
            );

            result!(result => Ok(Event::from(new_event)))
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

            // TODO https://github.com/rust-lang/rust/issues/32146
            let list = if num_events_in_wait_list == 0 {
                ptr::null()
            } else {
                event_wait_list.as_ptr() as *const sys::cl_event
            };

            // let local_work_size = if local_work_size.len() == 0 {
            //     ptr::null()
            // } else {
            //     local_work_size.as_ptr()
            // };

            let mut new_event = 0 as sys::cl_event;

            let ret = sys::clEnqueueNDRangeKernel(
                self.0,
                kernel.0,
                work_dim,
                global_work_offset,
                global_work_size.as_ptr(),
                local_work_size.as_ptr(),
                num_events_in_wait_list,
                list,
                &mut new_event
            );

            result!(ret => Ok(Event::from(new_event)))
        }
    }

    /// Increments the command_queue reference count.
    fn retain(&self) -> Result {

        unsafe {

            result!(sys::clRetainCommandQueue(self.0))
        }
    }

    /// Decrements the command_queue reference count.
    fn release(&self) -> Result {

        unsafe {

            result!(sys::clReleaseCommandQueue(self.0))
        }
    }
}

impl Clone for Queue {

    fn clone(&self) -> Self {

        self.retain().unwrap();

        Queue(self.0)
    }
}

impl Drop for Queue {

    fn drop(&mut self) {

        self.release().unwrap()
    }
}

impl Deref for Queue {
    
    type Target = sys::cl_command_queue;
    
    fn deref(&self) -> &Self::Target {
        
        &self.0
    }
}