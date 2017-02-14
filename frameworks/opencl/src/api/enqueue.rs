use std::os::raw::c_void;
use std::ptr;
use super::error::Result;
use super::{Event, Memory, Queue, sys};

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
    command_queue:   &Queue, 
    buffer:          &Memory, 
    blocking_write:  bool, 
    offset:          usize,
    cb:              usize, 
    ptr:             *const c_void,
    event_wait_list: &[Event]) 
    -> Result<Event> {

    unsafe {

        let num_events_in_wait_list = event_wait_list.len() as u32;

        let list = if num_events_in_wait_list == 0 {
            ptr::null()
        } else {
            event_wait_list.as_ptr() as *const sys::cl_event
        };

        let mut new_event = 0 as sys::cl_event;

        let blocking_write_u32 = if blocking_write { 1 } else { 0 };

        let result = sys::clEnqueueWriteBuffer(
            **command_queue, 
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
    command_queue:   &Queue,
    buffer:          &Memory,
    blocking_read:   bool,
    offset:          usize,
    cb:              usize,
    ptr:             *mut c_void,
    event_wait_list: &[Event])
    -> Result<Event> {


    unsafe {

        let num_events_in_wait_list = event_wait_list.len() as u32;

        let list = if num_events_in_wait_list == 0 {
            ptr::null()
        } else {
            event_wait_list.as_ptr() as *const sys::cl_event
        };

        let mut new_event = 0 as sys::cl_event;

        let blocking_read_u32 = if blocking_read { 1 } else { 0 };

        let result = sys::clEnqueueReadBuffer(
            **command_queue, 
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