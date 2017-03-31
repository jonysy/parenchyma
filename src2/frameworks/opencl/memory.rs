use alloc::raw_vec::RawVec;
use error::{Error, ErrorKind, Result};
use frameworks::native::{NativeChunk, NativeDevice};
use futures::{Async, Future};
use hardware::HardwareDevice;
use memory::{BoxFuture, Chunk, FlatBox, Memory, Synch};
use std::cell::{Cell, RefCell};
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use super::api::{core, import};
use super::{ComputeDevice, Event};

/// A `Memory` wraps around an OpenCL buffer id that manages its deallocation, named 
/// as such for consistency's sake.
///
/// Memory objects can be copied to host memory, from host memory, or to other memory objects.
/// Copying from the host to a device is considered _writing_. Copying from a device to the host is
/// considered _reading_.
///
/// Unlike CUDA, OpenCL [buffers][1] are only context specific, not device specific. Also note: 
/// currently, lazy allocation is used on the NVIDIA driver. That is, the buffer object, in a sense,
/// is located _nowhere_ when allocated. It only exists when needed.
///
/// **note**:
///
/// The lock determines whether or not the memory object is mapped/unmapped (locked = true = mapped)
/// The borrowck takes care of mapping and un-mapping buffers via the guards. The borrowck 
/// determines whether are not the current memory object is mapped or if the host application can 
/// access `pointer`. If un-mapped, you should assume that the `host` pointer is no longer 
/// valid. If mapped, you should assume that only the `host` pointer is valid.
///
/// [1]: https://goo.gl/S9B3TL
#[derive(Debug)]
pub struct MemoryLock {
    /// The `host_ptr`.
    host: FlatBox,
    /// The OpenCL memory object
    memory: core::Buffer,
    /// The command queue associated with the `memory_object`
    queue: core::CommandQueue,
    /// The _lock_/_unlock_ event (`clEnqueue*`)
    ///
    /// **Approach**:
    ///
    ///     1) Memory buffers will (*most likely..?) only store a single event. Once the `events` 
    ///     are provided to a function, it will then be replaced with the returning event.
    ///         a) Reasoning
    ///             1) `event_wait_list` is a list containing events that need to complete before a 
    ///             particular command can be executed
    ///             2) The command returns an event object that identifies that 
    ///             particular read/write command and can be used to query or queue a wait for that 
    ///             particular command to complete.
    ///     2) This approach needs to be benchmarked
    event: RefCell<Event>,
}

impl MemoryLock {

    /// Constructs a new `MemoryLock.
    ///
    /// # Arguments
    ///
    /// * `cx` - The context.
    /// * `queue` - The command queue.
    /// * `capacity` - The capacity of the raw vector in bytes.
    pub(super) fn new(cx: &core::Context, queue: core::CommandQueue, capacity: usize) -> Result<MemoryLock> {
        // returns a `RawVec` (of bytes (`u8`)) with exactly the capacity and alignment requirements 
        // for a `[u8; memory::allocated_nbytes::<T>(capacity)]`.
        //
        // **note**:
        //
        // `capacity` is the number of bytes the buffer can store.
        let raw_vector: RawVec<u8> = RawVec::with_capacity(capacity);
        let boxed: Box<[u8]> = unsafe { raw_vector.into_box() };
        let boxed = FlatBox::from(boxed);

        MemoryLock::with(cx, queue, boxed)
    }

    /// Constructs a new `MemoryLock` from a `FlatBox`.
    ///
    /// # Arguments
    ///
    /// * `cx` - The context.
    /// * `queue` - The command queue.
    /// * `host` - The `host_ptr`.
    pub(super) fn with(cx: &core::Context, queue: core::CommandQueue, host: FlatBox) -> Result<MemoryLock> {
        use super::api::import::CL_COMPLETE;

        // turn over access from the host to the device
        let memory = {
            use super::api::import::{CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR};
            // Create a buffer from the provided `data`.
            //
            // **note**: 
            //
            // Writing to a buffer or image object created with `CL_MEM_READ_ONLY` inside a 
            // kernel is undefined.
            //
            // `CL_MEM_READ_WRITE` specifies that the memory object will be read and written 
            // by a kernel (the default).
            // `CL_MEM_COPY_HOST_PTR` copies the values of `data`.
            cx.create_buffer(
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 
                host.byte_size,
                host.pointer as *mut c_void
            )?
        };

        let locked = Cell::new(true);

        let event = RefCell::new(Event::new(cx, CL_COMPLETE)?);

        Ok(MemoryLock { host, memory, queue, event })
    }

    // pub fn as_slice<'a, T: 'a>(&'a self) -> impl Future<Item = Result<&'a [T]>, Error = Error> {
    //     // let unlocked = self.read::<'a>().expect("an error occurred during unmapping").__lock;
    //     // let event = unlocked.event.clone().into_inner();

    //     // event.map(move |_| Ok(unlocked.host.as_slice::<T>()))

    //     unimplemented!()
    // }

    // pub fn as_mut_slice<'a, T>(&'a mut self) -> Result<&'a mut [T]> {
    //     // event wait
    //     // Ok(self.read_write()?.__lock.host.as_mut_slice::<T>())
    // }

    /// The application can access the memory `pointer` is pointing to once the `event` is complete.
    fn connect(&self, markers: u64) -> impl Future<Item = (), Error = Error> {
        //use super::api::import::{CL_MAP_READ, CL_MAP_WRITE};

        let event = {
            core::enqueue_map_buffer(
                &self.queue,
                &self.memory,
                markers, //CL_MAP_READ | CL_MAP_WRITE,
                0,
                self.host.byte_size,
                &[self.event.borrow().clone()]
            )
            .expect("an error occurred while mapping ")
        };

        event
    }

    /// The host application no longer has access to the host memory, and access has reverted
    /// back to the device once the event is complete.
    fn disconnect(&self) {
        let event = {
            core::enqueue_unmap_mem_object(
                &self.queue,
                &self.memory,
                self.host.pointer as *mut c_void,
                &[]
            )
            .expect("an error occurred while un-mapping ")
        };

        *self.event.borrow_mut() = event;
    }

    pub fn read<'a>(&'a self) -> impl Future<Item=View<'a>, Error=Error> {
        use super::api::import::CL_MAP_READ;

        self.connect(CL_MAP_READ).map(move |_| 
            View { lock: self })
    }

    // pub fn view_mut<'a, T: 'a>(&'a mut self) -> impl Future<Item=ViewMut<'a>, Error=Error> {
    //     use super::api::import::{CL_MAP_READ, CL_MAP_WRITE};

    //     self.connect(CL_MAP_READ | CL_MAP_WRITE).map(move |_| 
    //         ViewMut { lock: self })
    // }

    pub fn write<'a>(&'a mut self) -> impl Future<Item=ViewMut<'a>, Error=Error> {
        use super::api::import::CL_MAP_WRITE;

        self.connect(CL_MAP_WRITE).map(move |_| 
            ViewMut { lock: self })
    }

    // /// OpenCL <- Host
    // pub fn synchronize_in<'a, T: 'a>(&'a mut self, source: &'a mut [T]) 
    //     -> impl Future<Item=(), Error=Error> + 'a
    //     where [T]: Read + Write 
    // {   
    //     use super::api::import::CL_MAP_WRITE;

    //     self.write::<'a, T>().and_then(move |mut v| {
    //         let r = {
    //             let slice: &mut [T] = &mut v;

    //             io::copy(source, slice)
    //                 .map_err(|e| Error::new(ErrorKind::Other, e))
    //                 .map(|_| { })
    //         };
    //         ::futures::future::result::<(), Error>(r)
    //     })
    // }
}

pub struct View<'a> {
    lock: &'a MemoryLock,
}

impl<'a> Drop for View<'a> {

    fn drop(&mut self) {
        // disconnect before dropping
        self.lock.disconnect()
    }
}

// impl<'a> Deref for View<'a, T> {

//     type Target = [T];

//     fn deref(&self) -> &[T] {
//         self.lock.host.as_slice::<T>()
//     }
// }

pub struct ViewMut<'a> {
    lock: &'a mut MemoryLock,
}

// impl<'a, T> Deref for ViewMut<'a, T> {

//     type Target = [T];

//     fn deref(&self) -> &[T] {
//         self.lock.host.as_slice::<T>()
//     }
// }

// impl<'a, T> DerefMut for ViewMut<'a, T> {

//     fn deref_mut(&mut self) -> &mut [T] {
//         self.lock.host.as_mut_slice::<T>()
//     }
// }

impl<'a> Drop for ViewMut<'a> {

    fn drop(&mut self) {
        // disconnect before dropping
        self.lock.disconnect()
    }
}

impl Memory for MemoryLock { }

pub type OpenCLChunk = (ComputeDevice, MemoryLock);

impl Chunk for OpenCLChunk {

    fn this(&self) -> (&HardwareDevice, &Memory) {
        let &(ref h, ref m) = self;
        (h, m)
    }

    fn located_on(&self, other: &HardwareDevice) -> bool {
        match other.downcast_ref::<ComputeDevice>() {
            Some(oth) => self.0.eq(oth),
            _ => false,
        }
    }
}

impl<T> Synch<T> for OpenCLChunk {

    fn syncable(&self, device: &HardwareDevice) -> bool {
        device.is::<NativeDevice>() | device.is::<ComputeDevice>()
    }

    fn synchronize_in<'a>(&'a mut self, source: &'a mut Chunk) -> BoxFuture {

        use futures::future;
    
        if let Some(cl_chunk_source) = source.downcast_mut::<OpenCLChunk>() {
            // opencl <-> opencl
            if self.0.context == cl_chunk_source.0.context {
                unimplemented!()
            }

            // opencl context a <-> opencl context b
            else {
                unimplemented!()
            }
        }

        if let Some(native_chunk_source) = source.downcast_mut::<NativeChunk>() {
            // opencl <- host
            let fut =
                self.1.write().and_then(move |mut v| unsafe {
                    let count = v.lock.host.byte_size;

                    future::ok(
                        ::std::ptr::copy(
                            native_chunk_source.1.pointer as *const [u8] as *const u8, 
                            v.lock.host.pointer as *mut u8, 
                            count))
                });

            return Box::new(fut);
        }

        Box::new(::futures::future::err(
            ErrorKind::NoAvailableSynchronizationRouteFound.into()))
    }

    /// Synchronizes `memory` to `destination`.
    fn synchronize_out<'a>(&'a mut self, destination: &'a mut Chunk) -> BoxFuture {

        use futures::future;

        if let Some(cl_chunk_destination) = destination.downcast_mut::<OpenCLChunk>() {
            // opencl <-> opencl
            if self.0.context == cl_chunk_destination.0.context {
                unimplemented!()
            }

            // opencl context a <-> opencl context b
            else {
                unimplemented!()
            }
        }

        if let Some(native_chunk_destination) = destination.downcast_mut::<NativeChunk>() {
            // opencl -> host
            let fut =
                self.1.read().and_then(move |v| unsafe {
                    let count = v.lock.host.byte_size;

                    future::ok(
                        ::std::ptr::copy(
                            v.lock.host.pointer as *const [u8] as *const u8,
                            native_chunk_destination.1.pointer as *mut u8,
                            count))
                });

            return Box::new(fut);
        }

        Box::new(::futures::future::err(
            ErrorKind::NoAvailableSynchronizationRouteFound.into()))
    }
}