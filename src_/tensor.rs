use super::{DeviceView, ErrorKind, MemoryView, Result};
use std::{convert, mem};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Tensor<T> {
    /// The number of components.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // The following tensor has 9 components
    ///
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    capacity: usize,
    /// The total number of indices.
    ///
    /// # Example
    ///
    /// The following tensor has a rank of 2:
    ///
    /// ```ignore
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    /// ```
    rank: usize,
    /// The dimensions of the tensor.
    shape: Shape,

    buffers: Buffers,
    version: U64Map,

    /// Phantom marker for `T`
    phantom_data: PhantomData<T>,
}

pub type Shape = Vec<usize>;

#[derive(Debug)]
pub struct Buffers {
    re: RefCell<Vec<Buffer>>,
}

#[derive(Debug)]
pub struct Buffer {
    device: DeviceView,
    memory: MemoryView,
}

impl Buffers {

    fn new() -> Buffers {
        Buffers { re: RefCell::new(vec![]) }
    }

    fn position<P>(&self, predicate: P) -> Option<usize> 
        where P: FnMut(&Buffer) -> bool {

        self.re.borrow().iter().position(predicate)
    }

    fn push(&self, device: DeviceView, memory: MemoryView) {
        self.re.borrow_mut().push(Buffer {
            device: device,
            memory: memory,
        })
    }

    fn len(&self) -> usize {
        self.re.borrow().len()
    }

    fn remove(&self, index: usize) -> Buffer {
        self.re.borrow_mut().remove(index)
    }
}

/// A bit-map for keeping track of up-to-date entries. If the number of entries provided by
/// the integer isn't enough, this type can be easily replaced with `BitSet` at the cost of a heap
/// allocation and an extra indirection on access.
#[derive(Debug)]
pub struct U64Map {
    cell: Cell<u64>,
}

impl U64Map {
    /// The number of bits in `BitMap`. It's currently not possible to get this information 
    /// from `BitMap` in a clean manner, though there are plans to add a static method or an 
    /// associated constant.
    const CAPACITY: usize = 64;

    /// Constructs an empty bit-map.
    fn new() -> U64Map {
        U64Map { cell: Cell::new(0) }
    }

    fn set(&self, n: u64) {
        self.cell.set(n)
    }

    fn get(&self) -> u64 {
        self.cell.get()
    }
}

impl<T> Tensor<T> {

    /// Returns the number of elements for which the tensor has been allocated.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the allocated memory size in bytes.
    pub fn mem_size(&self) -> usize {
        mem::size_of::<T>() * self.capacity
    }

    // Functions `read()`, `read_write()`, `write_only()` use `unsafe` to
    // extend lifetime of returned reference to internally owned memory chunk.
    // Borrow guarantees that SharedTensor outlives all of its Tensors, and
    // there is only one mutable borrow. So we only need to make sure that
    // memory entries won't be dropped or moved while there are live Tensors.
    // It's quite easy to do: by convention we only allow to remove elements from
    // `self.entries` in methods with `&mut self`. Since we store device's memory
    // objects in a Box, reference to it won't change during Vec realentries.

    /// Get memory for reading for the specified `device`.
    ///
    /// ## Note
    ///
    /// Can fail if memory allocation fails or if tensor wasn't initialized yet.
    pub fn read<'m>(&'m self, device: &DeviceView) -> Result<&'m MemoryView> {

        if self.version.get() == 0 {

            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(device)?;
        self.sync_if_needed(i)?;
        self.version.set(self.version.get() | (1 << i));

        let re = self.buffers.re.borrow();
        let memory: &MemoryView = &re[i].memory;

        let memory: &'m MemoryView= unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    /// Get memory for reading and writing for the specified `device`.
    /// Can fail if memory allocation fails, or if tensor wasn't initialized yet.
    pub fn read_write<'m>(&'m self, device: &DeviceView) -> Result<&'m mut MemoryView> {

        if self.version.get() == 0 {

            return Err(ErrorKind::UninitializedMemory.into());
        }

        let i = self.get_or_create_location_index(device)?;
        self.sync_if_needed(i)?;
        self.version.set(1 << i);

        let mut re = self.buffers.re.borrow_mut();
        let memory: &mut MemoryView = &mut re[i].memory;

        let memory: &'m mut MemoryView = unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    /// Get memory for writing only.
    ///
    /// This function skips synchronization and initialization checks, since
    /// contents will be overwritten anyway. By convention caller must fully
    /// initialize returned memory. Failure to do so may result in use of
    /// uninitialized data later. If caller has failed to overwrite memory,
    /// for some reason, it must call `invalidate()` to return vector to
    /// uninitialized state.
    pub fn write_only<'m>(&'m mut self, device: &DeviceView) -> Result<&'m mut MemoryView> {

        let i = self.get_or_create_location_index(device)?;
        self.version.set(1 << i);

        let mut re = self.buffers.re.borrow_mut();
        let memory: &mut MemoryView = &mut re[i].memory;

        let memory: &'m mut MemoryView = unsafe { mem::transmute(memory) };

        Ok(memory)
    }

    // FIXME: synchronize memory elsewhere if possible?
    /// Drops memory allocation on the specified device. Returns error if
    /// no memory has been allocated on this device.
    pub fn drop_device(&mut self, device: &DeviceView) -> Result {
        match self.get_location_index(device) {
            Some(i) => {
                self.buffers.remove(i);
                let up_to_date = self.version.get();
                let mask = (1 << i) - 1;
                let lower = up_to_date & mask;
                let upper = (up_to_date >> 1) & (!mask);
                self.version.set(lower | upper);

                Ok(())
            },

            _ => {
                Err(ErrorKind::AllocatedMemoryNotFoundForDevice.into()) // TODO more info on dev
            }
        }
    }

    // ====================================================
    // ====================================================

    // TODO: chose the best source to copy data from.
    // That would require some additional traits that return costs for
    // transferring data between different backends.
    // Actually I think that there would be only transfers between
    // `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    // so it's best to not over-engineer here.
    fn sync_if_needed(&self, dst_i: usize) -> Result {

        if self.version.get() & (1 << dst_i) != 0 {

            return Ok(());
        }

        let src_i = self.version.get().trailing_zeros() as usize;

        assert!(src_i != U64Map::CAPACITY);

        // We need to borrow two different Vec elements: src and mut dst.
        // `Borrow` doesn't allow for that to be done in a straightforward way, so here is 
        // workaround.
        assert!(src_i != dst_i);

        let mut re = self.buffers.re.borrow_mut();

        let (src_loc, mut dst_loc) = if src_i < dst_i {

            let (left, right) = re.split_at_mut(dst_i);

            (&left[src_i], &mut right[0])
        } else {

            let (left, right) = re.split_at_mut(src_i);

            (&right[0], &mut left[dst_i])
        };

        // Backends may define transfers asymmetrically. E. g. CUDA may know how
        // to transfer to and from Native backend, while Native may know nothing
        // about CUDA at all. So if first attempt fails we change order and
        // try again.

        src_loc.device.synch_out(&src_loc.memory, &dst_loc.device, &mut dst_loc.memory)?;

        dst_loc.device.synch_in(&mut dst_loc.memory, &src_loc.device, &src_loc.memory)?;

        // TODO: try transfer indirectly via Native backend

        Ok(())

        // {

        //     match src_loc.device.synch_out(read) {
        
        //         Err(ref e) if e.kind() == ErrorKind::NoAvailableSynchronizationRouteFound => { },

        //         x @ _ => return x
        //     }
        // }

        // {

        //     dst_loc.device.synch(write)
        // }
    }

    /// Looks up `device` in `self.entries` and returns its index. If lookup fails then new 
    /// location is created and its index is returned.
    fn get_or_create_location_index(&self, device: &DeviceView) -> Result<usize> {

        if let Some(i) = self.get_location_index(device) {

            return Ok(i);
        }

        if self.buffers.len() == U64Map::CAPACITY {

            return Err(ErrorKind::BitMapCapacityExceeded.into());
        }

        let size = self.mem_size();

        let memory = device.allocate_memory(size)?;

        self.buffers.push(device.clone(), memory);

        Ok(self.buffers.len() - 1)
    }

    fn get_location_index(&self, device: &DeviceView) -> Option<usize> {

        self.buffers.position(|buffer| buffer.device.eq(device))
    }
}

impl<A, T> From<A> for Tensor<T> where A: AsRef<[usize]> {

    fn from(s: A) -> Tensor<T> {

        let shape: Vec<usize> = s.as_ref().to_vec();
        let rank = shape.len();
        let capacity = shape.iter().fold(1, |s, &a| s * a);

        Tensor {
            capacity: capacity,
            rank: rank,
            shape: shape,
            buffers: Buffers::new(),
            version: U64Map::new(),
            phantom_data: PhantomData,
        }
    }
}