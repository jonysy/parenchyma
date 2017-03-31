//! Helper functions and traits

//use SharedTensor;
use error::Result;
use std::mem;

/// Returns the size (in bytes) of `size_of::<T>` * `length`.
pub fn allocated<T>(length: usize) -> usize {

    length * mem::size_of::<T>()
}

// /// Write into a native Parenchyma `Memory`.
// pub fn write_to_memory<T>(tensor: &mut SharedTensor<T>, data: &[T]) -> Result where T: Copy {
//     write_to_memory_offset(tensor, data, 0)
// }

// /// Write into a native Parenchyma `Memory` with an offset.
// pub fn write_to_memory_offset<T>(tensor: &mut SharedTensor<T>, data: &[T], offset: usize) -> Result 
//     where T: Copy {

//     tensor.shape.check(data)?;

//     // let mut memory = tensor.write(HOST)?;
//     // let buffer = unsafe { memory.as_mut_native_unchecked().as_mut_flat() };

//     // for (index, datum) in data.iter().enumerate() {
//     //     buffer[index + offset] = *datum;
//     // }
        
//     // Ok(())

//     unimplemented!()
// }