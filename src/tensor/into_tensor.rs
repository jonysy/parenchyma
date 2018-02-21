use ndarray::{Array, Dimension};
use std::cell::RefCell;

use super::{SharedTensor, TensorMap, TensorShape};
use super::super::memory::Memory;
use super::super::frameworks::NativeMemory;

pub trait IntoTensor<T> {
    fn into_tensor(self) -> SharedTensor<T>;
}

impl<T: 'static, D> IntoTensor<T> for Array<T, D> where D: Dimension {
    fn into_tensor(self) -> SharedTensor<T> {
        SharedTensor::<T>::from(self)
    }
}

impl<T, Dim> From<Array<T, Dim>> for SharedTensor<T> where 
    T: 'static,
    Dim: Dimension {

    fn from(array: Array<T, Dim>) -> Self {
        if !array.is_standard_layout() {
            panic!("Array data must be laid out in contiguous “C order” in memory");
        }

        let shape = TensorShape::from(array.shape());
        let n = NativeMemory(array.into_dyn());

        let memories = RefCell::new(vec![
            Box::new(n) as Box<Memory<T>>
        ]);
        
        let synch_map = TensorMap::with(1 << 0);
        SharedTensor { memories, shape, synch_map }
    }
}