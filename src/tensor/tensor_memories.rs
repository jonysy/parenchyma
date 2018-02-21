use std::cell::RefCell;
use super::super::memory::Memory;

pub type TensorMemories<T> = RefCell<Vec<Box<Memory<T>>>>;