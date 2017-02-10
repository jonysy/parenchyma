use std::cell::Cell;

/// A bit-map for keeping track of up-to-date entries. If the number of entries provided by
/// the integer isn't enough, this type can be easily replaced with `BitSet` at the cost of a heap
/// allocation and an extra indirection on access.
pub struct u64Map {
    n: Cell<u64>,
}

impl u64Map {

    /// The number of bits in `BitMap`. It's currently not possible to get this information 
    /// from `BitMap` in a clean manner, though there are plans to add a static method or an 
    /// associated constant.
    pub const capacity: usize = 64;

    /// Makes an empty bit-map.
    pub fn new() -> u64Map {
        u64Map {
            n: Cell::new(0),
        }
    }

    pub fn set(&self, value: u64) {
        self.n.set(value)
    }

    pub fn get(&self) -> u64 {
        self.n.get()
    }
}