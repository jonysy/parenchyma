use std::cell::Cell;

/// A "newtype" with an internal type of `Cell<u64>`. `TensorMap` uses [bit manipulation][1] to manage 
/// memory versions.
///
/// [1]: http://stackoverflow.com/a/141873/2561805
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub(in super) struct TensorMap(Cell<u64>);

impl TensorMap {
    /// The maximum number of bits in the bit map can contain.
    pub const CAPACITY: usize = 64;

    /// Constructs a new `TensorMap`.
    pub(in super) fn new() -> TensorMap {
        TensorMap::with(0)
    }

    /// Constructs a new `TensorMap` with the supplied `n`.
    pub(in super) fn with(n: u64) -> TensorMap {
        TensorMap(Cell::new(n))
    }

    // fn get(&self) -> u64 {
    //     self.0.get()
    // }

    pub(in super) fn set(&self, v: u64) {
        self.0.set(v)
    }

    pub(in super) fn empty(&self) -> bool {
        self.0.get() == 0
    }

    pub(in super) fn insert(&self, k: usize) {
        self.0.set(self.0.get() | (1 << k))
    }

    pub(in super) fn contains(&self, k: usize) -> bool {
        k < Self::CAPACITY && (self.0.get() & (1 << k) != 0)
    }

    pub(in super) fn latest(&self) -> u32 {
        self.0.get().trailing_zeros()
    }
}