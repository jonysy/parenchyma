use std::cell::Cell;

/// A "newtype" with an internal type of `Cell<u64>`. `u64Map` uses [bit manipulation][1] to manage 
/// memory versions.
///
/// [1]: http://stackoverflow.com/a/141873/2561805
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct u64Map(Cell<u64>);

impl u64Map {
    /// The maximum number of bits in the bit map can contain.
    pub const CAPACITY: usize = 64;

    /// Constructs a new `u64Map`.
    pub fn new() -> u64Map {
        u64Map::with(0)
    }

    /// Constructs a new `u64Map` with the supplied `n`.
    pub fn with(n: u64) -> u64Map {
        u64Map(Cell::new(n))
    }

    pub fn get(&self) -> u64 {
        self.0.get()
    }

    pub fn set(&self, v: u64) {
        self.0.set(v)
    }

    pub fn empty(&self) -> bool {
        self.0.get() == 0
    }

    pub fn insert(&self, k: usize) {
        self.0.set(self.0.get() | (1 << k))
    }

    pub fn contains(&self, k: usize) -> bool {
        k < Self::CAPACITY && (self.0.get() & (1 << k) != 0)
    }

    pub fn latest(&self) -> u32 {
        self.0.get().trailing_zeros()
    }
}