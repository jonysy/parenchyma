use std::cell::Cell;

/// A "newtype" with an internal type of `Cell<u64>`. `Bitmap` uses [bit manipulation][1] to manage 
/// memory versions.
///
/// [1]: http://stackoverflow.com/a/141873/2561805
#[derive(Debug)]
pub struct Bitmap(Cell<u64>);

impl Bitmap {
    /// The maximum number of bits in the bit map can contain.
    pub const CAPACITY: usize = 64;

    /// Constructs a new `Bitmap`.
    pub fn new() -> Bitmap {
        Bitmap::with(0)
    }

    /// Constructs a new `Bitmap` with the supplied `n`.
    pub fn with(n: u64) -> Bitmap {
        Bitmap(Cell::new(n))
    }

    pub fn get(&self) -> u64 {
        self.0.get()
    }

    pub fn set(&self, v: u64) {
        self.0.set(v)
    }

    pub fn clear(&self) {
        self.set(0)
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