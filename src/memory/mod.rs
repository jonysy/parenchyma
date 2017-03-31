//! Provides a unified representation of memory across different frameworks.

pub use self::chunk::Chunk;
pub use self::synchronization::Synch;
pub use self::transfer::AsyncTransfer;

mod chunk;
mod synchronization;
mod transfer;

pub trait Memory {

    // /// Returns `true` if the memory is pinned.
    // fn pinned(&self) -> bool;
}