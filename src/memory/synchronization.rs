use super::AsyncTransfer;

/// Specifies synchronization behavior for keeping data consistent across frameworks and contexts.
///
/// **note**
///
/// _Synch_ shouldn't be confused with the marker type `Sync` found in the standard library. 
/// The less common abbreviation for _synchronize_ (the extra _h_) is used here to avoid confusion.
pub trait Synch: AsyncTransfer {

}