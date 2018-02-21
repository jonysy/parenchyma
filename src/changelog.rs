//! Project changelog (YEAR-MONTH-DAY)

/// Release 0.0.4 (2017-11-08)
///
/// * Partially sketched out a transfer matrix addressing issue#23
/// * Simplified the complicated extension/build system resolving issue#25
///     * The new extension/build system allows for framework specific backends.
/// * Worked on a OpenCL solution to issue#16
/// * Removed ndarray as it's not needed, which closes issue#20
///     * Mapped memory process doesn't work well with ndarray + reshaping a tensor means reshaping
/// the native array
/// * Lazy synchronization via auto-sync has been fully integrated
/// * Implemented logic around pinned memory with unpinned memory fallback
pub mod r0_0_4 {}

/// Release 0.0.3 (2017-03-04)
///
/// * Implemented an OpenCL API wrapper
/// * Partially implemented a CUDA API wrapper
/// * Partially implemented native support
/// * Worked on a fallback mechanism (see issue#15)
///     * No longer requires framework related feature flags (from the original Collenchyma project)
///     * No longer requires backends parameterized by a framework
/// * New memory access API
///     * Implemented auto-sync
///     * Use a tensor lib (ndarray) as the underlying native memory representation
/// * Add `Bundle` logic
/// * Removed `IBinary`/`HashMap` technique. Use structs instead
pub mod r0_0_3 {}