use ocl::traits::OclPrm as PrimitiveType;

/// A marker trait implemented by primitive types that usable within kernels.
pub trait TensorType: PrimitiveType {
    // ..
}

impl<T: PrimitiveType> TensorType for T {
    // ..
}