use {ContextImp, MemoryImp};
use linear_map::LinearMap;
use std::marker::PhantomData;
use super::TensorDesc;

pub struct SharedTensor<T> {
	desc: TensorDesc,
	latest_location: ContextImp,
	latest_copy: MemoryImp,
	copies: LinearMap<ContextImp, MemoryImp>,
	phantom: PhantomData<T>,
}