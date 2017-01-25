use frameworks::NativeMemory;

pub enum MemoryImp {
	#[cfg(feature = "native")]
	Native(NativeMemory)
}