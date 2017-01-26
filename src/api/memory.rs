use error::Error;

/// Provides a representation for memory across different frameworks.
///
/// Memory is allocated by a device in a way that it is accessible for its computations.
pub enum Memory {
	/// A native memory representation.
	#[cfg(feature = "native")]
	Native(::frameworks::NativeMemory),
}

impl Memory {

	/// Returns `Ok(&FlatBox)` if `Memory` is `Native`.
	#[cfg(feature = "native")]
	pub fn as_native(&self) -> Result<&::frameworks::NativeMemory, Error> {

		match self {
			&Memory::Native(ref mem) => Ok(mem),

			#[cfg(any(feature = "cuda", feature = "opencl"))]
			_ => Err(Error::memory("Expected `Native` memory (`FlatBox`).")),
		}
	}

	#[cfg(feature = "native")]
	pub fn as_mut_native(&mut self) -> Result<&mut ::frameworks::NativeMemory, Error> {

		match self {
			&mut Memory::Native(ref mut mem) => Ok(mem),

			#[cfg(any(feature = "cuda", feature = "opencl"))]
			_ => Err(Error::memory("Expected `Native` memory (`FlatBox`).")),
		}
	}
}