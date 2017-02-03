use std::error;

#[derive(Debug)]
pub struct Error {
	category: Category,
	error: Box<error::Error + Send + Sync>,
}

/// A list specifying general error categories.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Category {
    Tensor(TensorCategory),

    Memory(MemoryCategory),
    
    /// A marker variant that tells the compiler that users of this enum cannot match 
    /// it exhaustively.
    ///
    /// [Private enum variants #32770](https://github.com/rust-lang/rust/issues/32770)
    #[doc(hidden)]
    _NonExhaustive,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorCategory {
	Shape, Remove, CapacityExceeded
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryCategory {
	NoMemorySyncRoute, Uninitialized, Remove
}

impl From<TensorCategory> for Category {

	fn from(category: TensorCategory) -> Category {
		Category::Tensor(category)
	}
}

impl From<MemoryCategory> for Category {

	fn from(category: MemoryCategory) -> Category {
		Category::Memory(category)
	}
}

impl Error {
	pub fn new<T, E>(category: T, error: E) -> Error
		where T: Into<Category>,
			  E: Into<Box<error::Error + Send + Sync>>
	{
		Error {
			category: category.into(),
			error: error.into()
		}
	}

	pub fn category(&self) -> &Category {
		&self.category
	}
}