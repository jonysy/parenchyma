use rblas;

/// Possible transpose operations that can be applied in Level 2 and Level 3 BLAS operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transposition {
    /// Take the conjugate transpose of the matrix.
    ConjugateTranspose,
    /// Take the matrix as it is.
    NoTranspose,
    /// Take the transpose of the matrix.
    Transpose,
}

impl Into<rblas::attribute::Transpose> for Transposition {
    /// Converts a `Transposition` to an rblas `Transpose`.
    fn into(self) -> rblas::attribute::Transpose {
        match self {
            Transposition::ConjugateTranspose => rblas::attribute::Transpose::ConjTrans,
            Transposition::NoTranspose => rblas::attribute::Transpose::NoTrans,
            Transposition::Transpose => rblas::attribute::Transpose::Trans,
        }
    }
}