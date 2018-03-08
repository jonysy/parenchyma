//! Provides backend-agnostic [BLAS] operations for [Parenchyma].
//!
//! BLAS (Basic Linear Algebra Subprograms) is a specification that prescribes a set of low-level
//! routines for performing common linear algebra operations such as vector addition, scalar
//! multiplication, dot products, linear combinations, and matrix multiplication. They are the de
//! facto standard low-level routines for linear algebra libraries; the routines have bindings for
//! both C and Fortran. Although the BLAS specification is general, BLAS implementations are often
//! optimized for speed on a particular machine, so using them can bring substantial performance
//! benefits. BLAS implementations will take advantage of special floating point hardware such as
//! vector registers or SIMD instructions.<br/>
//!
//! # Overview
//!
//! A Parenchyma extension package provides functionality through two types:
//!
//! * __Package__
//! This enum provides the actual initialized functions.
//!
//! * __Extension__
//! This trait provides methods that specify the exact backend-agnostic behavior of a collection of
//! operations. Since a shared tensor completely manages memory, tensors can simply be passed in as
//! arguments for the fastest possible execution.
//!
//! Aside from the generic functionality provided by the two traits, the extension can be further
//! extended.
//!
//! For more information, read the documentation.
//!
//! # Example Usage
//!
//! ```ignore
//! #[macro_use(array)]
//! extern crate parenchyma;
//! extern crate parenchyma_blas as blas;
//!
//! use parenchyma::frameworks::Native;
//! use parenchyma::prelude::*;
//!
//! let backend: Backend<blas::Package> = Backend::new::<Native>()?;
//! let ref x: SharedTensor = array![[1.5, 2.5, 3.5], [4.5, 5.5, 6.6]].into();
//! let ref mut result: SharedTensor = array![0.0].into();
//!
//! backend.asum(x, result)?;
//!
//! println!("{:?}", result);
//! ```
//!
//! [BLAS]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
//! [Parenchyma]: https://github.com/lychee-eng/parenchyma
#![allow(unused_variables)]
#![feature(non_modrs_mods)]

extern crate ocl;
extern crate parenchyma;
extern crate rblas;

pub use self::extension_package::{Extension, GenericMatrix, Package, Transposition};

mod extension_package;
mod frameworks;