#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]

/// Basic blocks
pub mod block;
/// Common code.
pub mod common;
/// Type sizes, align, abi.
pub mod datalayout;
/// The crate error type.
pub mod error;
/// IR Function.
pub mod function;
/// IR Module.
pub mod module;
/// IR Types.
pub mod types;
/// IR Values.
pub mod value;

pub use target_lexicon;
