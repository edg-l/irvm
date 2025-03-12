#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]

pub mod block;
pub mod common;
pub mod datalayout;
pub mod error;
pub mod function;
pub mod module;
pub mod types;
pub mod value;

pub use target_lexicon;
