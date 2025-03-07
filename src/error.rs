use thiserror::Error;

use crate::{block::BlockIdx, module::TypeIdx};

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("type mismatch, expected {expected:?} found {found:?}")]
    TypeMismatch { expected: TypeIdx, found: TypeIdx },
    #[error("block argument nth {nth:?} not found for block with id {block_id:?}")]
    BlockArgNotFound { block_id: BlockIdx, nth: usize },
    #[error("function param nth {nth:?} not found for function {name:?}")]
    FunctionParamNotFound { name: String, nth: usize },
}
