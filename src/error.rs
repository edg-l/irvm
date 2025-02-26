use thiserror::Error;

use crate::types::Type;


#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("type mismatch, expected {expected:?} found {found:?}")]
    TypeMismatch {
        expected: Type,
        found: Type,
    }
}
