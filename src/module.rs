use target_lexicon::Triple;
use typed_generational_arena::StandardArena;

use crate::{datalayout::DataLayout, function::Function};

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub data_layout: DataLayout,
    pub target_triple: Triple,
    pub functions: StandardArena<Function>,
}
