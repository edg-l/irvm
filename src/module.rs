use target_lexicon::Triple;
use typed_generational_arena::StandardSlab;

use crate::{datalayout::DataLayout, function::Function};

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub data_layout: DataLayout,
    pub target_triple: Triple,
    pub functions: StandardSlab<Function>,
}

impl Module {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            data_layout: DataLayout::default(),
            target_triple: Triple::host(),
            functions: StandardSlab::new(),
        }
    }
}
