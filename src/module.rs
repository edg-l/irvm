use target_lexicon::Triple;
use typed_generational_arena::StandardSlab;

use crate::{
    datalayout::DataLayout,
    function::{FnIdx, Function, Parameter},
    types::Type,
};

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
            data_layout: DataLayout::default_host(),
            target_triple: Triple::host(),
            functions: StandardSlab::new(),
        }
    }

    pub fn add_function(
        &mut self,
        name: &str,
        params: &[Parameter],
        ret_ty: Type,
    ) -> &mut Function {
        let id = self.functions.insert(Function::new(name, params, ret_ty));
        self.functions[id].id = Some(id);
        &mut self.functions[id]
    }

    pub fn get_function(&self, id: FnIdx) -> &Function {
        &self.functions[id]
    }

    pub fn func(&mut self, id: FnIdx) -> &mut Function {
        &mut self.functions[id]
    }

    pub fn get_function_mut(&mut self, id: FnIdx) -> &mut Function {
        &mut self.functions[id]
    }
}
