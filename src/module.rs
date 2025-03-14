use target_lexicon::Triple;
use typed_generational_arena::StandardSlab;

use crate::{
    common::Location,
    datalayout::DataLayout,
    function::{FnIdx, Function, Parameter},
    types::TypeIdx,
};

/// A module contains all the information about the given compilation unit.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub data_layout: DataLayout,
    pub target_triple: Triple,
    functions: StandardSlab<Function>,
    pub location: Location,
}

impl Module {
    pub fn new(name: &str, location: Location) -> Self {
        Self {
            name: name.to_string(),
            data_layout: DataLayout::default_host(),
            target_triple: Triple::host(),
            functions: StandardSlab::new(),
            location,
        }
    }

    pub fn functions(&self) -> &StandardSlab<Function> {
        &self.functions
    }

    pub fn add_function(
        &mut self,
        name: &str,
        params: &[Parameter],
        ret_ty: Option<TypeIdx>,
        location: Location,
    ) -> &mut Function {
        let id = self
            .functions
            .insert(Function::new(name, params, ret_ty, location));
        self.functions[id].id = Some(id);
        &mut self.functions[id]
    }

    pub fn get_function(&self, id: FnIdx) -> &Function {
        &self.functions[id]
    }

    pub fn get_function_mut(&mut self, id: FnIdx) -> &mut Function {
        &mut self.functions[id]
    }
}
