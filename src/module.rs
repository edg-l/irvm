use target_lexicon::Triple;
use typed_generational_arena::{StandardSlab, StandardSlabIndex};

use crate::{
    common::{DllStorageClass, Linkage, Location, ThreadLocalStorageModel, Visibility},
    datalayout::DataLayout,
    function::{FnIdx, Function, Parameter},
    types::TypeIdx,
    value::ConstValue,
};

/// A global variable index.
pub type GlobalIdx = StandardSlabIndex<GlobalVariable>;

/// A global variable in the module.
#[derive(Debug, Clone)]
pub struct GlobalVariable {
    pub id: Option<GlobalIdx>,
    pub name: String,
    pub ty: TypeIdx,
    pub initializer: Option<ConstValue>,
    pub linkage: Option<Linkage>,
    pub visibility: Option<Visibility>,
    pub dll_storage: Option<DllStorageClass>,
    pub thread_local: Option<ThreadLocalStorageModel>,
    pub unnamed_addr: bool,
    pub local_unnamed_addr: bool,
    pub addr_space: Option<u32>,
    pub externally_initialized: bool,
    /// If true, this is a constant (immutable).
    pub is_constant: bool,
    pub section: Option<String>,
    pub align: Option<u32>,
    pub location: Location,
}

impl GlobalVariable {
    /// Create a new global variable.
    pub fn new(name: &str, ty: TypeIdx, location: Location) -> Self {
        Self {
            id: None,
            name: name.to_string(),
            ty,
            initializer: None,
            linkage: None,
            visibility: None,
            dll_storage: None,
            thread_local: None,
            unnamed_addr: false,
            local_unnamed_addr: false,
            addr_space: None,
            externally_initialized: false,
            is_constant: false,
            section: None,
            align: None,
            location,
        }
    }
}

/// A module contains all the information about the given compilation unit.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub data_layout: DataLayout,
    pub target_triple: Triple,
    functions: StandardSlab<Function>,
    globals: StandardSlab<GlobalVariable>,
    pub location: Location,
}

impl Module {
    pub fn new(name: &str, location: Location) -> Self {
        Self {
            name: name.to_string(),
            data_layout: DataLayout::default_host(),
            target_triple: Triple::host(),
            functions: StandardSlab::new(),
            globals: StandardSlab::new(),
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

    // ==================== Global Variables ====================

    /// Get all global variables.
    pub fn globals(&self) -> &StandardSlab<GlobalVariable> {
        &self.globals
    }

    /// Add a global variable.
    pub fn add_global(
        &mut self,
        name: &str,
        ty: TypeIdx,
        initializer: Option<ConstValue>,
        is_constant: bool,
        location: Location,
    ) -> GlobalIdx {
        let id = self.globals.insert(GlobalVariable {
            id: None,
            name: name.to_string(),
            ty,
            initializer,
            linkage: None,
            visibility: None,
            dll_storage: None,
            thread_local: None,
            unnamed_addr: false,
            local_unnamed_addr: false,
            addr_space: None,
            externally_initialized: false,
            is_constant,
            section: None,
            align: None,
            location,
        });
        self.globals[id].id = Some(id);
        id
    }

    /// Add a global variable with full configuration.
    pub fn add_global_ex(&mut self, global: GlobalVariable) -> GlobalIdx {
        let id = self.globals.insert(global);
        self.globals[id].id = Some(id);
        id
    }

    /// Get a global variable by index.
    pub fn get_global(&self, id: GlobalIdx) -> &GlobalVariable {
        &self.globals[id]
    }

    /// Get a mutable reference to a global variable.
    pub fn get_global_mut(&mut self, id: GlobalIdx) -> &mut GlobalVariable {
        &mut self.globals[id]
    }
}
