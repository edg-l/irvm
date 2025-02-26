use typed_generational_arena::StandardArena;

use crate::{
    block::{Block, BlockIdx},
    common::{CConv, DllStorageClass, Linkage, Visibility},
    types::Type,
};

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub cconv: Option<CConv>,
    pub linkage: Option<Linkage>,
    pub visibility: Option<Visibility>,
    pub dll_storage: Option<DllStorageClass>,
    pub blocks: StandardArena<Block>,
    pub entry_block: BlockIdx,
    pub result_type: Type,
    pub parameters: Vec<Parameter>,
    pub align: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub ty: Type,
    pub zeroext: bool,
    pub signext: bool,
    pub noext: bool,
    pub inreg: bool,
    pub byval: Option<Type>,
    pub byref: Option<Type>,
    pub preallocated: Option<Type>,
    pub inalloca: Option<Type>,
    pub sret: Option<Type>,
    pub element_type: Option<Type>,
    pub align: Option<u32>,
    pub noalias: bool,
    pub nofree: bool,
    pub nest: bool,
    pub returned: bool,
    pub nonnull: bool,
    pub noundef: bool,
    pub readonly: bool,
    pub writeonly: bool,
    pub deferenceable: Option<u32>,
    // todo:  more attributes
}

impl Parameter {
    pub fn new(ty: Type) -> Self {
        Self {
            ty,
            zeroext: false,
            signext: false,
            noext: false,
            byref: None,
            byval: None,
            preallocated: None,
            inalloca: None,
            sret: None,
            element_type: None,
            align: None,
            inreg: false,
            noalias: false,
            nofree: false,
            nest: false,
            returned: false,
            nonnull: false,
            noundef: false,
            readonly: false,
            writeonly: false,
            deferenceable: None,
        }
    }
}

impl Function {
    pub fn new(name: &str, params: &[Parameter], ret_ty: Type) -> Self {
        let mut blocks = StandardArena::new();
        let entry_block = blocks.insert(Block::new(Vec::new()));
        Self {
            name: name.to_string(),
            cconv: None,
            linkage: None,
            visibility: None,
            dll_storage: None,
            blocks,
            entry_block,
            result_type: ret_ty,
            parameters: params.to_vec(),
            align: None,
        }
    }

    pub fn entry_block(&mut self) -> &mut Block {
        &mut self.blocks[self.entry_block]
    }

    pub fn add_block(&mut self, block: Block) -> BlockIdx {
        self.blocks.insert(block)
    }
}
