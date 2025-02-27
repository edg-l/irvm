use typed_generational_arena::{StandardIndex, StandardSlab};

use crate::{
    block::{Block, BlockIdx, Terminator},
    common::{CConv, DllStorageClass, Linkage, Visibility},
    error::Error,
    types::Type,
    value::Operand,
};

pub type FnIdx = StandardIndex<Function>;

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub cconv: Option<CConv>,
    pub linkage: Option<Linkage>,
    pub visibility: Option<Visibility>,
    pub dll_storage: Option<DllStorageClass>,
    pub blocks: StandardSlab<Block>,
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
        let mut blocks = StandardSlab::new();
        let entry_block = blocks.insert(Block::new(&[]));
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

    pub fn param(&self, nth: usize) -> Result<Operand, Error> {
        self.parameters
            .get(nth)
            .map(|x| Operand::Parameter(nth, x.ty.clone()))
            .ok_or_else(|| Error::FunctionParamNotFound {
                name: self.name.clone(),
                nth,
            })
    }

    pub fn entry_block(&mut self) -> &mut Block {
        &mut self.blocks[self.entry_block]
    }

    pub fn add_block(&mut self, arguments: &[Type]) -> BlockIdx {
        let id = self.blocks.insert(Block::new(arguments));
        self.blocks[id].id = Some(id);
        id
    }

    pub fn find_preds_for(&self, target_block: BlockIdx) -> Vec<(BlockIdx, Vec<Operand>)> {
        let mut preds = Vec::new();
        for (i, b) in self.blocks.iter() {
            match &b.terminator {
                Terminator::Ret(_) => {}
                Terminator::Br { block, arguments } => {
                    if block == &target_block {
                        preds.push((i, arguments.clone()))
                    }
                }
                Terminator::CondBr {
                    then_block,
                    else_block,
                    if_args,
                    then_args,
                    ..
                } => {
                    if then_block == &target_block {
                        preds.push((i, if_args.clone()))
                    }
                    if else_block == &target_block {
                        preds.push((i, then_args.clone()))
                    }
                }
            }
        }
        preds
    }
}
