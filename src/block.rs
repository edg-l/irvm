use typed_generational_arena::{StandardSlab, StandardSlabIndex};

use crate::{
    common::{CConv, Location},
    error::Error,
    function::{DebugVarIdx, FnIdx},
    module::TypeIdx,
    types::{FunctionType, Type, TypeStorage},
    value::Operand,
};

pub type BlockIdx = StandardSlabIndex<Block>;
pub type InstIdx = StandardSlabIndex<(Location, Instruction)>;

/// A Block that holds instructions executed in a sequence and a terminator for control flow.
///
/// Terminator default to Ret.
#[derive(Debug, Clone)]
pub struct Block {
    // The id is always set, but this is needed because first we need to
    // insert the block into the arena to get an id.
    pub(crate) id: Option<BlockIdx>,
    /// Arguments are made to model phi nodes.
    pub arguments: Vec<TypeIdx>,
    pub instructions: StandardSlab<(Location, Instruction)>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    BinaryOp(BinaryOp),
    BitwiseBinaryOp(BitwiseBinaryOp),
    VectorOp(VectorOp),
    MemoryOp(MemoryOp),
    OtherOp(OtherOp),
    DebugOp(DebugOp),
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Ret((Location, Option<Operand>)),
    Br {
        block: BlockIdx,
        arguments: Vec<Operand>,
        location: Location,
    },
    CondBr {
        then_block: BlockIdx,
        else_block: BlockIdx,
        cond: Operand,
        if_args: Vec<Operand>,
        then_args: Vec<Operand>,
        location: Location,
    },
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Sub {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Mul {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Div {
        lhs: Operand,
        rhs: Operand,
        signed: bool,
        /// If the exact keyword is present,
        /// the result value of the udiv is a poison value if %op1
        /// is not a multiple of %op2 (as such, “((a udiv exact b) mul b) == a”).
        exact: bool,
    },
    Rem {
        lhs: Operand,
        rhs: Operand,
        signed: bool,
    },
    FAdd {
        lhs: Operand,
        rhs: Operand,
        // todo: fast math flags
    },
    FSub {
        lhs: Operand,
        rhs: Operand,
        // todo: fast math flags
    },
    FMul {
        lhs: Operand,
        rhs: Operand,
        // todo: fast math flags
    },
    FDiv {
        lhs: Operand,
        rhs: Operand,
        // todo: fast math flags
    },
    FRem {
        lhs: Operand,
        rhs: Operand,
        // todo: fast math flags
    },
}

#[derive(Debug, Clone)]
pub enum BitwiseBinaryOp {
    Shl {
        lhs: Operand,
        rhs: Operand,
    },
    Lshr {
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    },
    Ashr {
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    },
    And {
        lhs: Operand,
        rhs: Operand,
    },
    Or {
        lhs: Operand,
        rhs: Operand,
        disjoint: bool,
    },
    Xor {
        lhs: Operand,
        rhs: Operand,
    },
}

#[derive(Debug, Clone)]
pub enum VectorOp {
    ExtractElement { vector: Operand, idx: Operand },
}

#[derive(Debug, Clone)]
pub enum MemoryOp {
    Alloca {
        ty: TypeIdx,
        num_elements: u32,
        inalloca: bool,
        align: Option<u32>,
        addr_space: Option<u32>,
    },
}

#[derive(Debug, Clone)]
pub enum OtherOp {
    Call(CallOp),
    Icmp {
        cond: IcmpCond,
        lhs: Operand,
        rhs: Operand,
    },
    Fcmp {
        cond: FcmpCond,
        lhs: Operand,
        rhs: Operand,
    },
}

#[derive(Debug, Clone)]
pub struct CallOp {
    pub tail: bool,
    pub musttail: bool,
    pub notail: bool,
    /// Must match the target fn cconv or ub.
    pub cconv: CConv,
    pub params: Vec<Operand>,
    pub ret_ty: TypeIdx,
    pub ret_attrs: Option<CallReturnAttrs>,
    pub addr_space: Option<u32>,
    /// Only needed if its a varargs function.
    pub fn_ty: Option<FunctionType>,
    pub fn_target: CallableValue,
}

#[derive(Debug, Clone, Copy)]
pub enum IcmpCond {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

#[derive(Debug, Clone, Copy)]
pub enum FcmpCond {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

#[derive(Debug, Clone)]
pub enum CallableValue {
    Symbol(FnIdx),
    Pointer(Operand, FunctionType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallReturnAttrs {
    pub zeroext: bool,
    pub signext: bool,
    pub noext: bool,
    pub inreg: bool,
}

/// A debug operation.
#[derive(Debug, Clone)]
pub enum DebugOp {
    /// Provides information about a local element (e.g., variable).
    Declare {
        /// Must be of pointer type. Usually used for variables allocated in a alloca or behind a pointer.
        address: Operand,
        variable: DebugVarIdx,
    },
    /// Provides information when a user source variable is set to a new value.
    ///
    /// Useful for variables whose address can't be taken.
    Value {
        new_value: Operand,
        variable: DebugVarIdx,
    },
    /// This marks the position where a source assignment occurred. It encodes the value of the variable.
    /// It references the store, if any, that performs the assignment, and the destination address.
    Assign {
        address: Operand,
        new_value: Operand,
        store_inst: InstIdx,
        variable: DebugVarIdx,
    },
}

/// Debug info for a variable.
#[derive(Debug, Clone)]
pub struct DebugVariable {
    /// The name.
    pub name: String,
    /// If it's a parameter, the number.
    pub parameter: Option<u32>,
    /// The type of the variable.
    pub ty: TypeIdx,
    /// The source location.
    pub location: Location,
}

macro_rules! binop_float {
    ($name:ident, $variant:ident) => {
        pub fn $name(
            &mut self,
            lhs: Operand,
            rhs: Operand,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.instructions.insert((
                location,
                Instruction::BinaryOp(BinaryOp::$variant { lhs, rhs }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }
    };
}

macro_rules! binop_with_overflow_flags {
    ($name:ident, $name_ex:ident, $variant:ident) => {
        pub fn $name(
            &mut self,
            lhs: &Operand,
            rhs: &Operand,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.instructions.insert((
                location,
                Instruction::BinaryOp(BinaryOp::$variant {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    nsw: false,
                    nuw: false,
                }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }

        pub fn $name_ex(
            &mut self,
            lhs: &Operand,
            rhs: &Operand,
            nsw: bool,
            nuw: bool,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.instructions.insert((
                location,
                Instruction::BinaryOp(BinaryOp::$variant {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    nsw,
                    nuw,
                }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }
    };
}

impl Block {
    pub(crate) fn new(arguments: &[TypeIdx]) -> Self {
        Self {
            instructions: StandardSlab::new(),
            terminator: Terminator::Ret((Location::Unknown, None)),
            arguments: arguments.to_vec(),
            id: None,
        }
    }

    pub fn id(&self) -> BlockIdx {
        self.id.unwrap()
    }

    pub fn arg(&self, nth: usize) -> Result<Operand, Error> {
        self.arguments
            .get(nth)
            .map(|x| Operand::BlockArgument {
                block_idx: self.id.unwrap().to_idx(),
                nth,
                ty: *x,
            })
            .ok_or_else(|| Error::BlockArgNotFound {
                block_id: self.id(),
                nth,
            })
    }

    pub fn instr_ret(&mut self, value: Option<&Operand>, location: Location) {
        self.terminator = Terminator::Ret((location, value.cloned()));
    }

    pub fn instr_jmp(&mut self, target: BlockIdx, arguments: &[Operand], location: Location) {
        self.terminator = Terminator::Br {
            block: target,
            arguments: arguments.to_vec(),
            location,
        };
    }

    pub fn instr_cond_jmp(
        &mut self,
        then_block: BlockIdx,
        else_block: BlockIdx,
        cond: &Operand,
        then_block_args: &[Operand],
        else_block_args: &[Operand],
        location: Location,
    ) {
        self.terminator = Terminator::CondBr {
            then_block,
            else_block,
            cond: cond.clone(),
            if_args: then_block_args.to_vec(),
            then_args: else_block_args.to_vec(),
            location,
        };
    }

    binop_with_overflow_flags!(instr_add, instr_add_ex, Add);
    binop_with_overflow_flags!(instr_sub, instr_sub_ex, Sub);
    binop_with_overflow_flags!(instr_mul, instr_mul_ex, Mul);

    pub fn instr_div(
        &mut self,
        lhs: &Operand,
        rhs: &Operand,
        signed: bool,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BinaryOp(BinaryOp::Div {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                signed,
                exact,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_rem(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        signed: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BinaryOp(BinaryOp::Rem { lhs, rhs, signed }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    binop_float!(instr_fadd, FAdd);
    binop_float!(instr_fsub, FSub);
    binop_float!(instr_fmul, FMul);
    binop_float!(instr_fdiv, FDiv);
    binop_float!(instr_frem, FRem);

    pub fn instr_shl(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Shl { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_lshr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Lshr { lhs, rhs, exact }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_ashr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Ashr { lhs, rhs, exact }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_and(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::And { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_or(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        disjoint: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Or { lhs, rhs, disjoint }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_xor(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Xor { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_alloca(
        &mut self,
        pointer_type_idx: TypeIdx,
        num_elements: u32,
        align: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(pointer_type_idx);
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            panic!("invalid pointer type")
        };
        let idx = self.instructions.insert((
            location,
            Instruction::MemoryOp(MemoryOp::Alloca {
                ty: inner,
                num_elements,
                inalloca: false,
                align,
                addr_space: None,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, pointer_type_idx))
    }

    pub fn instr_alloca_ex(
        &mut self,
        pointer_type_idx: TypeIdx,
        num_elements: u32,
        align: Option<u32>,
        inalloca: bool,
        addr_space: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(pointer_type_idx);
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            panic!("invalid pointer type")
        };

        let idx = self.instructions.insert((
            location,
            Instruction::MemoryOp(MemoryOp::Alloca {
                ty: inner,
                num_elements,
                inalloca,
                align,
                addr_space,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, pointer_type_idx))
    }

    pub fn instr_call(
        &mut self,
        fn_idx: FnIdx,
        params: &[Operand],
        ret_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.instructions.insert((
            location,
            Instruction::OtherOp(OtherOp::Call(CallOp {
                tail: false,
                musttail: false,
                notail: false,
                cconv: CConv::default(),
                params: params.to_vec(),
                ret_ty,
                ret_attrs: None,
                addr_space: None,
                fn_ty: None,
                fn_target: CallableValue::Symbol(fn_idx),
            })),
        ));

        Ok(Operand::Value(self.id(), idx, ret_ty))
    }

    pub fn instr_call_ex(&mut self, call_op: CallOp, location: Location) -> Result<Operand, Error> {
        let ret_ty = call_op.ret_ty;
        let idx = self
            .instructions
            .insert((location, Instruction::OtherOp(OtherOp::Call(call_op))));

        Ok(Operand::Value(self.id(), idx, ret_ty))
    }

    pub fn instr_icmp(
        &mut self,
        cond: IcmpCond,
        lhs: Operand,
        rhs: Operand,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type_idx = lhs.get_type();
        let idx = self.instructions.insert((
            location,
            Instruction::OtherOp(OtherOp::Icmp { cond, lhs, rhs }),
        ));

        let result_type = type_storage.get_type_info(result_type_idx);
        if let Type::Vector(_) = result_type.ty {
            Ok(Operand::Value(self.id(), idx, result_type_idx))
        } else {
            Ok(Operand::Value(
                self.id(),
                idx,
                type_storage.i1_ty.expect("i1 type missing"),
            ))
        }
    }
}
