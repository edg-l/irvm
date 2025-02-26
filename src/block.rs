use typed_generational_arena::{StandardArena, StandardIndex};

use crate::{
    common::CConv,
    error::Error,
    types::{FunctionType, Type},
    value::Operand,
};

pub type BlockIdx = StandardIndex<Block>;
pub type InstIdx = StandardIndex<Instruction>;

/// A Block.
///
/// Terminator default to Ret.
#[derive(Debug, Clone)]
pub struct Block {
    /// Arguments are made to model phi nodes.
    pub arguments: Vec<Operand>,
    pub instructions: StandardArena<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    BinaryOp(BinaryOp),
    BitwiseBinaryOp(BitwiseBinaryOp),
    VectorOp(VectorOp),
    MemoryOp(MemoryOp),
    OtherOp(OtherOp),
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Ret(Option<Operand>),
    Br {
        block: BlockIdx,
        arguments: Vec<Operand>,
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
        nsw: bool,
        nuw: bool,
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
        ty: Type,
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
    pub ret_ty: Type,
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
    Symbol(String),
    Pointer(Operand),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallReturnAttrs {
    pub zeroext: bool,
    pub signext: bool,
    pub noext: bool,
    pub inreg: bool,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            instructions: StandardArena::new(),
            terminator: Terminator::Ret(None),
            arguments: Vec::new(),
        }
    }
}

macro_rules! binop_float {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self, lhs: Operand, rhs: Operand) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type().clone(),
                    expected: lhs.get_type().clone(),
                });
            }

            let result_type = lhs.get_type().clone();
            let idx = self
                .instructions
                .insert(Instruction::BinaryOp(BinaryOp::$variant { lhs, rhs }));

            Ok(Operand::Value(idx, result_type))
        }
    };
}

macro_rules! binop_with_overflow_flags {
    ($name:ident, $variant:ident) => {
        pub fn $name(
            &mut self,
            lhs: Operand,
            rhs: Operand,
            nsw: bool,
            nuw: bool,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type().clone(),
                    expected: lhs.get_type().clone(),
                });
            }

            let result_type = lhs.get_type().clone();
            let idx = self
                .instructions
                .insert(Instruction::BinaryOp(BinaryOp::$variant {
                    lhs,
                    rhs,
                    nsw,
                    nuw,
                }));

            Ok(Operand::Value(idx, result_type))
        }
    };
}

impl Block {
    pub fn new(arguments: Vec<Operand>) -> Self {
        Self {
            instructions: StandardArena::new(),
            terminator: Terminator::Ret(None),
            arguments,
        }
    }

    binop_with_overflow_flags!(instr_add, Add);
    binop_with_overflow_flags!(instr_sub, Sub);
    binop_with_overflow_flags!(instr_mul, Mul);

    pub fn instr_div(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        signed: bool,
        exact: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BinaryOp(BinaryOp::Div {
                lhs,
                rhs,
                signed,
                exact,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_rem(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        signed: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BinaryOp(BinaryOp::Rem { lhs, rhs, signed }));

        Ok(Operand::Value(idx, result_type))
    }

    binop_float!(inst_fadd, FAdd);
    binop_float!(inst_fsub, FSub);
    binop_float!(inst_fmul, FMul);
    binop_float!(inst_fdiv, FDiv);
    binop_float!(inst_frem, FRem);

    pub fn instr_shl(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Shl {
                lhs,
                rhs,
                nsw,
                nuw,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_lshr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Lshr {
                lhs,
                rhs,
                exact,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_ashr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Ashr {
                lhs,
                rhs,
                exact,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_and(&mut self, lhs: Operand, rhs: Operand) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::And {
                lhs,
                rhs,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_or(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        disjoint: bool,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Or {
                lhs,
                rhs,
                disjoint,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_xor(&mut self, lhs: Operand, rhs: Operand) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type().clone(),
                expected: lhs.get_type().clone(),
            });
        }

        let result_type = lhs.get_type().clone();
        let idx = self
            .instructions
            .insert(Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Xor {
                lhs,
                rhs,
            }));

        Ok(Operand::Value(idx, result_type))
    }

    pub fn instr_alloca(
        &mut self,
        ty: Type,
        num_elements: u32,
        align: Option<u32>,
    ) -> Result<Operand, Error> {
        let idx = self
            .instructions
            .insert(Instruction::MemoryOp(MemoryOp::Alloca {
                ty,
                num_elements,
                inalloca: false,
                align,
                addr_space: None,
            }));

        Ok(Operand::Value(idx, Type::Ptr(None)))
    }

    pub fn instr_alloca_ex(
        &mut self,
        ty: Type,
        num_elements: u32,
        align: Option<u32>,
        inalloca: bool,
        addr_space: Option<u32>,
    ) -> Result<Operand, Error> {
        let idx = self
            .instructions
            .insert(Instruction::MemoryOp(MemoryOp::Alloca {
                ty,
                num_elements,
                inalloca,
                align,
                addr_space,
            }));

        Ok(Operand::Value(idx, Type::Ptr(None)))
    }

    pub fn instr_call(
        &mut self,
        symbol: &str,
        params: &[Operand],
        ret_ty: &Type,
    ) -> Result<Operand, Error> {
        let idx = self
            .instructions
            .insert(Instruction::OtherOp(OtherOp::Call(CallOp {
                tail: false,
                musttail: false,
                notail: false,
                cconv: CConv::default(),
                params: params.to_vec(),
                ret_ty: ret_ty.clone(),
                ret_attrs: None,
                addr_space: None,
                fn_ty: None,
                fn_target: CallableValue::Symbol(symbol.to_string()),
            })));

        Ok(Operand::Value(idx, ret_ty.clone()))
    }

    pub fn instr_call_ex(&mut self, call_op: CallOp) -> Result<Operand, Error> {
        let ret_ty = call_op.ret_ty.clone();
        let idx = self
            .instructions
            .insert(Instruction::OtherOp(OtherOp::Call(call_op)));

        Ok(Operand::Value(idx, ret_ty))
    }
}
