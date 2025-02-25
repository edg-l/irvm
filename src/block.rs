use typed_generational_arena::{StandardArena, StandardIndex};

use crate::{types::{Type, VectorType}, value::Operand};


pub type BlockIdx = StandardIndex<Block>;
pub type InstIdx = StandardIndex<Instruction>;

#[derive(Debug, Clone)]
pub struct Block {
    pub instructions: StandardArena<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    BinaryOp(BinaryOp),
    BitwiseBinaryOp(BitwiseBinaryOp),
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Ret,
    Br(BlockIdx),
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
    ExtractElement {
        vector: Operand,
        idx: Operand,
    }
}
