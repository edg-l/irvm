use crate::{
    block::{BlockIdx, InstIdx},
    types::Type,
};

#[derive(Debug, Clone)]
pub enum Operand {
    Parameter(usize, Type),
    BlockArgument {
        block_idx: usize,
        nth: usize,
        ty: Type,
    },
    Value(BlockIdx, InstIdx, Type),
    Constant(ConstValue, Type),
}

impl Operand {
    /// Gets the type of this operand.
    pub fn get_type(&self) -> &Type {
        match self {
            Operand::Parameter(_, ty) => ty,
            Operand::BlockArgument { ty, .. } => ty,
            Operand::Value(_, _, ty) => ty,
            Operand::Constant(_, ty) => ty,
        }
    }

    pub fn const_int(value: u64, width: u32) -> Self {
        Self::Constant(ConstValue::Int(value), Type::Int(width))
    }

    pub fn const_bool(value: bool) -> Self {
        Self::Constant(ConstValue::Bool(value), Type::Int(1))
    }

    pub fn const_i8(value: u64) -> Self {
        Self::Constant(ConstValue::Int(value), Type::Int(8))
    }

    pub fn const_i16(value: u64) -> Self {
        Self::Constant(ConstValue::Int(value), Type::Int(16))
    }

    pub fn const_i32(value: u64) -> Self {
        Self::Constant(ConstValue::Int(value), Type::Int(32))
    }

    pub fn const_i64(value: u64) -> Self {
        Self::Constant(ConstValue::Int(value), Type::Int(64))
    }

    pub fn const_f32(value: f32) -> Self {
        Self::Constant(ConstValue::Float(value as f64), Type::Float)
    }

    pub fn const_f64(value: f64) -> Self {
        Self::Constant(ConstValue::Float(value), Type::Double)
    }
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    Bool(bool),
    Int(u64),
    Float(f64),
    Array(Vec<ConstValue>),
    Vector(Vec<ConstValue>),
    Struct(Vec<ConstValue>),
    NullPtr,
    Undef,
    Poison,
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match self {
            ConstValue::Bool(_) => matches!(
                other,
                ConstValue::Bool(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Int(_) => matches!(
                other,
                ConstValue::Int(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Float(_) => matches!(
                other,
                ConstValue::Float(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Array(_) => matches!(
                other,
                ConstValue::Array(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Vector(_) => matches!(
                other,
                ConstValue::Vector(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Struct(_) => matches!(
                other,
                ConstValue::Struct(_) | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::NullPtr => matches!(
                other,
                ConstValue::NullPtr | ConstValue::Undef | ConstValue::Poison
            ),
            ConstValue::Undef => true,
            ConstValue::Poison => true,
        }
    }
}

impl Eq for ConstValue {}
