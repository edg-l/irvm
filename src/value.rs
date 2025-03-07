use crate::{
    block::{BlockIdx, InstIdx},
    module::TypeIdx,
};

#[derive(Debug, Clone)]
pub enum Operand {
    Parameter(usize, TypeIdx),
    BlockArgument {
        block_idx: usize,
        nth: usize,
        ty: TypeIdx,
    },
    Value(BlockIdx, InstIdx, TypeIdx),
    Constant(ConstValue, TypeIdx),
}

impl Operand {
    /// Gets the type of this operand.
    pub fn get_type(&self) -> TypeIdx {
        match self {
            Operand::Parameter(_, ty) => *ty,
            Operand::BlockArgument { ty, .. } => *ty,
            Operand::Value(_, _, ty) => *ty,
            Operand::Constant(_, ty) => *ty,
        }
    }

    pub fn const_int(value: u64, ty: TypeIdx) -> Self {
        Self::Constant(ConstValue::Int(value), ty)
    }

    pub fn const_f32(value: f32, ty: TypeIdx) -> Self {
        Self::Constant(ConstValue::Float(value as f64), ty)
    }

    pub fn const_f64(value: f64, ty: TypeIdx) -> Self {
        Self::Constant(ConstValue::Float(value), ty)
    }
}

#[derive(Debug, Clone)]
pub enum ConstValue {
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
