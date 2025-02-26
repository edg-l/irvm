use crate::{block::InstIdx, types::Type};

#[derive(Debug, Clone)]
pub enum Operand {
    Parameter(usize, Type),
    Value(InstIdx, Type),
    Constant(ConstValue, Type),
}

impl Operand {
    /// Gets the type of this operand.
    pub fn get_type(&self) -> &Type {
        match self {
            Operand::Parameter(_, ty) => ty,
            Operand::Value(_, ty) => ty,
            Operand::Constant(_, ty) => ty,
        }
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
