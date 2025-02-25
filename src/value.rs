use crate::{block::InstIdx, types::Type};


#[derive(Debug, Clone)]
pub enum Operand {
    Parameter(usize, Type),
    Value(InstIdx, Type),
    Constant(ConstValue, Type),
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
}
