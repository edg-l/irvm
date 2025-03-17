use crate::{
    block::{BlockIdx, InstIdx},
    error::Error,
    types::{TypeIdx, TypeStorage},
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

    pub fn get_inner_type(&self, storage: &TypeStorage) -> Result<TypeIdx, Error> {
        let ty = self.get_type();
        let ty_info = storage.get_type_info(ty);

        let opt = match &ty_info.ty {
            crate::types::Type::Int(_) => None,
            crate::types::Type::Half => None,
            crate::types::Type::BFloat => None,
            crate::types::Type::Float => None,
            crate::types::Type::Double => None,
            crate::types::Type::Fp128 => None,
            crate::types::Type::X86Fp80 => None,
            crate::types::Type::PpcFp128 => None,
            crate::types::Type::Ptr { pointee, .. } => Some(*pointee),
            crate::types::Type::Vector(vector_type) => Some(vector_type.ty),
            crate::types::Type::Array(array_type) => Some(array_type.ty),
            crate::types::Type::Struct(..) => None,
        };

        opt.ok_or(Error::InnerTypeNotFound)
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
