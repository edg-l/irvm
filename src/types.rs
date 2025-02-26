use std::sync::Arc;

/// First class types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int(u32),
    Half,
    BFloat,
    Float,
    Double,
    Fp128,
    X86Fp80,
    PpcFp128,
    Ptr(Option<u32>),
    Vector(VectorType),
    Array(ArrayType),
    Struct(Arc<StructType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayType {
    pub size: u64,
    pub ty: Arc<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructType {
    pub packed: bool,
    pub ident: Option<String>,
    pub fields: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorType {
    pub size: u32,
    pub ty: Arc<Type>,
    pub vscale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub return_type: Type,
    pub parameters: Vec<Type>,
}
