use std::sync::Arc;

/// First class types
#[derive(Debug, Clone)]
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
    Struct(StructType),
    Opaque(Option<String>),
}

#[derive(Debug, Clone)]
pub struct ArrayType {
    pub size: u32,
    pub ty: Arc<Type>,
}

#[derive(Debug, Clone)]
pub struct StructType {
    pub packed: bool,
    pub ident: Option<String>,
    pub fields: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct VectorType {
    pub size: u32,
    pub ty: Arc<Type>,
}

#[derive(Debug, Clone)]
pub struct FunctionType {
    pub return_type: Type,
    pub parameters: Vec<Type>,
}
