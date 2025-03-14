use std::sync::Arc;

use typed_generational_arena::{StandardSlab, StandardSlabIndex};

use crate::common::Location;

pub type TypeIdx = StandardSlabIndex<TypeInfo>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub ty: Type,
    pub debug_info: Option<DebugTypeInfo>,
}

/// DWARF Debug info for the given type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DebugTypeInfo {
    pub name: String,
    // Define the type is a reference (in case the IR type used is a ptr).
    pub is_reference: bool,
    pub is_class: bool,
    pub class_inherits: Option<TypeIdx>,
    pub location: Location,
}

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
    Ptr {
        pointee: TypeIdx,
        address_space: Option<u32>,
    },
    Vector(VectorType),
    Array(ArrayType),
    Struct(Arc<StructType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayType {
    pub size: u64,
    pub ty: TypeIdx,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructType {
    pub packed: bool,
    pub ident: Option<String>,
    pub fields: Vec<TypeIdx>,
    pub debug_field_names: Vec<(String, Location)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorType {
    pub size: u32,
    pub ty: TypeIdx,
    pub vscale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub return_type: TypeIdx,
    pub parameters: Vec<TypeIdx>,
}

#[derive(Debug, Clone)]
pub struct TypeStorage {
    pub(crate) types: StandardSlab<TypeInfo>,
    pub(crate) i1_ty: Option<TypeIdx>,
    pub(crate) i64_ty: Option<TypeIdx>,
}

impl Default for TypeStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeStorage {
    pub fn new() -> Self {
        Self {
            types: StandardSlab::new(),
            i1_ty: None,
            i64_ty: None,
        }
    }

    pub fn i1_ty(&self) -> Option<TypeIdx> {
        self.i1_ty
    }

    pub fn i64_ty(&self) -> Option<TypeIdx> {
        self.i64_ty
    }

    pub fn add_type(&mut self, ty: Type, debug_name: Option<&str>) -> TypeIdx {
        self.add_type_ex(
            ty,
            debug_name.map(|name| DebugTypeInfo {
                name: name.to_string(),
                ..Default::default()
            }),
        )
    }

    pub fn add_type_ex(&mut self, ty: Type, debug_info: Option<DebugTypeInfo>) -> TypeIdx {
        let id = self.types.insert(TypeInfo {
            ty: ty.clone(),
            debug_info,
        });

        if let Type::Int(1) = ty {
            self.i1_ty = Some(id);
        }

        if let Type::Int(64) = ty {
            self.i64_ty = Some(id);
        }

        id
    }

    pub fn add_type_info(&mut self, ty: TypeInfo) -> TypeIdx {
        self.types.insert(ty)
    }

    pub fn get_type_info(&self, ty: TypeIdx) -> &TypeInfo {
        &self.types[ty]
    }
}
