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
    // If the type is a int, specify if it's signed.
    pub is_signed: bool,
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

    /// Get or create the i1 type (boolean).
    /// This is useful for icmp which returns i1.
    pub fn get_or_create_i1(&mut self) -> TypeIdx {
        if let Some(ty) = self.i1_ty {
            ty
        } else {
            self.add_type(Type::Int(1), None)
        }
    }

    /// Get or create the i64 type.
    pub fn get_or_create_i64(&mut self) -> TypeIdx {
        if let Some(ty) = self.i64_ty {
            ty
        } else {
            self.add_type(Type::Int(64), None)
        }
    }

    /// Compute the result type of a GEP (GetElementPtr) operation.
    ///
    /// Given a pointer type and a sequence of indices, this function computes
    /// the type of the resulting pointer. The result is always a pointer type.
    ///
    /// # Arguments
    /// * `ptr_ty` - The type of the pointer operand (must be a Ptr type)
    /// * `indices` - The GEP indices. For struct indices, use `GepIndexKind::Const`.
    ///
    /// # Returns
    /// The TypeIdx of the result pointer type, or an error if the GEP is invalid.
    ///
    /// # GEP Semantics
    /// - The first index offsets within the pointee type (like array indexing)
    /// - Subsequent indices descend into aggregate types:
    ///   - Array: descends to element type (index can be dynamic)
    ///   - Struct: descends to field type (index must be constant)
    ///   - Vector: descends to element type (index can be dynamic)
    pub fn compute_gep_result_type(
        &mut self,
        ptr_ty: TypeIdx,
        indices: &[GepIndexKind],
    ) -> Result<TypeIdx, GepTypeError> {
        let ptr_info = self.get_type_info(ptr_ty);
        let (pointee, addr_space) = match &ptr_info.ty {
            Type::Ptr {
                pointee,
                address_space,
            } => (*pointee, *address_space),
            _ => return Err(GepTypeError::NotAPointer),
        };

        if indices.is_empty() {
            return Err(GepTypeError::NoIndices);
        }

        // First index just offsets within the pointee array, doesn't change type
        let mut current_ty = pointee;

        // Process remaining indices
        for (i, idx) in indices.iter().skip(1).enumerate() {
            let ty_info = self.get_type_info(current_ty);
            current_ty = match &ty_info.ty {
                Type::Array(array_ty) => array_ty.ty,
                Type::Struct(struct_ty) => {
                    let field_idx = match idx {
                        GepIndexKind::Const(n) => *n,
                        GepIndexKind::Dynamic => {
                            return Err(GepTypeError::DynamicStructIndex { index: i + 1 });
                        }
                    };
                    *struct_ty.fields.get(field_idx).ok_or(
                        GepTypeError::StructFieldOutOfBounds {
                            index: i + 1,
                            field: field_idx,
                            num_fields: struct_ty.fields.len(),
                        },
                    )?
                }
                Type::Vector(vec_ty) => vec_ty.ty,
                _ => {
                    return Err(GepTypeError::CannotIndex {
                        index: i + 1,
                        ty: format!("{:?}", ty_info.ty),
                    });
                }
            };
        }

        // The result is a pointer to the final type
        let result_ty = self.add_type(
            Type::Ptr {
                pointee: current_ty,
                address_space: addr_space,
            },
            None,
        );

        Ok(result_ty)
    }
}

/// Index kind for GEP type computation.
#[derive(Debug, Clone, Copy)]
pub enum GepIndexKind {
    /// A constant index value.
    Const(usize),
    /// A dynamic (non-constant) index.
    Dynamic,
}

/// Errors that can occur during GEP type computation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GepTypeError {
    #[error("GEP base must be a pointer type")]
    NotAPointer,
    #[error("GEP requires at least one index")]
    NoIndices,
    #[error("struct index at position {index} must be constant")]
    DynamicStructIndex { index: usize },
    #[error(
        "struct field index {field} out of bounds (struct has {num_fields} fields) at index position {index}"
    )]
    StructFieldOutOfBounds {
        index: usize,
        field: usize,
        num_fields: usize,
    },
    #[error("cannot index into type {ty} at position {index}")]
    CannotIndex { index: usize, ty: String },
}
