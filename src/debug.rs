use crate::types::Type;

pub trait DebugNameProvider {
    fn get_name_for_type(&self, ty: &Type) -> String;
}

pub struct DefaultDebugNameProvider;

impl DebugNameProvider for DefaultDebugNameProvider {
    fn get_name_for_type(&self, ty: &Type) -> String {
        match ty {
            Type::Int(w) => format!("i{w}"),
            Type::Half => todo!(),
            Type::BFloat => todo!(),
            Type::Float => todo!(),
            Type::Double => todo!(),
            Type::Fp128 => todo!(),
            Type::X86Fp80 => todo!(),
            Type::PpcFp128 => todo!(),
            Type::Ptr(_) => todo!(),
            Type::Vector(_vector_type) => todo!(),
            Type::Array(_array_type) => todo!(),
            Type::Struct(_struct_type) => todo!(),
        }
    }
}
