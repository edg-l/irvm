pub mod llvm;

#[cfg(test)]
mod test {
    use std::error::Error;

    use irvm::{
        block::Terminator, datalayout::DataLayout, function::{Function, Parameter}, module::Module, target_lexicon::Triple, types::Type, value::{ConstValue, Operand}
    };
    use typed_generational_arena::StandardArena;

    use crate::llvm::lower_module;

    #[test]
    fn test_function() -> Result<(), Box<dyn Error>> {
        let mut module = Module {
            name: "example".to_string(),
            data_layout: DataLayout::default(),
            target_triple: Triple::host(),
            functions: StandardArena::new(),
        };

        let mut func = Function::new("main", &[Parameter::new(Type::Int(32))], Type::Int(32));

        let value = func.entry_block().instr_add(
            Operand::Parameter(0, Type::Int(32)),
            Operand::Constant(ConstValue::Int(4), Type::Int(32)),
            false,
            false,
        )?;

        func.entry_block().terminator = Terminator::Ret(Some(value));

        module.functions.insert(func);

        lower_module(&module)?;

        Ok(())
    }
}
