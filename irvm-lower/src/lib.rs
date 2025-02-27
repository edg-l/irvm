pub mod llvm;

#[cfg(test)]
mod test {
    use std::error::Error;

    use irvm::{
        block::{Block, IcmpCond},
        function::{Function, Parameter},
        module::Module,
        types::Type,
        value::Operand,
    };

    use crate::llvm::lower_module;

    #[test]
    fn test_function() -> Result<(), Box<dyn Error>> {
        let mut module = Module::new("example");

        let mut func = Function::new("main", &[Parameter::new(Type::Int(32))], Type::Int(32));

        {
            let param = func.param(0).unwrap();
            let entry_block = func.entry_block;

            let value = func.blocks[entry_block].instr_add(param, Operand::const_i32(4))?;

            func.blocks[entry_block].instr_ret(Some(value.clone()));

            let then_block = func.add_block(Block::new(&[]));
            let else_block = func.add_block(Block::new(&[]));
            let final_block = func.add_block(Block::new(&[Type::Int(32)]));

            let cond =
                func.blocks[entry_block].instr_icmp(IcmpCond::Eq, value, Operand::const_i32(6))?;

            func.blocks[entry_block].instr_cond_jmp(then_block, else_block, cond, &[], &[]);
        }

        module.functions.insert(func);

        lower_module(&module)?;

        Ok(())
    }
}
