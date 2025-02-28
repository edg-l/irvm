pub mod llvm;

#[cfg(test)]
mod test {
    use std::error::Error;

    use irvm::{block::IcmpCond, function::Parameter, module::Module, types::Type, value::Operand};

    use crate::llvm::lower_module_to_llvmir;

    #[test]
    fn test_function_llvm() -> Result<(), Box<dyn Error>> {
        let mut module = Module::new("example");

        // todo: make arenas use option and reserve ids.
        let main_func = module
            .add_function("main", &[Parameter::new(Type::Int(32))], Type::Int(32))
            .get_id();
        let test_func = module
            .add_function("test", &[Parameter::new(Type::Int(32))], Type::Int(32))
            .get_id();

        {
            let func = module.get_function_mut(main_func);
            let param = func.param(0)?;
            let entry_block = func.entry_block;

            let value = func.blocks[entry_block].instr_add(&param, &Operand::const_i32(4))?;

            let then_block = func.add_block(&[]);
            let else_block = func.add_block(&[]);
            let final_block = func.add_block(&[Type::Int(32)]);

            let cond = func.blocks[entry_block].instr_icmp(
                IcmpCond::Eq,
                value.clone(),
                Operand::const_i32(6),
            )?;

            func.blocks[entry_block].instr_cond_jmp(then_block, else_block, &cond, &[], &[]);

            {
                let value = func.blocks[then_block].instr_add(&value, &Operand::const_i32(2))?;
                func.blocks[then_block].instr_jmp(final_block, &[value]);
            }

            {
                let value = func.blocks[else_block].instr_add(&value, &Operand::const_i32(6))?;
                func.blocks[else_block].instr_jmp(final_block, &[value]);
            }

            {
                let param = func.blocks[final_block].arg(0)?;
                let ret_ty = module.get_function(test_func).result_type.clone();
                let func = module.get_function_mut(main_func);
                let value = func.blocks[final_block].instr_call(test_func, &[param], &ret_ty)?;
                func.blocks[final_block].instr_ret(Some(&value));
            }
        }

        {
            let func = module.get_function_mut(test_func);
            let value = func.param(0)?;
            func.entry_block().instr_ret(Some(&value));
        }

        lower_module_to_llvmir(&module)?;

        Ok(())
    }
}
