pub mod llvm;

#[cfg(test)]
mod test {
    use std::error::Error;

    use irvm::{
        block::IcmpCond,
        function::{Function, Parameter},
        module::Module,
        types::Type,
        value::Operand,
    };

    use crate::llvm::lower_module_to_llvmir;

    #[test]
    fn test_function_llvm() -> Result<(), Box<dyn Error>> {
        let mut module = Module::new("example");

        // todo: make arenas use option and reserve ids.
        let func = module
            .add_function("main", &[Parameter::new(Type::Int(32))], Type::Int(32))
            .get_id();
        let func2 = module
            .add_function("test", &[Parameter::new(Type::Int(32))], Type::Int(32))
            .get_id();

        {
            let param = module.func(func).param(0)?;
            let entry_block = module.func(func).entry_block;

            let value =
                module.func(func).blocks[entry_block].instr_add(&param, &Operand::const_i32(4))?;

            let then_block = module.func(func).add_block(&[]);
            let else_block = module.func(func).add_block(&[]);
            let final_block = module.func(func).add_block(&[Type::Int(32)]);

            let cond = module.func(func).blocks[entry_block].instr_icmp(
                IcmpCond::Eq,
                value.clone(),
                Operand::const_i32(6),
            )?;

            module.func(func).blocks[entry_block].instr_cond_jmp(
                then_block,
                else_block,
                &cond,
                &[],
                &[],
            );

            {
                let value = module.func(func).blocks[then_block]
                    .instr_add(&value, &Operand::const_i32(2))?;
                module.func(func).blocks[then_block].instr_jmp(final_block, &[value]);
            }

            {
                let value = module.func(func).blocks[else_block]
                    .instr_add(&value, &Operand::const_i32(6))?;
                module.func(func).blocks[else_block].instr_jmp(final_block, &[value]);
            }

            {
                let param = module.func(func).blocks[final_block].arg(0)?;
                let ret_ty = module.get_function(func2).result_type.clone();
                let value =
                    module.func(func).blocks[final_block].instr_call(func2, &[param], &ret_ty)?;
                module.func(func).blocks[final_block].instr_ret(Some(&value));
            }
        }

        {
            let func = module.get_function_mut(func2);
            let value = func.param(0)?;
            func.entry_block().instr_ret(Some(&value));
        }

        lower_module_to_llvmir(&module)?;

        Ok(())
    }
}
