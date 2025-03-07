pub mod llvm;

#[cfg(test)]
mod test {
    use std::error::Error;

    use irvm::{
        block::IcmpCond,
        common::Location,
        function::Parameter,
        module::Module,
        types::{Type, TypeStorage},
        value::Operand,
    };

    use crate::llvm::lower_module_to_llvmir;

    #[test]
    fn test_function_llvm() -> Result<(), Box<dyn Error>> {
        let mut module = Module::new("example");
        let mut storage = TypeStorage::new();
        let bool_ty = storage.add_type(Type::Int(1), Location::Unknown, Some("bool"));
        let i32_ty = storage.add_type(Type::Int(32), Location::Unknown, Some("i32"));
        let i64_ty = storage.add_type(Type::Int(64), Location::Unknown, Some("i64"));
        let ptr_ty = storage.add_type(Type::Ptr(None), Location::Unknown, Some("ptr"));

        let main_func = module
            .add_function(
                "main",
                &[Parameter::new(i32_ty, Location::Unknown)],
                i32_ty,
                Location::Unknown,
            )
            .get_id();
        let test_func = module
            .add_function(
                "test",
                &[Parameter::new(i32_ty, Location::Unknown)],
                i32_ty,
                Location::Unknown,
            )
            .get_id();

        let test_func_ret_ty = module.get_function(test_func).result_type;

        // main function
        {
            let func = module.get_function_mut(main_func);
            let param = func.param(0)?;
            let entry_block = func.entry_block;

            let value = func.blocks[entry_block].instr_add(
                &param,
                &Operand::const_int(4, i32_ty),
                Location::Unknown,
            )?;

            let then_block = func.add_block(&[]);
            let else_block = func.add_block(&[]);
            let final_block = func.add_block(&[i32_ty]);

            let cond = func.blocks[entry_block].instr_icmp(
                IcmpCond::Eq,
                value.clone(),
                Operand::const_int(6, i32_ty),
                Location::Unknown,
                &storage,
            )?;

            func.blocks[entry_block].instr_cond_jmp(
                then_block,
                else_block,
                &cond,
                &[],
                &[],
                Location::Unknown,
            );

            // then block
            {
                let value = func.blocks[then_block].instr_add(
                    &value,
                    &Operand::const_int(2, i32_ty),
                    Location::Unknown,
                )?;
                func.blocks[then_block].instr_jmp(final_block, &[value], Location::Unknown);
            }

            // else block
            {
                let value = func.blocks[else_block].instr_add(
                    &value,
                    &Operand::const_int(6, i32_ty),
                    Location::Unknown,
                )?;
                func.blocks[else_block].instr_jmp(final_block, &[value], Location::Unknown);
            }

            // final block
            {
                let param = func.blocks[final_block].arg(0)?;
                let value = func.blocks[final_block].instr_call(
                    test_func,
                    &[param],
                    test_func_ret_ty,
                    Location::Unknown,
                )?;
                func.blocks[final_block].instr_ret(Some(&value), Location::Unknown);
            }
        }

        // test functin
        {
            let func = module.get_function_mut(test_func);
            let value = func.param(0)?;
            func.entry_block()
                .instr_ret(Some(&value), Location::Unknown);
        }

        lower_module_to_llvmir(&module, &storage)?;

        Ok(())
    }
}
