//! ## irmv-lower
//!
//! Lower irvm IR to other IRs.
//!
//! This crates currently only allows you to lower irmv IR to LLVM IR.
//!
//! See the [`llvm`] submodule for more information.
//!
//! ```bash
//! cargo add irvm-lower
//! ```
//!

pub mod llvm;

#[cfg(test)]
mod test {

    use irvm::{
        block::{GepIndex, IcmpCond},
        common::Location,
        function::Parameter,
        module::Module,
        types::{StructType, Type, TypeStorage},
        value::Operand,
    };

    use crate::llvm::{JitValue, create_jit_engine, lower_module_to_llvmir};

    #[test]
    fn test_function_llvm() -> Result<(), Box<dyn std::error::Error>> {
        let mut module = Module::new("example", Location::unknown());
        let mut storage = TypeStorage::new();
        let _bool_ty = storage.add_type(Type::Int(1), Some("bool"));
        let i32_ty = storage.add_type(Type::Int(32), Some("u32"));
        let _i64_ty = storage.add_type(Type::Int(64), Some("u64"));
        let _ptr_ty = storage.add_type(
            Type::Ptr {
                pointee: i32_ty,
                address_space: None,
            },
            Some("*i32"),
        );

        let main_func = module
            .add_function(
                "main",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        let test_func = module
            .add_function(
                "test",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();

        let test_func_ret_ty = module.get_function(test_func).result_type;

        // main function
        {
            let func = module.get_function_mut(main_func);
            let param = func.param(0)?;
            let param_dbg = func.create_debug_var_param("argv", i32_ty, 0, &Location::Unknown);
            let entry_block = func.entry_block;

            let value = func.blocks[entry_block].instr_add(
                &param,
                &Operand::const_int(4, i32_ty),
                Location::Unknown,
            )?;

            func.blocks[entry_block].instr_dbg_value(
                value.clone(),
                param_dbg,
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
                    test_func_ret_ty.unwrap(),
                    Location::Unknown,
                )?;
                func.blocks[final_block].instr_ret(Some(&value), Location::Unknown);
            }
        }

        // test function
        {
            let func = module.get_function_mut(test_func);
            let value = func.param(0)?;
            func.entry_block()
                .instr_ret(Some(&value), Location::Unknown);
        }

        let result = test_run_module(
            &module,
            &storage,
            "main",
            &[JitValue::U32(4)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(14));

        Ok(())
    }

    #[test]
    fn test_struct() -> Result<(), Box<dyn std::error::Error>> {
        let mut module = Module::new("example", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), Some("u32"));
        let ptr_ty = storage.add_type(
            Type::Ptr {
                pointee: i32_ty,
                address_space: None,
            },
            Some("*u32"),
        );

        let strct_type = storage.add_type(
            Type::Struct(
                StructType {
                    packed: false,
                    ident: None,
                    fields: vec![ptr_ty, i32_ty],
                    debug_field_names: vec![
                        ("ptr".to_string(), Location::unknown()),
                        ("x".to_string(), Location::unknown()),
                    ],
                }
                .into(),
            ),
            Some("hello"),
        );

        let func = module
            .add_function(
                "example",
                &[
                    Parameter::new(i32_ty, Location::Unknown),
                    Parameter::new(strct_type, Location::Unknown),
                ],
                None,
                Location::Unknown,
            )
            .get_id();

        let func = module.get_function_mut(func);

        let entry = func.entry_block;

        func.blocks[entry].instr_ret(None, Location::Unknown);

        let ir = lower_module_to_llvmir(&module, &storage)?;

        ir.dump();

        Ok(())
    }

    #[test]
    fn test_gep() -> Result<(), Box<dyn std::error::Error>> {
        let mut module = Module::new("gepexample", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), Some("u32"));
        let ptr_ty = storage.add_type(
            Type::Ptr {
                pointee: i32_ty,
                address_space: None,
            },
            Some("*u32"),
        );

        let func = module
            .add_function(
                "example",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();

        let func = module.get_function_mut(func);

        let entry = func.entry_block;

        let param1 = func.param(0)?;

        let ptr_val =
            func.blocks[entry].instr_alloca(ptr_ty, 4, None, Location::Unknown, &storage)?;
        let k1 = Operand::const_int(1, i32_ty);
        func.blocks[entry].instr_store(
            ptr_val.clone(),
            k1.clone(),
            None,
            Location::Unknown,
            &storage,
        )?;

        let ptr_idx1 = func.blocks[entry].instr_gep(
            ptr_val,
            &[GepIndex::Const(1)],
            ptr_ty,
            Location::Unknown,
            &storage,
        )?;
        func.blocks[entry].instr_store(
            ptr_idx1.clone(),
            param1,
            None,
            Location::Unknown,
            &storage,
        )?;

        let result = func.blocks[entry].instr_load(ptr_idx1, None, Location::Unknown, &storage)?;

        func.blocks[entry].instr_ret(Some(&result), Location::Unknown);

        let result = test_run_module(
            &module,
            &storage,
            "example",
            &[JitValue::U32(2)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(2));

        Ok(())
    }

    fn test_run_module(
        module: &Module,
        storage: &TypeStorage,
        name: &str,
        args: &[JitValue],
        ret_ty: JitValue,
    ) -> Result<JitValue, crate::llvm::Error> {
        let result = lower_module_to_llvmir(module, storage)?;
        let engine = create_jit_engine(result, 3)?;

        let res = unsafe { engine.execute(name, args, ret_ty)? };

        Ok(res)
    }
}
