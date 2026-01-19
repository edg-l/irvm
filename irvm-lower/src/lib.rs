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

    #[test]
    fn test_type_casts() -> Result<(), Box<dyn std::error::Error>> {
        let mut module = Module::new("casts", Location::unknown());
        let mut storage = TypeStorage::new();
        let i16_ty = storage.add_type(Type::Int(16), None);
        let i32_ty = storage.add_type(Type::Int(32), None);

        // Test zext: trunc i32->i16, then zext i16->i32 (MCJIT only supports i32 params/returns reliably)
        let func = module
            .add_function(
                "test_zext",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;
            // trunc to i16 first, then zext back to i32
            let truncated = func
                .entry_block()
                .instr_trunc(param, i16_ty, Location::Unknown)?;
            let extended = func
                .entry_block()
                .instr_zext(truncated, i32_ty, Location::Unknown)?;
            func.entry_block()
                .instr_ret(Some(&extended), Location::Unknown);
        }

        // Test sext: trunc i32->i16, then sext i16->i32
        let func = module
            .add_function(
                "test_sext",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;
            let truncated = func
                .entry_block()
                .instr_trunc(param, i16_ty, Location::Unknown)?;
            let extended = func
                .entry_block()
                .instr_sext(truncated, i32_ty, Location::Unknown)?;
            func.entry_block()
                .instr_ret(Some(&extended), Location::Unknown);
        }

        // zext: 50000 -> trunc to i16 -> zext to i32 = 50000 (no change, fits in u16)
        let result = test_run_module(
            &module,
            &storage,
            "test_zext",
            &[JitValue::U32(50000)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(50000));

        // zext: 0x12345 -> trunc to i16 (0x2345) -> zext to i32 = 0x2345
        let result = test_run_module(
            &module,
            &storage,
            "test_zext",
            &[JitValue::U32(0x12345)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(0x2345));

        // sext: 0xFFFF (= -1 as i16) -> sext to i32 = 0xFFFFFFFF (-1)
        let result = test_run_module(
            &module,
            &storage,
            "test_sext",
            &[JitValue::U32(0xFFFF)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(0xFFFFFFFF));

        Ok(())
    }

    #[test]
    fn test_select() -> Result<(), Box<dyn std::error::Error>> {
        let mut module = Module::new("select", Location::unknown());
        let mut storage = TypeStorage::new();
        let _i1_ty = storage.get_or_create_i1(); // Required for icmp result
        let i32_ty = storage.add_type(Type::Int(32), None);

        // Use i32 param and compare to create i1 condition (MCJIT doesn't like i1 params)
        let func = module
            .add_function(
                "test_select",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;
            // cond = (param != 0)
            let cond = func.entry_block().instr_icmp(
                IcmpCond::Ne,
                param,
                Operand::const_int(0, i32_ty),
                Location::Unknown,
                &storage,
            )?;
            let true_val = Operand::const_int(42, i32_ty);
            let false_val = Operand::const_int(100, i32_ty);
            let result =
                func.entry_block()
                    .instr_select(cond, true_val, false_val, Location::Unknown)?;
            func.entry_block()
                .instr_ret(Some(&result), Location::Unknown);
        }

        // select(1 != 0, 42, 100) = 42
        let result = test_run_module(
            &module,
            &storage,
            "test_select",
            &[JitValue::U32(1)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(42));

        // select(0 != 0, 42, 100) = 100
        let result = test_run_module(
            &module,
            &storage,
            "test_select",
            &[JitValue::U32(0)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(100));

        Ok(())
    }

    #[test]
    fn test_switch() -> Result<(), Box<dyn std::error::Error>> {
        use irvm::block::SwitchCase;

        let mut module = Module::new("switch", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), None);

        let func = module
            .add_function(
                "test_switch",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;

            let case1_block = func.add_block(&[]);
            let case2_block = func.add_block(&[]);
            let default_block = func.add_block(&[]);

            let entry = func.entry_block;
            func.blocks[entry].instr_switch(
                param,
                default_block,
                &[],
                vec![
                    SwitchCase {
                        value: 1,
                        block: case1_block,
                        arguments: vec![],
                    },
                    SwitchCase {
                        value: 2,
                        block: case2_block,
                        arguments: vec![],
                    },
                ],
                Location::Unknown,
            );

            // case 1: return 10
            func.blocks[case1_block]
                .instr_ret(Some(&Operand::const_int(10, i32_ty)), Location::Unknown);
            // case 2: return 20
            func.blocks[case2_block]
                .instr_ret(Some(&Operand::const_int(20, i32_ty)), Location::Unknown);
            // default: return 0
            func.blocks[default_block]
                .instr_ret(Some(&Operand::const_int(0, i32_ty)), Location::Unknown);
        }

        assert_eq!(
            test_run_module(
                &module,
                &storage,
                "test_switch",
                &[JitValue::U32(1)],
                JitValue::U32(0)
            )?,
            JitValue::U32(10)
        );
        assert_eq!(
            test_run_module(
                &module,
                &storage,
                "test_switch",
                &[JitValue::U32(2)],
                JitValue::U32(0)
            )?,
            JitValue::U32(20)
        );
        assert_eq!(
            test_run_module(
                &module,
                &storage,
                "test_switch",
                &[JitValue::U32(99)],
                JitValue::U32(0)
            )?,
            JitValue::U32(0)
        );

        Ok(())
    }

    #[test]
    fn test_global_variable() -> Result<(), Box<dyn std::error::Error>> {
        use irvm::value::ConstValue;

        let mut module = Module::new("globals", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), None);
        let ptr_ty = storage.add_type(
            Type::Ptr {
                pointee: i32_ty,
                address_space: None,
            },
            None,
        );

        // Create a global variable initialized to 42
        let global_idx = module.add_global(
            "my_global",
            i32_ty,
            Some(ConstValue::Int(42)),
            false,
            Location::Unknown,
        );

        let func = module
            .add_function("read_global", &[], Some(i32_ty), Location::Unknown)
            .get_id();
        {
            let func = module.get_function_mut(func);
            let global_ptr = Operand::global(global_idx, ptr_ty);
            let loaded =
                func.entry_block()
                    .instr_load(global_ptr, None, Location::Unknown, &storage)?;
            func.entry_block()
                .instr_ret(Some(&loaded), Location::Unknown);
        }

        let result = test_run_module(&module, &storage, "read_global", &[], JitValue::U32(0))?;
        assert_eq!(result, JitValue::U32(42));

        Ok(())
    }

    #[test]
    fn test_atomicrmw() -> Result<(), Box<dyn std::error::Error>> {
        use irvm::block::{AtomicOrdering, AtomicRMWOp};

        let mut module = Module::new("atomic", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), None);
        let ptr_ty = storage.add_type(
            Type::Ptr {
                pointee: i32_ty,
                address_space: None,
            },
            None,
        );

        let func = module
            .add_function(
                "test_atomic_add",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;
            let entry = func.entry_block;

            // Allocate, store initial value, atomicrmw add, return old value
            let ptr =
                func.blocks[entry].instr_alloca(ptr_ty, 4, None, Location::Unknown, &storage)?;
            func.blocks[entry].instr_store(
                ptr.clone(),
                Operand::const_int(10, i32_ty),
                None,
                Location::Unknown,
                &storage,
            )?;
            let old_val = func.blocks[entry].instr_atomicrmw(
                AtomicRMWOp::Add,
                ptr,
                param,
                AtomicOrdering::SeqCst,
                Location::Unknown,
            )?;
            func.blocks[entry].instr_ret(Some(&old_val), Location::Unknown);
        }

        // atomicrmw add returns the old value (10), adds param (5)
        let result = test_run_module(
            &module,
            &storage,
            "test_atomic_add",
            &[JitValue::U32(5)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(10));

        Ok(())
    }

    #[test]
    fn test_extractvalue_insertvalue() -> Result<(), Box<dyn std::error::Error>> {
        use irvm::types::StructType;
        use irvm::value::ConstValue;

        let mut module = Module::new("aggregate", Location::unknown());
        let mut storage = TypeStorage::new();
        let i32_ty = storage.add_type(Type::Int(32), None);
        let i64_ty = storage.add_type(Type::Int(64), None);

        // Create struct { i32, i64 }
        let struct_ty = storage.add_type(
            Type::Struct(
                StructType {
                    packed: false,
                    ident: None,
                    fields: vec![i32_ty, i64_ty],
                    debug_field_names: vec![],
                }
                .into(),
            ),
            None,
        );

        // Test insertvalue + extractvalue: create struct, insert value, extract it back
        let func = module
            .add_function(
                "test_aggregate",
                &[Parameter::new(i32_ty, Location::Unknown)],
                Some(i32_ty),
                Location::Unknown,
            )
            .get_id();
        {
            let func = module.get_function_mut(func);
            let param = func.param(0)?;
            // Start with undef struct, insert param into field 0, extract it back
            let undef_struct = Operand::Constant(ConstValue::Undef, struct_ty);
            let with_field = func.entry_block().instr_insertvalue(
                undef_struct,
                param.clone(),
                &[0],
                Location::Unknown,
            )?;
            let extracted = func.entry_block().instr_extractvalue(
                with_field,
                &[0],
                i32_ty,
                Location::Unknown,
            )?;
            func.entry_block()
                .instr_ret(Some(&extracted), Location::Unknown);
        }

        // insertvalue then extractvalue should return the same value
        let result = test_run_module(
            &module,
            &storage,
            "test_aggregate",
            &[JitValue::U32(123)],
            JitValue::U32(0),
        )?;
        assert_eq!(result, JitValue::U32(123));

        Ok(())
    }
}
