# IRVM

A IR compiler target with a native Rust friendly API that lowers to LLVM IR (or other targets).

## How it works

Basically mimic a IR that closely resembles LLVM IR in Rust structures and only interface with LLVM at the time of lowering to LLVM IR / compilation.

Ideally when lowering to LLVM IR the IR in IRVM should be valid due to checks on our side.

## Why?

There are some nice crates to use LLVM from Rust, like [inkwell](https://github.com/TheDan64/inkwell), but due to the need to model the C++ ownership (ffi) in Rust, the API tends to not be so user friendly, even if they try hard, also some functions like [GEP](https://thedan64.github.io/inkwell/inkwell/builder/struct.Builder.html#method.build_gep) are `unsafe` if used incorrectly, this library strives to provide a Rust friendly API thats fully safe.

```rust,ignore
use std::error::Error;

use irvm::{
    block::IcmpCond,
    common::Location,
    function::Parameter,
    module::Module,
    types::{Type, TypeStorage},
    value::Operand,
};

use irvm_lower::llvm::{lower_module_to_llvmir, OutputCompilation};

fn main() -> Result<(), Box<dyn Error>> {
    let mut module = Module::new("example", Location::unknown());
    let mut storage = TypeStorage::new();
    let _bool_ty = storage.add_type(Type::Int(1), Some("bool"));
    let i32_ty = storage.add_type(Type::Int(32), Some("i32"));
    let _i64_ty = storage.add_type(Type::Int(64), Some("i64"));
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

    // test function
    {
        let func = module.get_function_mut(test_func);
        let value = func.param(0)?;
        func.entry_block()
            .instr_ret(Some(&value), Location::Unknown);
    }

    let compile_result = lower_module_to_llvmir(&module, &storage)?;
    compile_result.dump();
    Ok(())
}
```
