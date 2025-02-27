use std::{
    collections::{HashMap, hash_map::Entry},
    error::Error,
    ffi::CString,
    ptr::null_mut,
};

use irvm::{
    block::{BlockIdx, Instruction},
    function::Function,
    module::Module,
    types::Type,
    value::Operand,
};

use itertools::Itertools;
use llvm_sys::{
    core,
    prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMContextRef, LLVMTypeRef, LLVMValueRef},
};

pub fn lower_module(module: &Module) -> Result<(), Box<dyn Error>> {
    unsafe {
        let ctx = core::LLVMContextCreate();
        let module_name = CString::new(module.name.clone())?;
        let llvm_module = core::LLVMModuleCreateWithNameInContext(module_name.as_ptr(), ctx);

        for (_fun_idx, func) in module.functions.iter() {
            let name = CString::new(func.name.as_str()).unwrap();

            let ret_ty = lower_type(ctx, &func.result_type);
            let mut params = func
                .parameters
                .iter()
                .map(|x| lower_type(ctx, &x.ty))
                .collect_vec();
            let fn_ty = core::LLVMFunctionType(ret_ty, params.as_mut_ptr(), params.len() as u32, 0);
            let fn_ptr = core::LLVMAddFunction(llvm_module, name.as_ptr(), fn_ty);
            let builder = core::LLVMCreateBuilderInContext(ctx);

            let mut fn_ctx = FnCtx {
                ctx,
                fn_ptr,
                func: func.clone(),
                builder,
                blocks: Default::default(),
                values: Default::default(),
                block_args: Default::default(),
            };

            lower_block(&mut fn_ctx, func.entry_block, true);
        }

        core::LLVMDumpModule(llvm_module);

        core::LLVMDisposeModule(llvm_module);
        core::LLVMContextDispose(ctx);
    }

    Ok(())
}

#[derive(Debug)]
struct FnCtx {
    ctx: LLVMContextRef,
    fn_ptr: LLVMValueRef,
    func: Function,
    builder: LLVMBuilderRef,
    blocks: HashMap<usize, LLVMBasicBlockRef>,
    values: HashMap<usize, LLVMValueRef>,
    block_args: HashMap<usize, Vec<LLVMValueRef>>,
}

fn lower_block(ctx: &mut FnCtx, block_idx: BlockIdx, create_block: bool) {
    unsafe {
        let block_name = CString::new(if block_idx.to_idx() == 0 {
            "entry".to_string()
        } else {
            format!("bb{}", block_idx.to_idx())
        })
        .unwrap();
        let null_name = c"";
        let block_ptr = if create_block {
            core::LLVMAppendBasicBlock(ctx.fn_ptr, block_name.as_ptr())
        } else {
            *ctx.blocks.get(&block_idx.to_idx()).unwrap()
        };
        core::LLVMPositionBuilderAtEnd(ctx.builder, block_ptr);

        let preds = ctx.func.find_preds_for(block_idx);
        let mut block_args = Vec::new();

        if !preds.is_empty() {
            let operand_len = preds.first().unwrap().1.len();

            for i in 0..operand_len {
                for (block_idx, operands) in &preds {
                    let value = lower_operand(ctx, &operands[i]);
                    let pred_ptr = ctx.blocks.get(&block_idx.to_idx()).unwrap();
                    let label = core::LLVMGetBasicBlockName(*pred_ptr);
                    let ty = core::LLVMTypeOf(value);
                    block_args.push(core::LLVMBuildPhi(ctx.builder, ty, label));
                }
            }
        }

        ctx.block_args.insert(block_idx.to_idx(), Vec::new());

        let block = &ctx.func.blocks[block_idx];

        for (inst_idx, inst) in block.instructions.iter() {
            match inst {
                Instruction::BinaryOp(binary_op) => match binary_op {
                    irvm::block::BinaryOp::Add { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildAdd(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                    irvm::block::BinaryOp::Sub { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildSub(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                    irvm::block::BinaryOp::Mul { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildMul(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                    irvm::block::BinaryOp::Div {
                        lhs,
                        rhs,
                        signed,
                        exact,
                    } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value = if *signed {
                            core::LLVMBuildSDiv(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        } else {
                            core::LLVMBuildUDiv(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        };
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                    irvm::block::BinaryOp::Rem { lhs, rhs, signed } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value = if *signed {
                            core::LLVMBuildSRem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        } else {
                            core::LLVMBuildURem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        };
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                    irvm::block::BinaryOp::FAdd { lhs, rhs } => todo!(),
                    irvm::block::BinaryOp::FSub { lhs, rhs } => todo!(),
                    irvm::block::BinaryOp::FMul { lhs, rhs } => todo!(),
                    irvm::block::BinaryOp::FDiv { lhs, rhs } => todo!(),
                    irvm::block::BinaryOp::FRem { lhs, rhs } => todo!(),
                },
                Instruction::BitwiseBinaryOp(bitwise_binary_op) => match bitwise_binary_op {
                    irvm::block::BitwiseBinaryOp::Shl { lhs, rhs, nsw, nuw } => todo!(),
                    irvm::block::BitwiseBinaryOp::Lshr { lhs, rhs, exact } => todo!(),
                    irvm::block::BitwiseBinaryOp::Ashr { lhs, rhs, exact } => todo!(),
                    irvm::block::BitwiseBinaryOp::And { lhs, rhs } => todo!(),
                    irvm::block::BitwiseBinaryOp::Or { lhs, rhs, disjoint } => todo!(),
                    irvm::block::BitwiseBinaryOp::Xor { lhs, rhs } => todo!(),
                },
                Instruction::VectorOp(vector_op) => todo!(),
                Instruction::MemoryOp(memory_op) => match memory_op {
                    irvm::block::MemoryOp::Alloca {
                        ty,
                        num_elements,
                        inalloca,
                        align,
                        addr_space,
                    } => {
                        let ty_ptr = lower_type(ctx.ctx, ty);
                        let value = if *num_elements > 1 {
                            let int_ty_ptr = lower_type(ctx.ctx, &Type::Int(64));
                            let const_val =
                                core::LLVMConstInt(int_ty_ptr, (*num_elements) as u64, 0);
                            core::LLVMBuildArrayAlloca(ctx.builder, ty_ptr, const_val, null_mut())
                        } else {
                            core::LLVMBuildAlloca(ctx.builder, ty_ptr, null_mut())
                        };
                        ctx.values.insert(inst_idx.to_idx(), value);
                    }
                },
                Instruction::OtherOp(other_op) => match other_op {
                    irvm::block::OtherOp::Call(call_op) => {
                        // core::LLVMBuildCall2(ctx.builder, arg2, Fn, Args, NumArgs, Name)
                    }
                    irvm::block::OtherOp::Icmp { cond, lhs, rhs } => todo!(),
                    irvm::block::OtherOp::Fcmp { cond, lhs, rhs } => todo!(),
                },
            }
        }

        match &block.terminator {
            irvm::block::Terminator::Ret(op) => {
                if let Some(op) = op {
                    let value = lower_operand(ctx, op);
                    core::LLVMBuildRet(ctx.builder, value);
                } else {
                    core::LLVMBuildRetVoid(ctx.builder);
                }
            }
            irvm::block::Terminator::Br { block, .. } => {
                if let Entry::Vacant(e) = ctx.blocks.entry(block.to_idx()) {
                    let block_name = CString::new(format!("bb{}", block.to_idx())).unwrap();
                    let block_ptr = core::LLVMAppendBasicBlock(ctx.fn_ptr, block_name.as_ptr());
                    e.insert(block_ptr);
                }

                let target_block = *ctx.blocks.get(&block.to_idx()).unwrap();

                core::LLVMBuildBr(ctx.builder, target_block);
            }
            irvm::block::Terminator::CondBr {
                then_block: if_block,
                else_block: then_block,
                cond,
                ..
            } => {
                if let Entry::Vacant(e) = ctx.blocks.entry(if_block.to_idx()) {
                    let block_name = CString::new(format!(
                        "bb_{}_true_{}",
                        block_idx.to_idx(),
                        if_block.to_idx()
                    ))
                    .unwrap();
                    let block_ptr = core::LLVMAppendBasicBlock(ctx.fn_ptr, block_name.as_ptr());
                    e.insert(block_ptr);
                }

                if let Entry::Vacant(e) = ctx.blocks.entry(then_block.to_idx()) {
                    let block_name = CString::new(format!(
                        "bb_{}_false_{}",
                        block_idx.to_idx(),
                        then_block.to_idx()
                    ))
                    .unwrap();
                    let block_ptr = core::LLVMAppendBasicBlock(ctx.fn_ptr, block_name.as_ptr());
                    e.insert(block_ptr);
                }

                let cond = lower_operand(ctx, cond);

                let if_block = *ctx.blocks.get(&if_block.to_idx()).unwrap();
                let then_block = *ctx.blocks.get(&then_block.to_idx()).unwrap();

                core::LLVMBuildCondBr(ctx.builder, cond, if_block, then_block);
            }
        }
    }
}

fn lower_operand(ctx: &FnCtx, operand: &Operand) -> LLVMValueRef {
    unsafe {
        match operand {
            Operand::Parameter(idx, _ty) => core::LLVMGetParam(ctx.fn_ptr, (*idx) as u32),
            Operand::Value(index, _) => *ctx.values.get(&index.to_idx()).unwrap(),
            Operand::Constant(const_value, ty) => {
                let ty_ptr = lower_type(ctx.ctx, ty);
                match const_value {
                    irvm::value::ConstValue::Bool(value) => {
                        core::LLVMConstInt(ty_ptr, *value as u64, 0 as i32)
                    }
                    irvm::value::ConstValue::Int(value) => {
                        core::LLVMConstInt(ty_ptr, *value, 0 as i32)
                    }
                    irvm::value::ConstValue::Float(value) => core::LLVMConstReal(ty_ptr, *value),
                    irvm::value::ConstValue::Array(const_values) => todo!(),
                    irvm::value::ConstValue::Vector(const_values) => todo!(),
                    irvm::value::ConstValue::Struct(const_values) => todo!(),
                    irvm::value::ConstValue::NullPtr => todo!(),
                    irvm::value::ConstValue::Undef => core::LLVMGetUndef(ty_ptr),
                    irvm::value::ConstValue::Poison => core::LLVMGetPoison(ty_ptr),
                }
            }
            Operand::BlockArgument { block_idx, nth, .. } => {
                ctx.block_args.get(block_idx).unwrap()[*nth]
            }
        }
    }
}

fn lower_type(ctx: LLVMContextRef, ty: &Type) -> LLVMTypeRef {
    unsafe {
        match ty {
            Type::Int(width) => core::LLVMIntTypeInContext(ctx, *width),
            Type::Half => core::LLVMHalfTypeInContext(ctx),
            Type::BFloat => core::LLVMBFloatTypeInContext(ctx),
            Type::Float => core::LLVMFloatTypeInContext(ctx),
            Type::Double => core::LLVMDoubleTypeInContext(ctx),
            Type::Fp128 => core::LLVMFP128TypeInContext(ctx),
            Type::X86Fp80 => core::LLVMX86FP80TypeInContext(ctx),
            Type::PpcFp128 => core::LLVMPPCFP128TypeInContext(ctx),
            Type::Ptr(address_space) => {
                core::LLVMPointerTypeInContext(ctx, address_space.unwrap_or(0))
            }
            Type::Vector(vector_type) => {
                let inner = lower_type(ctx, &vector_type.ty);
                core::LLVMVectorType(inner, vector_type.size)
            }
            Type::Array(array_type) => {
                let inner = lower_type(ctx, &array_type.ty);
                core::LLVMArrayType2(inner, array_type.size)
            }
            Type::Struct(struct_type) => {
                let mut fields = Vec::new();

                for field in struct_type.fields.iter() {
                    fields.push(lower_type(ctx, field));
                }

                if let Some(ident) = &struct_type.ident {
                    let name = CString::new(ident.as_str()).unwrap();
                    let ptr = core::LLVMStructCreateNamed(ctx, name.as_ptr());

                    core::LLVMStructSetBody(
                        ptr,
                        fields.as_mut_ptr(),
                        fields.len() as u32,
                        struct_type.packed as i32,
                    );

                    ptr
                } else {
                    core::LLVMStructTypeInContext(
                        ctx,
                        fields.as_mut_ptr(),
                        fields.len() as u32,
                        struct_type.packed as i32,
                    )
                }
            }
        }
    }
}
