use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    path::{Path, PathBuf},
    ptr::null_mut,
    rc::Rc,
};

use irvm::{
    block::{BlockIdx, DebugOp, DebugVariable, Instruction},
    common::Location,
    datalayout::DataLayout,
    function::Function,
    module::{Module, TypeIdx},
    types::{Type, TypeStorage},
    value::{ConstValue, Operand},
};

use itertools::Itertools;
use llvm_sys::{
    LLVMIntPredicate, LLVMOpaqueMetadata, LLVMRealPredicate, core,
    debuginfo::{self, LLVMDIFlagPublic, LLVMDWARFEmissionKind},
    prelude::{
        LLVMBasicBlockRef, LLVMBuilderRef, LLVMContextRef, LLVMDIBuilderRef, LLVMMetadataRef,
        LLVMTypeRef, LLVMValueRef,
    },
};

#[derive(Debug, Clone)]
pub enum OutputCompilation {
    Stdout,
    File(PathBuf),
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum Error {
    #[error("llvm error: {:?}", 0)]
    LLVMError(String),
    #[error(transparent)]
    NulError(#[from] std::ffi::NulError),
}

pub fn lower_module_to_llvmir(
    module: &Module,
    storage: &TypeStorage,
    output: OutputCompilation,
) -> Result<(), Error> {
    unsafe {
        let ctx = core::LLVMContextCreate();
        let module_name = CString::new(module.name.clone())?;
        let llvm_module = core::LLVMModuleCreateWithNameInContext(module_name.as_ptr(), ctx);

        let datalayout_str = CString::new(module.data_layout.to_llvm_string()).unwrap();
        core::LLVMSetDataLayout(llvm_module, datalayout_str.as_ptr());
        let triple_str = CString::new(module.target_triple.to_string()).unwrap();
        core::LLVMSetTarget(llvm_module, triple_str.as_ptr());

        let mut functions: HashMap<_, _> = Default::default();
        let mut dfunctions: HashMap<_, _> = Default::default();
        // let mut debug_functions: HashMap<_, _> = Default::default();
        let builder = core::LLVMCreateBuilderInContext(ctx);
        let dibuilder = debuginfo::LLVMCreateDIBuilder(llvm_module);

        let compile_unit_file = get_difile_location(dibuilder, &module.location);

        let producer = c"IRVM version 0.1.0";
        let flags = c"";
        let splitname = c"";
        let sysroot = c"";
        let sdk = c"";

        let compile_unit = debuginfo::LLVMDIBuilderCreateCompileUnit(
            dibuilder,
            debuginfo::LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC17,
            compile_unit_file,
            producer.as_ptr(),
            producer.count_bytes(),
            0,
            flags.as_ptr(),
            flags.count_bytes(),
            0,
            splitname.as_ptr(),
            splitname.count_bytes(),
            LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
            0,
            0,
            0,
            sysroot.as_ptr(),
            sysroot.count_bytes(),
            sdk.as_ptr(),
            sdk.count_bytes(),
        );

        let debug_module = debuginfo::LLVMDIBuilderCreateModule(
            dibuilder,
            compile_unit,
            module_name.as_ptr(),
            module.name.len(),
            c"".as_ptr(),
            0,
            c"".as_ptr(),
            0,
            c"".as_ptr(),
            0,
        );

        for (fun_idx, func) in module.functions.iter() {
            let name = CString::new(func.name.as_str()).unwrap();

            let ret_ty = lower_type(ctx, storage, func.result_type);
            let mut params = func
                .parameters
                .iter()
                .map(|x| lower_type(ctx, storage, x.ty))
                .collect_vec();
            let fn_ty = core::LLVMFunctionType(ret_ty, params.as_mut_ptr(), params.len() as u32, 0);
            let fn_ptr = core::LLVMAddFunction(llvm_module, name.as_ptr(), fn_ty);
            functions.insert(fun_idx.to_idx(), (fn_ptr, fn_ty));

            let mut file = compile_unit_file;

            let mut line = 0;
            match &func.location {
                Location::Unknown => {}
                Location::File(file_location) => {
                    file = get_difile(dibuilder, &file_location.file);
                    line = file_location.line;
                }
            }

            let mut debug_param_types = Vec::new();

            for param in func.parameters.iter() {
                let ptr = lower_debug_type(&module.data_layout, dibuilder, storage, param.ty);
                debug_param_types.push(ptr);
            }

            let debug_func_ty = debuginfo::LLVMDIBuilderCreateSubroutineType(
                dibuilder,
                file,
                debug_param_types.as_mut_ptr(),
                debug_param_types.len() as u32,
                0,
            );

            let _ret_debug_ty =
                lower_debug_type(&module.data_layout, dibuilder, storage, func.result_type);

            let di_func = debuginfo::LLVMDIBuilderCreateFunction(
                dibuilder,
                debug_module,
                name.as_ptr(),
                name.count_bytes(),
                name.as_ptr(),
                name.count_bytes(),
                file,
                line,
                debug_func_ty,
                0,
                1,
                line,
                0,
                0,
            );
            dfunctions.insert(fun_idx.to_idx(), di_func);
            debuginfo::LLVMSetSubprogram(fn_ptr, di_func);
        }

        let functions = Rc::new(functions);

        for (fun_idx, func) in module.functions.iter() {
            let fn_ptr = functions.get(&fun_idx.to_idx()).unwrap().0;
            let dfunc = *dfunctions.get(&fun_idx.to_idx()).unwrap();

            let mut fn_ctx = FnCtx {
                ctx,
                fn_ptr,
                func: func.clone(),
                builder,
                dibuilder,
                storage,
                blocks: Default::default(),
                values: Default::default(),
                block_args: Default::default(),
                functions: Rc::clone(&functions),
                debug_scope: dfunc,
                datalayout: &module.data_layout,
            };

            for (id, _) in func.blocks.iter() {
                add_block(&mut fn_ctx, id, None);
            }

            for (id, _) in func.blocks.iter() {
                lower_block(&mut fn_ctx, id)?;
            }

            debuginfo::LLVMDIBuilderFinalizeSubprogram(dibuilder, dfunc);
        }

        if let OutputCompilation::Stdout = output {
            core::LLVMDumpModule(llvm_module);
        }

        let mut out_msg: *mut i8 = null_mut();
        let ok = llvm_sys::analysis::LLVMVerifyModule(
            llvm_module,
            llvm_sys::analysis::LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &raw mut out_msg,
        );

        if ok != 0 {
            let msg = {
                let msg = CStr::from_ptr(out_msg);
                msg.to_string_lossy().to_string()
            };

            if !out_msg.is_null() {
                core::LLVMDisposeMessage(out_msg);
            }

            core::LLVMDisposeModule(llvm_module);
            core::LLVMContextDispose(ctx);

            return Err(Error::LLVMError(msg));
        }

        if !out_msg.is_null() {
            core::LLVMDisposeMessage(out_msg);
            out_msg = null_mut();
        }

        if let OutputCompilation::File(file) = output {
            let file = CString::new(&*file.to_string_lossy())?;
            core::LLVMPrintModuleToFile(llvm_module, file.as_ptr(), &raw mut out_msg);
        }

        core::LLVMDisposeModule(llvm_module);
        core::LLVMContextDispose(ctx);
    }

    Ok(())
}

#[derive(Debug)]
struct FnCtx<'m> {
    ctx: LLVMContextRef,
    fn_ptr: LLVMValueRef,
    func: Function,
    storage: &'m TypeStorage,
    builder: LLVMBuilderRef,
    dibuilder: LLVMDIBuilderRef,
    debug_scope: LLVMMetadataRef,
    functions: Rc<HashMap<usize, (LLVMValueRef, LLVMTypeRef)>>,
    blocks: HashMap<usize, LLVMBasicBlockRef>,
    // block, inst
    values: HashMap<(usize, usize), LLVMValueRef>,
    block_args: HashMap<usize, Vec<LLVMValueRef>>,
    datalayout: &'m DataLayout,
}

// Returns the next block to lower.
fn lower_block(ctx: &mut FnCtx, block_idx: BlockIdx) -> Result<(), Error> {
    unsafe {
        let null_name = c"";
        let block_ptr = *ctx.blocks.get(&block_idx.to_idx()).unwrap();
        core::LLVMPositionBuilderAtEnd(ctx.builder, block_ptr);
        add_preds(ctx, block_idx);

        for (inst_idx, (loc, inst)) in ctx.func.blocks[block_idx].instructions.iter() {
            let loc = set_loc(ctx.ctx, ctx.builder, loc, ctx.debug_scope);

            match inst {
                Instruction::BinaryOp(binary_op) => match binary_op {
                    irvm::block::BinaryOp::Add { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildAdd(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        if *nuw {
                            core::LLVMSetNUW(value, (*nuw) as i32);
                        }
                        if *nsw {
                            core::LLVMSetNSW(value, (*nsw) as i32);
                        }
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::Sub { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildSub(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        if *nuw {
                            core::LLVMSetNUW(value, (*nuw) as i32);
                        }
                        if *nsw {
                            core::LLVMSetNSW(value, (*nsw) as i32);
                        }
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::Mul { lhs, rhs, nsw, nuw } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildMul(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        if *nuw {
                            core::LLVMSetNUW(value, (*nuw) as i32);
                        }
                        if *nsw {
                            core::LLVMSetNSW(value, (*nsw) as i32);
                        }
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
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
                            if *exact {
                                core::LLVMBuildExactSDiv(
                                    ctx.builder,
                                    lhs_ptr,
                                    rhs_ptr,
                                    null_name.as_ptr(),
                                )
                            } else {
                                core::LLVMBuildSDiv(
                                    ctx.builder,
                                    lhs_ptr,
                                    rhs_ptr,
                                    null_name.as_ptr(),
                                )
                            }
                        } else if *exact {
                            core::LLVMBuildExactUDiv(
                                ctx.builder,
                                lhs_ptr,
                                rhs_ptr,
                                null_name.as_ptr(),
                            )
                        } else {
                            core::LLVMBuildUDiv(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::Rem { lhs, rhs, signed } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value = if *signed {
                            core::LLVMBuildSRem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        } else {
                            core::LLVMBuildURem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr())
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FAdd { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFAdd(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FSub { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFSub(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FMul { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFMul(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FDiv { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFDiv(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FRem { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFRem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
                Instruction::BitwiseBinaryOp(bitwise_binary_op) => match bitwise_binary_op {
                    irvm::block::BitwiseBinaryOp::Shl { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildShl(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BitwiseBinaryOp::Lshr { lhs, rhs, exact } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildLShr(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        core::LLVMSetExact(value, (*exact) as i32);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BitwiseBinaryOp::Ashr { lhs, rhs, exact } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildAShr(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        core::LLVMSetExact(value, (*exact) as i32);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BitwiseBinaryOp::And { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildAnd(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BitwiseBinaryOp::Or { lhs, rhs, disjoint } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildOr(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        core::LLVMSetIsDisjoint(value, (*disjoint) as i32);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BitwiseBinaryOp::Xor { lhs, rhs } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildXor(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
                Instruction::VectorOp(vector_op) => match vector_op {
                    irvm::block::VectorOp::ExtractElement { vector, idx } => {
                        let vector = lower_operand(ctx, vector);
                        let idx = lower_operand(ctx, idx);
                        let value = core::LLVMBuildExtractElement(
                            ctx.builder,
                            vector,
                            idx,
                            null_name.as_ptr(),
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
                Instruction::MemoryOp(memory_op) => match memory_op {
                    irvm::block::MemoryOp::Alloca {
                        ty, num_elements, ..
                    } => {
                        let ty_ptr = lower_type(ctx.ctx, ctx.storage, *ty);
                        let value = if *num_elements > 1 {
                            let int_ty_ptr =
                                lower_type(ctx.ctx, ctx.storage, ctx.storage.i64_ty().unwrap());
                            let const_val =
                                core::LLVMConstInt(int_ty_ptr, (*num_elements) as u64, 0);
                            core::LLVMBuildArrayAlloca(
                                ctx.builder,
                                ty_ptr,
                                const_val,
                                null_name.as_ptr(),
                            )
                        } else {
                            core::LLVMBuildAlloca(ctx.builder, ty_ptr, null_mut())
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
                Instruction::OtherOp(other_op) => match other_op {
                    irvm::block::OtherOp::Call(call_op) => {
                        let (target_fn_ptr, fn_ty) = match &call_op.fn_target {
                            irvm::block::CallableValue::Symbol(id) => {
                                *ctx.functions.get(&id.to_idx()).expect("function not found")
                            }
                            irvm::block::CallableValue::Pointer(operand, fn_ty) => {
                                let ptr = lower_operand(ctx, operand);

                                let ret_ty = lower_type(ctx.ctx, ctx.storage, fn_ty.return_type);
                                let mut params = fn_ty
                                    .parameters
                                    .iter()
                                    .map(|x| lower_type(ctx.ctx, ctx.storage, *x))
                                    .collect_vec();
                                let fn_ty = core::LLVMFunctionType(
                                    ret_ty,
                                    params.as_mut_ptr(),
                                    params.len() as u32,
                                    0,
                                );
                                (ptr, fn_ty)
                            } // todo: how to get fn ty?
                        };

                        let mut args = call_op
                            .params
                            .iter()
                            .map(|p| lower_operand(ctx, p))
                            .collect_vec();

                        let value = core::LLVMBuildCall2(
                            ctx.builder,
                            fn_ty,
                            target_fn_ptr,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            null_name.as_ptr(),
                        );

                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::OtherOp::Icmp { cond, lhs, rhs } => {
                        let lhs_val = lower_operand(ctx, lhs);
                        let rhs_val = lower_operand(ctx, rhs);
                        let value = core::LLVMBuildICmp(
                            ctx.builder,
                            match cond {
                                irvm::block::IcmpCond::Eq => LLVMIntPredicate::LLVMIntEQ,
                                irvm::block::IcmpCond::Ne => LLVMIntPredicate::LLVMIntNE,
                                irvm::block::IcmpCond::Ugt => LLVMIntPredicate::LLVMIntUGT,
                                irvm::block::IcmpCond::Uge => LLVMIntPredicate::LLVMIntUGE,
                                irvm::block::IcmpCond::Ult => LLVMIntPredicate::LLVMIntULT,
                                irvm::block::IcmpCond::Ule => LLVMIntPredicate::LLVMIntULE,
                                irvm::block::IcmpCond::Sgt => LLVMIntPredicate::LLVMIntSGT,
                                irvm::block::IcmpCond::Sge => LLVMIntPredicate::LLVMIntSGE,
                                irvm::block::IcmpCond::Slt => LLVMIntPredicate::LLVMIntSLT,
                                irvm::block::IcmpCond::Sle => LLVMIntPredicate::LLVMIntSLE,
                            },
                            lhs_val,
                            rhs_val,
                            null_name.as_ptr(),
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::OtherOp::Fcmp { cond, lhs, rhs } => {
                        let lhs_val = lower_operand(ctx, lhs);
                        let rhs_val = lower_operand(ctx, rhs);
                        let value = core::LLVMBuildFCmp(
                            ctx.builder,
                            match cond {
                                irvm::block::FcmpCond::False => {
                                    LLVMRealPredicate::LLVMRealPredicateFalse
                                }
                                irvm::block::FcmpCond::Oeq => LLVMRealPredicate::LLVMRealOEQ,
                                irvm::block::FcmpCond::Ogt => LLVMRealPredicate::LLVMRealOGT,
                                irvm::block::FcmpCond::Oge => LLVMRealPredicate::LLVMRealOGE,
                                irvm::block::FcmpCond::Olt => LLVMRealPredicate::LLVMRealOLT,
                                irvm::block::FcmpCond::Ole => LLVMRealPredicate::LLVMRealOLE,
                                irvm::block::FcmpCond::One => LLVMRealPredicate::LLVMRealONE,
                                irvm::block::FcmpCond::Ord => LLVMRealPredicate::LLVMRealORD,
                                irvm::block::FcmpCond::Ueq => LLVMRealPredicate::LLVMRealUEQ,
                                irvm::block::FcmpCond::Ugt => LLVMRealPredicate::LLVMRealUGT,
                                irvm::block::FcmpCond::Ult => LLVMRealPredicate::LLVMRealULT,
                                irvm::block::FcmpCond::Ule => LLVMRealPredicate::LLVMRealULE,
                                irvm::block::FcmpCond::Une => LLVMRealPredicate::LLVMRealUNE,
                                irvm::block::FcmpCond::Uno => LLVMRealPredicate::LLVMRealUNO,
                                irvm::block::FcmpCond::True => {
                                    LLVMRealPredicate::LLVMRealPredicateTrue
                                }
                            },
                            lhs_val,
                            rhs_val,
                            null_name.as_ptr(),
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
                Instruction::DebugOp(debug_op) => match debug_op {
                    DebugOp::Declare { address, variable } => {
                        let var = ctx.func.debug_vars.get(*variable).unwrap();
                        let address_ptr = lower_operand(ctx, address);
                        let var_ptr = lower_debug_var(
                            ctx.dibuilder,
                            ctx.debug_scope,
                            ctx.datalayout,
                            var,
                            ctx.storage,
                        )?;

                        let diexpr =
                            debuginfo::LLVMDIBuilderCreateExpression(ctx.dibuilder, null_mut(), 0);
                        debuginfo::LLVMDIBuilderInsertDeclareRecordAtEnd(
                            ctx.dibuilder,
                            address_ptr,
                            var_ptr,
                            diexpr,
                            loc,
                            block_ptr,
                        );
                    }
                    DebugOp::Value {
                        new_value,
                        variable,
                    } => {
                        let var = ctx.func.debug_vars.get(*variable).unwrap();
                        let value_ptr = lower_operand(ctx, new_value);
                        let var_ptr = lower_debug_var(
                            ctx.dibuilder,
                            ctx.debug_scope,
                            ctx.datalayout,
                            var,
                            ctx.storage,
                        )?;

                        let diexpr =
                            debuginfo::LLVMDIBuilderCreateExpression(ctx.dibuilder, null_mut(), 0);
                        debuginfo::LLVMDIBuilderInsertDbgValueRecordAtEnd(
                            ctx.dibuilder,
                            value_ptr,
                            var_ptr,
                            diexpr,
                            loc,
                            block_ptr,
                        );
                    }
                    DebugOp::Assign {
                        address,
                        new_value,
                        store_inst,
                        variable,
                    } => todo!(),
                },
            }
        }

        match ctx.func.blocks[block_idx].terminator.clone() {
            irvm::block::Terminator::Ret(op) => {
                set_loc(ctx.ctx, ctx.builder, &op.0, ctx.debug_scope);
                if let Some(op) = op.1 {
                    let value = lower_operand(ctx, &op);
                    core::LLVMBuildRet(ctx.builder, value);
                } else {
                    core::LLVMBuildRetVoid(ctx.builder);
                }
            }
            irvm::block::Terminator::Br {
                block: jmp_block,
                location,
                ..
            } => {
                set_loc(ctx.ctx, ctx.builder, &location, ctx.debug_scope);
                let target_block = *ctx.blocks.get(&jmp_block.to_idx()).unwrap();

                core::LLVMBuildBr(ctx.builder, target_block);
            }
            irvm::block::Terminator::CondBr {
                then_block: if_block,
                else_block: then_block,
                cond,
                ..
            } => {
                let cond = lower_operand(ctx, &cond);

                let if_block_value = *ctx.blocks.get(&if_block.to_idx()).unwrap();
                let then_block_value = *ctx.blocks.get(&then_block.to_idx()).unwrap();

                core::LLVMBuildCondBr(ctx.builder, cond, if_block_value, then_block_value);
            }
        }

        Ok(())
    }
}

fn add_block(ctx: &mut FnCtx, block_idx: BlockIdx, name: Option<String>) -> LLVMBasicBlockRef {
    unsafe {
        let block_name = CString::new(if block_idx.to_idx() == 0 {
            "entry".to_string()
        } else if let Some(name) = name {
            format!("bb{}", name)
        } else {
            format!("bb{}", block_idx.to_idx())
        })
        .unwrap();
        let block_ptr = core::LLVMAppendBasicBlock(ctx.fn_ptr, block_name.as_ptr());
        ctx.blocks.insert(block_idx.to_idx(), block_ptr);
        block_ptr
    }
}

fn lower_debug_var(
    dibuilder: LLVMDIBuilderRef,
    scope: LLVMMetadataRef,
    datalayout: &DataLayout,
    variable: &DebugVariable,
    storage: &TypeStorage,
) -> Result<LLVMMetadataRef, Error> {
    let name = CString::new(variable.name.clone())?;
    let difile = get_difile_location(dibuilder, &variable.location);

    let (line, _col) = match &variable.location {
        Location::Unknown => (0, 0),
        Location::File(file_location) => (file_location.line, file_location.col),
    };

    let ty_ptr = lower_debug_type(datalayout, dibuilder, storage, variable.ty);
    let align = datalayout.get_type_align(storage, variable.ty);

    Ok(unsafe {
        if let Some(param) = variable.parameter {
            debuginfo::LLVMDIBuilderCreateParameterVariable(
                dibuilder,
                scope,
                name.as_ptr(),
                name.count_bytes(),
                param,
                difile,
                line,
                ty_ptr,
                1,
                0,
            )
        } else {
            debuginfo::LLVMDIBuilderCreateAutoVariable(
                dibuilder,
                scope,
                name.as_ptr(),
                name.count_bytes(),
                difile,
                line,
                ty_ptr,
                1,
                0,
                align,
            )
        }
    })
}

fn get_difile_location(dibuilder: LLVMDIBuilderRef, location: &Location) -> LLVMMetadataRef {
    match location {
        Location::Unknown => unsafe {
            debuginfo::LLVMDIBuilderCreateFile(
                dibuilder,
                c"/dev/stdin".as_ptr(),
                c"/dev/stdin".count_bytes(),
                c"".as_ptr(),
                0,
            )
        },
        Location::File(file_location) => get_difile(dibuilder, &file_location.file),
    }
}

fn get_difile(dibuilder: LLVMDIBuilderRef, file: &Path) -> LLVMMetadataRef {
    let parent = if let Some(parent) = file.parent() {
        CString::new(parent.display().to_string()).unwrap()
    } else {
        CString::new("").unwrap()
    };

    let filename = CString::new(file.display().to_string()).unwrap();

    unsafe {
        debuginfo::LLVMDIBuilderCreateFile(
            dibuilder,
            filename.as_ptr(),
            filename.count_bytes(),
            parent.as_ptr(),
            parent.count_bytes(),
        )
    }
}

fn set_loc(
    ctx: LLVMContextRef,
    builder: LLVMBuilderRef,
    location: &Location,
    scope: LLVMMetadataRef,
) -> *mut LLVMOpaqueMetadata {
    match location {
        Location::Unknown => unsafe {
            let loc = debuginfo::LLVMDIBuilderCreateDebugLocation(ctx, 0, 0, scope, null_mut());
            core::LLVMSetCurrentDebugLocation2(builder, loc);
            loc
        },
        Location::File(file_location) => unsafe {
            let loc = debuginfo::LLVMDIBuilderCreateDebugLocation(
                ctx,
                file_location.line,
                file_location.col,
                scope,
                null_mut(),
            );
            core::LLVMSetCurrentDebugLocation2(builder, loc);
            loc
        },
    }
}

fn add_preds(ctx: &mut FnCtx, block_idx: BlockIdx) {
    unsafe {
        let block_ptr = *ctx.blocks.get(&block_idx.to_idx()).unwrap();
        core::LLVMPositionBuilderAtEnd(ctx.builder, block_ptr);

        let preds = ctx.func.find_preds_for(block_idx);
        let mut block_args = Vec::new();

        if !preds.is_empty() {
            let operand_len = preds.first().unwrap().1.len();

            for i in 0..(operand_len) {
                let phy_ty =
                    lower_type(ctx.ctx, ctx.storage, preds.first().unwrap().1[i].get_type());
                let phi_node = core::LLVMBuildPhi(ctx.builder, phy_ty, c"".as_ptr());
                let mut blocks = Vec::new();
                let mut values = Vec::new();
                for (pred_block_idx, operands) in &preds {
                    let pred_ptr = ctx.blocks.get(&pred_block_idx.to_idx()).unwrap();
                    let value = lower_operand(ctx, &operands[i]);

                    blocks.push(*pred_ptr);
                    values.push(value);
                }

                assert_eq!(values.len(), values.len());

                core::LLVMAddIncoming(
                    phi_node,
                    values.as_mut_ptr().cast(),
                    blocks.as_mut_ptr().cast(),
                    blocks.len() as u32,
                );
                block_args.push(phi_node);
            }
        }

        ctx.block_args.insert(block_idx.to_idx(), block_args);
    }
}

fn lower_operand(ctx: &FnCtx, operand: &Operand) -> LLVMValueRef {
    unsafe {
        match operand {
            Operand::Parameter(idx, _ty) => core::LLVMGetParam(ctx.fn_ptr, (*idx) as u32),
            Operand::Value(block_idx, index, _) => *ctx
                .values
                .get(&(block_idx.to_idx(), index.to_idx()))
                .unwrap(),
            Operand::Constant(const_value, ty) => lower_constant(ctx, const_value, *ty),
            Operand::BlockArgument { block_idx, nth, .. } => {
                ctx.block_args.get(block_idx).unwrap()[*nth]
            }
        }
    }
}

fn lower_constant(ctx: &FnCtx, value: &ConstValue, ty: TypeIdx) -> LLVMValueRef {
    unsafe {
        let ty_ptr = lower_type(ctx.ctx, ctx.storage, ty);

        match value {
            irvm::value::ConstValue::Int(value) => core::LLVMConstInt(ty_ptr, *value, 0_i32),
            irvm::value::ConstValue::Float(value) => core::LLVMConstReal(ty_ptr, *value),
            irvm::value::ConstValue::Array(const_values) => {
                let mut values = Vec::new();
                let array_ty = if let Type::Array(array_ty) = &ctx.storage.get_type_info(ty).ty {
                    array_ty
                } else {
                    panic!("type mismatch")
                };

                let typtr = lower_type(ctx.ctx, ctx.storage, array_ty.ty);

                for value in const_values {
                    let ptr = lower_constant(ctx, value, array_ty.ty);
                    values.push(ptr);
                }

                core::LLVMConstArray2(typtr, values.as_mut_ptr(), values.len() as u64)
            }
            irvm::value::ConstValue::Vector(const_values) => {
                let mut values = Vec::new();
                let vec_ty = if let Type::Vector(vec_ty) = &ctx.storage.get_type_info(ty).ty {
                    vec_ty
                } else {
                    panic!("type mismatch")
                };

                for value in const_values {
                    let ptr = lower_constant(ctx, value, vec_ty.ty);
                    values.push(ptr);
                }

                core::LLVMConstVector(values.as_mut_ptr(), values.len() as u32)
            }
            irvm::value::ConstValue::Struct(const_values) => {
                let mut const_fields = Vec::new();
                let struct_ty = if let Type::Struct(struct_ty) = &ctx.storage.get_type_info(ty).ty {
                    &**struct_ty
                } else {
                    panic!("type mismatch")
                };
                for (value, field) in const_values.iter().zip(struct_ty.fields.iter()) {
                    let ptr = lower_constant(ctx, value, *field);
                    const_fields.push(ptr);
                }
                core::LLVMConstStructInContext(
                    ctx.ctx,
                    const_fields.as_mut_ptr(),
                    const_fields.len() as u32,
                    struct_ty.packed as i32,
                )
            }
            irvm::value::ConstValue::NullPtr => core::LLVMConstPointerNull(ty_ptr),
            irvm::value::ConstValue::Undef => core::LLVMGetUndef(ty_ptr),
            irvm::value::ConstValue::Poison => core::LLVMGetPoison(ty_ptr),
        }
    }
}

fn lower_type(ctx: LLVMContextRef, storage: &TypeStorage, ty: TypeIdx) -> LLVMTypeRef {
    let tyinfo = storage.get_type_info(ty);
    unsafe {
        match &tyinfo.ty {
            Type::Int(width) => core::LLVMIntTypeInContext(ctx, *width),
            Type::Half => core::LLVMHalfTypeInContext(ctx),
            Type::BFloat => core::LLVMBFloatTypeInContext(ctx),
            Type::Float => core::LLVMFloatTypeInContext(ctx),
            Type::Double => core::LLVMDoubleTypeInContext(ctx),
            Type::Fp128 => core::LLVMFP128TypeInContext(ctx),
            Type::X86Fp80 => core::LLVMX86FP80TypeInContext(ctx),
            Type::PpcFp128 => core::LLVMPPCFP128TypeInContext(ctx),
            Type::Ptr {
                pointee: _,
                address_space,
            } => core::LLVMPointerTypeInContext(ctx, address_space.unwrap_or(0)),
            Type::Vector(vector_type) => {
                let inner = lower_type(ctx, storage, vector_type.ty);
                core::LLVMVectorType(inner, vector_type.size)
            }
            Type::Array(array_type) => {
                let inner = lower_type(ctx, storage, array_type.ty);
                core::LLVMArrayType2(inner, array_type.size)
            }
            Type::Struct(struct_type) => {
                let mut fields = Vec::new();

                for field in struct_type.fields.iter() {
                    fields.push(lower_type(ctx, storage, *field));
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

fn lower_debug_type(
    datalayout: &DataLayout,
    builder: LLVMDIBuilderRef,
    storage: &TypeStorage,
    type_idx: TypeIdx,
) -> LLVMMetadataRef {
    let ty = storage.get_type_info(type_idx);

    let name = CString::new(ty.debug_name.clone().unwrap_or_default()).unwrap();

    // 1 == address
    // 2 = boolean
    // 4 = float
    // 5 = signed
    // 11 = numeric string
    // https://dwarfstd.org/doc/DWARF5.pdf#section.7.8

    let size_in_bits = datalayout.get_type_size(storage, type_idx);
    let align_in_bits = datalayout.get_type_align(storage, type_idx);

    unsafe {
        match &ty.ty {
            Type::Int(width) => {
                let mut encoding = 0x7;
                if *width == 1 {
                    encoding = 0x2;
                }
                debuginfo::LLVMDIBuilderCreateBasicType(
                    builder,
                    name.as_ptr(),
                    name.count_bytes(),
                    size_in_bits as u64,
                    encoding,
                    LLVMDIFlagPublic,
                )
            }
            Type::Half
            | Type::BFloat
            | Type::Float
            | Type::Double
            | Type::Fp128
            | Type::X86Fp80
            | Type::PpcFp128 => debuginfo::LLVMDIBuilderCreateBasicType(
                builder,
                name.as_ptr(),
                name.count_bytes(),
                size_in_bits as u64,
                0x4,
                LLVMDIFlagPublic,
            ),
            Type::Ptr {
                pointee,
                address_space,
            } => {
                let pointee_ptr = lower_debug_type(datalayout, builder, storage, *pointee);
                debuginfo::LLVMDIBuilderCreatePointerType(
                    builder,
                    pointee_ptr,
                    size_in_bits as u64,
                    align_in_bits,
                    address_space.unwrap_or(0),
                    name.as_ptr(),
                    name.count_bytes(),
                )
            }
            Type::Vector(_vector_type) => todo!(),
            Type::Array(_array_type) => todo!(),
            Type::Struct(_struct_type) => todo!(),
        }
    }
}
