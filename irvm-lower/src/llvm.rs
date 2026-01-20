//! ## LLVM IR lowering
//!
//! Lower irvm IR to LLVM IR.
//!
//! The main functions to use are [`lower_module_to_llvmir`] which gives you a [`CompileResult`].
//!
//! With this result you can either write it to a file (.ll), compile it to an object file
//! or create a JIT Engine to execute methods (a bit limited currently, useful for testing).
//!
//! ```rust,ignore
//!
//! // Lower the module.
//! let result = lower_module_to_llvmir(&mymodule, &my_type_storage)?;
//!
//! // Output to a file as a .ll
//! output_to_file(&result, Path::new("out.ll"))?;
//!
//! // Compile to an object file.
//! compile_object(&result, &mymodule.target_triple, CompileOptions::default(), Path::new("out.o"), false)?;
//!
//! // Or create a JIT engine.
//! let engine = create_jit_engine(result, 3)?;
//! let res = unsafe { engine.execute("main", &[JitValue::U32(4)], JitValue::U32(0))? };
//! ```

use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_void},
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    ptr::null_mut,
    rc::Rc,
    sync::OnceLock,
};

use gimli::{DW_ATE_boolean, DW_ATE_float, DW_ATE_unsigned, DW_TAG_reference_type};
use irvm::{
    block::{
        AtomicOrdering, AtomicRMWOp, BlockIdx, DebugOp, DebugVariable, FastMathFlags, Instruction,
        SyncScope,
    },
    common::{Linkage, Location},
    datalayout::DataLayout,
    function::Function,
    module::Module,
    target_lexicon::Triple,
    types::{Type, TypeIdx, TypeStorage},
    value::{ConstValue, Operand},
};

use itertools::Itertools;
use llvm_sys::{
    LLVMIntPredicate, LLVMModule, LLVMOpaqueMetadata, LLVMRealPredicate,
    core::{self, LLVMDisposeMessage, LLVMDumpModule},
    debuginfo::{self, LLVMDIFlagPublic, LLVMDWARFEmissionKind},
    error::LLVMGetErrorMessage,
    execution_engine::{self, LLVMExecutionEngineRef, LLVMLinkInMCJIT},
    prelude::{
        LLVMBasicBlockRef, LLVMBuilderRef, LLVMContextRef, LLVMDIBuilderRef, LLVMMetadataRef,
        LLVMTypeRef, LLVMValueRef,
    },
    target::{
        LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs,
        LLVM_InitializeAllTargets, LLVM_InitializeNativeAsmParser, LLVM_InitializeNativeAsmPrinter,
        LLVM_InitializeNativeDisassembler, LLVM_InitializeNativeTarget,
    },
    target_machine::{
        self, LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMDisposeTargetMachine,
        LLVMGetHostCPUFeatures, LLVMGetHostCPUName, LLVMRelocMode, LLVMTargetMachineEmitToFile,
    },
    transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    },
};

#[derive(Debug)]
pub enum OutputCompilation {
    File(PathBuf),
    Engine(*mut LLVMExecutionEngineRef),
}

/// A possible Error.
#[derive(Debug, thiserror::Error, Clone)]
pub enum Error {
    #[error("llvm error: {:?}", 0)]
    LLVMError(String),
    #[error("jit error: {:?}", 0)]
    JitError(String),
    #[error(transparent)]
    NulError(#[from] std::ffi::NulError),
    #[error("irvm error: {:?}", 0)]
    IRVMError(#[from] irvm::error::Error),
}

/// The target LLVM cpu.
#[derive(Debug, Clone, Default)]
pub enum TargetCpu {
    #[default]
    Host,
    Name(String),
}

/// The target LLVM cpu features.
#[derive(Debug, Clone, Default)]
pub enum TargetCpuFeatures {
    #[default]
    Host,
    Features(String),
}

/// The optimization level to use.
#[derive(Debug, Clone, Default)]
pub enum OptLevel {
    None,
    Less,
    #[default]
    Default,
    Aggressive,
}

/// The relocation model types supported by LLVM
#[derive(Debug, Clone, Default)]
pub enum RelocModel {
    /// Generated code will assume the default for a particular target architecture.
    #[default]
    Default,
    /// Generated code will exist at static offsets.
    Static,
    /// Generated code will be position-independent.
    Pic,
    /// Generated code will not be position-independent and may be used in static or dynamic executables but not necessarily a shared library.
    DynamicNoPic,
    /// Generated code will be compiled in read-only position independent mode.
    /// In this mode, all read-only data and functions are at a link-time constant offset from the program counter.
    /// ROPI is not supported by all target architectures and calling conventions. It is a particular feature of ARM targets, though.
    /// ROPI may be useful to avoid committing to compile-time constant locations for code in memory.
    Ropi,
    /// Generated code will be compiled in read-write position independent mode.
    ///
    /// In this mode, all writable data is at a link-time constant offset from the static base register.
    ///
    /// RWPI is not supported by all target architectures and calling conventions. It is a particular feature of ARM targets, though.
    ///
    /// RWPI may be useful to avoid committing to compile-time constant locations for code in memory
    Rwpi,
    /// Combines the ropi and rwpi modes. Generated code will be compiled in both read-only and read-write position independent modes.
    /// All read-only data appears at a link-time constant offset from the program counter,
    /// and all writable data appears at a link-time constant offset from the static base register.
    RopiRwpi,
}

/// The code model supported by LLVM.
#[derive(Debug, Clone, Default)]
pub enum CodeModel {
    #[default]
    Default,
    JitDefault,
    Tiny,
    Small,
    Kernel,
    Medium,
    Large,
}

/// Compile options to generate the object file.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    pub target_cpu: TargetCpu,
    pub target_cpu_features: TargetCpuFeatures,
    pub relocation_model: RelocModel,
    pub code_model: CodeModel,
    pub opt_level: u8,
}

/// A compile result from lowering a given Module.
#[derive(Debug)]
pub struct CompileResult {
    context: LLVMContextRef,
    module: *mut LLVMModule,
}

/// A prepared JIT engine.
#[derive(Debug)]
pub struct JitEngine {
    context: LLVMContextRef,
    engine: LLVMExecutionEngineRef,
}

impl Drop for CompileResult {
    fn drop(&mut self) {
        unsafe {
            core::LLVMDisposeModule(self.module);
            core::LLVMContextDispose(self.context);
        }
    }
}

impl Drop for JitEngine {
    fn drop(&mut self) {
        unsafe {
            execution_engine::LLVMDisposeExecutionEngine(self.engine);
            core::LLVMContextDispose(self.context);
        }
    }
}

/// Possible value/types to pass to the JIT engine execute method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitValue {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ptr(*mut c_void),
    Void,
}

impl JitEngine {
    /// Execute the given function.
    ///
    /// # Safety
    ///
    /// All arguments and return type must match the signature.
    pub unsafe fn execute(
        &self,
        symbol: &str,
        args: &[JitValue],
        return_ty: JitValue,
    ) -> Result<JitValue, Error> {
        unsafe {
            let sym = CString::new(symbol)?;
            let mut out_fn = null_mut();
            let ok = execution_engine::LLVMFindFunction(self.engine, sym.as_ptr(), &raw mut out_fn);

            if ok != 0 {
                return Err(Error::LLVMError("Function not found".to_string()));
            }

            let mut value_args = Vec::new();

            for arg in args {
                let value = match arg {
                    JitValue::U8(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt8Type(),
                        (*value) as _,
                        0,
                    ),
                    JitValue::U16(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt16Type(),
                        (*value) as _,
                        0,
                    ),
                    JitValue::U32(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt32Type(),
                        (*value) as _,
                        0,
                    ),
                    JitValue::U64(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt64Type(),
                        (*value) as _,
                        0,
                    ),
                    JitValue::I8(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt8Type(),
                        (*value) as _,
                        1,
                    ),
                    JitValue::I16(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt16Type(),
                        (*value) as _,
                        1,
                    ),
                    JitValue::I32(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt32Type(),
                        (*value) as _,
                        1,
                    ),
                    JitValue::I64(value) => execution_engine::LLVMCreateGenericValueOfInt(
                        core::LLVMInt64Type(),
                        (*value) as _,
                        1,
                    ),
                    JitValue::F32(value) => execution_engine::LLVMCreateGenericValueOfFloat(
                        core::LLVMFloatType(),
                        (*value) as _,
                    ),
                    JitValue::F64(value) => execution_engine::LLVMCreateGenericValueOfFloat(
                        core::LLVMDoubleType(),
                        (*value) as _,
                    ),
                    JitValue::Ptr(value) => {
                        execution_engine::LLVMCreateGenericValueOfPointer(*value)
                    }
                    JitValue::Void => {
                        return Err(Error::JitError(
                            "can't use void jit value as argument".to_string(),
                        ));
                    }
                };
                value_args.push(value);
            }

            let result = execution_engine::LLVMRunFunction(
                self.engine,
                out_fn,
                value_args.len() as _,
                value_args.as_mut_ptr(),
            );

            let res = match return_ty {
                JitValue::U8(_) => {
                    JitValue::U8(execution_engine::LLVMGenericValueToInt(result, 0) as u8)
                }
                JitValue::U16(_) => {
                    JitValue::U16(execution_engine::LLVMGenericValueToInt(result, 0) as u16)
                }
                JitValue::U32(_) => {
                    JitValue::U32(execution_engine::LLVMGenericValueToInt(result, 0) as u32)
                }
                JitValue::U64(_) => {
                    JitValue::U64(execution_engine::LLVMGenericValueToInt(result, 0) as u64)
                }
                JitValue::I8(_) => {
                    JitValue::I8(execution_engine::LLVMGenericValueToInt(result, 1) as i8)
                }
                JitValue::I16(_) => {
                    JitValue::I16(execution_engine::LLVMGenericValueToInt(result, 1) as i16)
                }
                JitValue::I32(_) => {
                    JitValue::I32(execution_engine::LLVMGenericValueToInt(result, 1) as i32)
                }
                JitValue::I64(_) => {
                    JitValue::I64(execution_engine::LLVMGenericValueToInt(result, 1) as i64)
                }
                JitValue::F32(_) => JitValue::F32(execution_engine::LLVMGenericValueToFloat(
                    core::LLVMFloatType(),
                    result,
                ) as f32),
                JitValue::F64(_) => JitValue::F64(execution_engine::LLVMGenericValueToFloat(
                    core::LLVMDoubleType(),
                    result,
                ) as f64),
                JitValue::Ptr(_) => {
                    JitValue::Ptr(execution_engine::LLVMGenericValueToPointer(result))
                }
                JitValue::Void => JitValue::Void,
            };

            for arg in &value_args {
                execution_engine::LLVMDisposeGenericValue(*arg);
            }
            execution_engine::LLVMDisposeGenericValue(result);

            Ok(res)
        }
    }
}

impl CompileResult {
    pub fn dump(&self) {
        unsafe {
            LLVMDumpModule(self.module);
        }
    }
}

/// Lowers the given module to llvm ir.
pub fn lower_module_to_llvmir(
    module: &Module,
    storage: &TypeStorage,
) -> Result<CompileResult, Error> {
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

        // Lower global variables
        let mut globals: HashMap<usize, LLVMValueRef> = Default::default();
        for (global_idx, global) in module.globals().iter() {
            let name = CString::new(global.name.as_str()).unwrap();
            let ty = lower_type(ctx, storage, global.ty);

            let global_var = core::LLVMAddGlobal(llvm_module, ty, name.as_ptr());

            // Set initializer if present
            if let Some(init) = &global.initializer {
                let init_val = lower_global_constant(ctx, storage, init, global.ty);
                core::LLVMSetInitializer(global_var, init_val);
            }

            // Set constant flag
            core::LLVMSetGlobalConstant(global_var, global.is_constant as i32);

            // Set linkage if specified
            if let Some(linkage) = &global.linkage {
                core::LLVMSetLinkage(global_var, lower_linkage(linkage));
            }

            // Set alignment if specified
            if let Some(align) = global.align {
                core::LLVMSetAlignment(global_var, align);
            }

            globals.insert(global_idx.to_idx(), global_var);
        }
        let globals = Rc::new(globals);

        for (fun_idx, func) in module.functions().iter() {
            let name = CString::new(func.name.as_str()).unwrap();

            let ret_ty = if let Some(ret_ty) = func.result_type {
                lower_type(ctx, storage, ret_ty)
            } else {
                core::LLVMVoidTypeInContext(ctx)
            };
            let mut params = func
                .parameters
                .iter()
                .map(|x| lower_type(ctx, storage, x.ty))
                .collect_vec();
            let fn_ty = core::LLVMFunctionType(ret_ty, params.as_mut_ptr(), params.len() as u32, 0);
            let fn_ptr = core::LLVMAddFunction(llvm_module, name.as_ptr(), fn_ty);
            apply_function_attrs(ctx, fn_ptr, &func.attrs);
            apply_parameter_attrs(ctx, fn_ptr, &func.parameters);
            apply_return_attrs(ctx, fn_ptr, &func.return_attrs);

            // Set GC name if specified
            if let Some(gc_name) = &func.gc_name {
                let gc_cstring = CString::new(gc_name.as_str()).unwrap();
                core::LLVMSetGC(fn_ptr, gc_cstring.as_ptr());
            }

            // Set prefix data if specified
            if let Some((value, ty)) = &func.prefix_data {
                let llvm_val = lower_global_constant(ctx, storage, value, *ty);
                core::LLVMSetPrefixData(fn_ptr, llvm_val);
            }

            // Set prologue data if specified
            if let Some((value, ty)) = &func.prologue_data {
                let llvm_val = lower_global_constant(ctx, storage, value, *ty);
                core::LLVMSetPrologueData(fn_ptr, llvm_val);
            }

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
                let ptr = lower_debug_type(
                    &module.data_layout,
                    dibuilder,
                    storage,
                    debug_module,
                    param.ty,
                );
                debug_param_types.push(ptr);
            }

            let debug_func_ty = debuginfo::LLVMDIBuilderCreateSubroutineType(
                dibuilder,
                file,
                debug_param_types.as_mut_ptr(),
                debug_param_types.len() as u32,
                0,
            );

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

        for (fun_idx, func) in module.functions().iter() {
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
                globals: Rc::clone(&globals),
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

        debuginfo::LLVMDIBuilderFinalize(dibuilder);
        debuginfo::LLVMDisposeDIBuilder(dibuilder);
        core::LLVMDisposeBuilder(builder);

        if std::env::var("IRVM_DUMP_IR").is_ok() {
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
        }

        Ok(CompileResult {
            context: ctx,
            module: llvm_module,
        })
    }
}

/// Compiles the given llvm compile result to an object or assembly file.
///
/// If output assembly is false it will output an object file.
pub fn compile_object(
    compile_result: &CompileResult,
    target_triple: Triple,
    options: CompileOptions,
    output_file: &Path,
    output_assembly: bool,
) -> Result<(), Error> {
    unsafe {
        static INITIALIZED: OnceLock<()> = OnceLock::new();
        INITIALIZED.get_or_init(|| {
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmPrinters();
        });

        let target_triple = CString::new(target_triple.to_string())?;

        let target_cpu = match &options.target_cpu {
            TargetCpu::Host => {
                let cpu = LLVMGetHostCPUName();
                CString::from(CStr::from_ptr(cpu))
            }
            TargetCpu::Name(name) => CString::new(name.as_bytes())?,
        };

        let target_cpu_features = match &options.target_cpu_features {
            TargetCpuFeatures::Host => {
                let cpu = LLVMGetHostCPUFeatures();
                CString::from(CStr::from_ptr(cpu))
            }
            TargetCpuFeatures::Features(name) => CString::new(name.as_bytes())?,
        };

        let mut out_msg = null_mut();

        let mut target = null_mut();

        let ok = target_machine::LLVMGetTargetFromTriple(
            target_triple.as_ptr(),
            &raw mut target,
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

            return Err(Error::LLVMError(msg));
        }

        if !out_msg.is_null() {
            core::LLVMDisposeMessage(out_msg);
        }

        let machine = target_machine::LLVMCreateTargetMachine(
            target,
            target_triple.as_ptr(),
            target_cpu.as_ptr(),
            target_cpu_features.as_ptr(),
            match options.opt_level {
                0 => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
                1 => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
                2 => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
                _ => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            },
            match options.relocation_model {
                RelocModel::Default => LLVMRelocMode::LLVMRelocDefault,
                RelocModel::Static => LLVMRelocMode::LLVMRelocStatic,
                RelocModel::Pic => LLVMRelocMode::LLVMRelocPIC,
                RelocModel::DynamicNoPic => LLVMRelocMode::LLVMRelocDynamicNoPic,
                RelocModel::Ropi => LLVMRelocMode::LLVMRelocROPI,
                RelocModel::Rwpi => LLVMRelocMode::LLVMRelocRWPI,
                RelocModel::RopiRwpi => LLVMRelocMode::LLVMRelocROPI_RWPI,
            },
            match options.code_model {
                CodeModel::Default => LLVMCodeModel::LLVMCodeModelDefault,
                CodeModel::JitDefault => LLVMCodeModel::LLVMCodeModelJITDefault,
                CodeModel::Tiny => LLVMCodeModel::LLVMCodeModelTiny,
                CodeModel::Small => LLVMCodeModel::LLVMCodeModelSmall,
                CodeModel::Kernel => LLVMCodeModel::LLVMCodeModelKernel,
                CodeModel::Medium => LLVMCodeModel::LLVMCodeModelMedium,
                CodeModel::Large => LLVMCodeModel::LLVMCodeModelLarge,
            },
        );

        let opts = LLVMCreatePassBuilderOptions();

        let passes = CString::new(format!("default<O{}>", options.opt_level)).unwrap();

        let error = LLVMRunPasses(compile_result.module, passes.as_ptr(), machine, opts);

        if !error.is_null() {
            let msg_ptr = LLVMGetErrorMessage(error);
            let msg = {
                let msg = CStr::from_ptr(msg_ptr);
                msg.to_string_lossy().to_string()
            };
            LLVMDisposeMessage(msg_ptr);
            LLVMDisposeTargetMachine(machine);
            return Err(Error::LLVMError(msg));
        }

        LLVMDisposePassBuilderOptions(opts);

        let mut out_msg: *mut i8 = null_mut();
        let ok = llvm_sys::analysis::LLVMVerifyModule(
            compile_result.module,
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

            LLVMDisposeTargetMachine(machine);

            return Err(Error::LLVMError(msg));
        }

        let filename = CString::new(output_file.as_os_str().to_string_lossy().as_bytes()).unwrap();

        let ok = LLVMTargetMachineEmitToFile(
            machine,
            compile_result.module,
            filename.as_ptr().cast_mut(),
            if output_assembly {
                LLVMCodeGenFileType::LLVMAssemblyFile
            } else {
                LLVMCodeGenFileType::LLVMObjectFile
            },
            &raw mut out_msg,
        );

        LLVMDisposeTargetMachine(machine);

        if ok != 0 {
            let msg = {
                let msg = CStr::from_ptr(out_msg);
                msg.to_string_lossy().to_string()
            };

            if !out_msg.is_null() {
                core::LLVMDisposeMessage(out_msg);
            }

            return Err(Error::LLVMError(msg));
        }

        Ok(())
    }
}

/// Outputs the given compile result to a llvm ir file.
pub fn output_to_file(compile_result: &CompileResult, output_ll: &Path) -> Result<(), Error> {
    unsafe {
        let file = CString::new(&*output_ll.to_string_lossy())?;

        let mut out_msg: *mut i8 = null_mut();

        let ok =
            core::LLVMPrintModuleToFile(compile_result.module, file.as_ptr(), &raw mut out_msg);

        if ok != 0 {
            let msg = {
                let msg = CStr::from_ptr(out_msg);
                msg.to_string_lossy().to_string()
            };

            if !out_msg.is_null() {
                core::LLVMDisposeMessage(out_msg);
            }

            return Err(Error::LLVMError(msg));
        }

        if !out_msg.is_null() {
            core::LLVMDisposeMessage(out_msg);
        }

        Ok(())
    }
}

/// Creates a jit engine for the given compile result.
pub fn create_jit_engine(result: CompileResult, optlevel: u32) -> Result<JitEngine, Error> {
    unsafe {
        let mut engine = null_mut();

        let mut out_msg: *mut i8 = null_mut();

        let result = ManuallyDrop::new(result);

        static INITIALIZED: OnceLock<()> = OnceLock::new();
        INITIALIZED.get_or_init(|| {
            LLVM_InitializeNativeTarget();
            LLVM_InitializeNativeAsmParser();
            LLVM_InitializeNativeAsmPrinter();
            LLVM_InitializeNativeDisassembler();
            LLVMLinkInMCJIT();
        });

        let ok = execution_engine::LLVMCreateJITCompilerForModule(
            &raw mut engine,
            result.module,
            optlevel,
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

            return Err(Error::LLVMError(msg));
        }

        let engine = JitEngine {
            context: result.context,
            engine,
        };

        Ok(engine)
    }
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
    globals: Rc<HashMap<usize, LLVMValueRef>>,
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

        for (inst_idx, (loc, inst)) in ctx.func.blocks[block_idx].instructions().iter() {
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
                    irvm::block::BinaryOp::FAdd { lhs, rhs, flags } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFAdd(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        apply_fast_math_flags(value, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FSub { lhs, rhs, flags } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFSub(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        apply_fast_math_flags(value, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FMul { lhs, rhs, flags } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFMul(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        apply_fast_math_flags(value, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FDiv { lhs, rhs, flags } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFDiv(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        apply_fast_math_flags(value, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FRem { lhs, rhs, flags } => {
                        let lhs_ptr = lower_operand(ctx, lhs);
                        let rhs_ptr = lower_operand(ctx, rhs);
                        let value =
                            core::LLVMBuildFRem(ctx.builder, lhs_ptr, rhs_ptr, null_name.as_ptr());
                        apply_fast_math_flags(value, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::BinaryOp::FNeg { value, flags } => {
                        let val_ptr = lower_operand(ctx, value);
                        let result = core::LLVMBuildFNeg(ctx.builder, val_ptr, null_name.as_ptr());
                        apply_fast_math_flags(result, flags);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
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
                    irvm::block::VectorOp::InsertElement {
                        vector,
                        element,
                        idx,
                    } => {
                        let vec_val = lower_operand(ctx, vector);
                        let elem_val = lower_operand(ctx, element);
                        let idx_val = lower_operand(ctx, idx);
                        let value = core::LLVMBuildInsertElement(
                            ctx.builder,
                            vec_val,
                            elem_val,
                            idx_val,
                            null_name.as_ptr(),
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::VectorOp::ShuffleVector { vec1, vec2, mask } => {
                        let v1 = lower_operand(ctx, vec1);
                        let v2 = lower_operand(ctx, vec2);

                        // Create mask as constant vector
                        let i32_ty = core::LLVMInt32TypeInContext(ctx.ctx);
                        let mut mask_vals: Vec<LLVMValueRef> = mask
                            .iter()
                            .map(|&m| {
                                if m < 0 {
                                    core::LLVMGetUndef(i32_ty)
                                } else {
                                    core::LLVMConstInt(i32_ty, m as u64, 0)
                                }
                            })
                            .collect();
                        let mask_val =
                            core::LLVMConstVector(mask_vals.as_mut_ptr(), mask_vals.len() as u32);

                        let value = core::LLVMBuildShuffleVector(
                            ctx.builder,
                            v1,
                            v2,
                            mask_val,
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
                            let const_val = core::LLVMConstInt(
                                core::LLVMInt64TypeInContext(ctx.ctx),
                                (*num_elements) as u64,
                                0,
                            );
                            core::LLVMBuildArrayAlloca(
                                ctx.builder,
                                ty_ptr,
                                const_val,
                                null_name.as_ptr(),
                            )
                        } else {
                            core::LLVMBuildAlloca(ctx.builder, ty_ptr, null_name.as_ptr())
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::MemoryOp::Load {
                        ptr,
                        align,
                        volatile,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let ty_ptr =
                            lower_type(ctx.ctx, ctx.storage, ptr.get_inner_type(ctx.storage)?);

                        let value =
                            core::LLVMBuildLoad2(ctx.builder, ty_ptr, ptr_val, null_name.as_ptr());
                        if let Some(align) = align {
                            core::LLVMSetAlignment(value, *align / 8);
                        }
                        if *volatile {
                            core::LLVMSetVolatile(value, 1);
                        }

                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::MemoryOp::Store {
                        value,
                        ptr,
                        align,
                        volatile,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let value_val = lower_operand(ctx, value);

                        let value = core::LLVMBuildStore(ctx.builder, value_val, ptr_val);
                        if let Some(align) = align {
                            core::LLVMSetAlignment(value, *align / 8);
                        }
                        if *volatile {
                            core::LLVMSetVolatile(value, 1);
                        }

                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::MemoryOp::GetElementPtr { ptr, indices } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let pointee_ty =
                            lower_type(ctx.ctx, ctx.storage, ptr.get_inner_type(ctx.storage)?);

                        let mut x = Vec::new();

                        for index in indices {
                            let value = match index {
                                irvm::block::GepIndex::Const(value) => core::LLVMConstInt(
                                    core::LLVMInt64TypeInContext(ctx.ctx),
                                    (*value) as u64,
                                    0,
                                ),
                                irvm::block::GepIndex::Value(operand) => {
                                    lower_operand(ctx, operand)
                                }
                            };

                            x.push(value);
                        }

                        let value = core::LLVMBuildGEP2(
                            ctx.builder,
                            pointee_ty,
                            ptr_val,
                            x.as_mut_ptr(),
                            x.len() as u32,
                            null_name.as_ptr(),
                        );

                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::MemoryOp::AtomicLoad {
                        ptr,
                        ordering,
                        align,
                        sync_scope,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let ty_ptr =
                            lower_type(ctx.ctx, ctx.storage, ptr.get_inner_type(ctx.storage)?);

                        let value =
                            core::LLVMBuildLoad2(ctx.builder, ty_ptr, ptr_val, null_name.as_ptr());
                        core::LLVMSetOrdering(value, lower_atomic_ordering(ordering));
                        core::LLVMSetVolatile(value, 0);
                        if let Some(a) = align {
                            core::LLVMSetAlignment(value, *a / 8);
                        }
                        let _ = sync_scope; // TODO: handle sync scope when LLVM-sys supports it
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::MemoryOp::AtomicStore {
                        value,
                        ptr,
                        ordering,
                        align,
                        sync_scope,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let val = lower_operand(ctx, value);
                        let store = core::LLVMBuildStore(ctx.builder, val, ptr_val);
                        core::LLVMSetOrdering(store, lower_atomic_ordering(ordering));
                        if let Some(a) = align {
                            core::LLVMSetAlignment(store, *a / 8);
                        }
                        let _ = sync_scope; // TODO: handle sync scope when LLVM-sys supports it
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), store);
                    }
                    irvm::block::MemoryOp::AtomicRMW {
                        op,
                        ptr,
                        value,
                        ordering,
                        sync_scope,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let val = lower_operand(ctx, value);
                        let result = core::LLVMBuildAtomicRMW(
                            ctx.builder,
                            lower_atomic_rmw_op(op),
                            ptr_val,
                            val,
                            lower_atomic_ordering(ordering),
                            matches!(sync_scope, SyncScope::SingleThread) as i32,
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::MemoryOp::CmpXchg {
                        ptr,
                        cmp,
                        new_val,
                        success_ordering,
                        failure_ordering,
                        weak,
                        sync_scope,
                    } => {
                        let ptr_val = lower_operand(ctx, ptr);
                        let cmp_val = lower_operand(ctx, cmp);
                        let new_val_ptr = lower_operand(ctx, new_val);
                        let result = core::LLVMBuildAtomicCmpXchg(
                            ctx.builder,
                            ptr_val,
                            cmp_val,
                            new_val_ptr,
                            lower_atomic_ordering(success_ordering),
                            lower_atomic_ordering(failure_ordering),
                            matches!(sync_scope, SyncScope::SingleThread) as i32,
                        );
                        core::LLVMSetWeak(result, *weak as i32);
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::MemoryOp::Fence {
                        ordering,
                        sync_scope,
                    } => {
                        core::LLVMBuildFence(
                            ctx.builder,
                            lower_atomic_ordering(ordering),
                            matches!(sync_scope, SyncScope::SingleThread) as i32,
                            null_name.as_ptr(),
                        );
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
                            }
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
                    irvm::block::OtherOp::Select {
                        cond,
                        true_val,
                        false_val,
                    } => {
                        let cond_val = lower_operand(ctx, cond);
                        let true_ptr = lower_operand(ctx, true_val);
                        let false_ptr = lower_operand(ctx, false_val);
                        let value = core::LLVMBuildSelect(
                            ctx.builder,
                            cond_val,
                            true_ptr,
                            false_ptr,
                            null_name.as_ptr(),
                        );
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::OtherOp::LandingPad {
                        result_ty,
                        cleanup,
                        clauses,
                    } => {
                        let ty = lower_type(ctx.ctx, ctx.storage, *result_ty);
                        let lp = core::LLVMBuildLandingPad(
                            ctx.builder,
                            ty,
                            null_mut(), // personality function is set on the function
                            clauses.len() as u32,
                            null_name.as_ptr(),
                        );

                        if *cleanup {
                            core::LLVMSetCleanup(lp, 1);
                        }

                        for clause in clauses {
                            match clause {
                                irvm::block::LandingPadClause::Catch(operand) => {
                                    let catch_val = lower_operand(ctx, operand);
                                    core::LLVMAddClause(lp, catch_val);
                                }
                                irvm::block::LandingPadClause::Filter(operands) => {
                                    // Filter is an array of type infos
                                    let i8_ptr_ty = core::LLVMPointerType(
                                        core::LLVMInt8TypeInContext(ctx.ctx),
                                        0,
                                    );
                                    let mut filter_vals: Vec<_> =
                                        operands.iter().map(|o| lower_operand(ctx, o)).collect();
                                    let filter_arr = core::LLVMConstArray2(
                                        i8_ptr_ty,
                                        filter_vals.as_mut_ptr(),
                                        filter_vals.len() as u64,
                                    );
                                    core::LLVMAddClause(lp, filter_arr);
                                }
                            }
                        }

                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), lp);
                    }
                    irvm::block::OtherOp::Intrinsic(intrinsic) => {
                        let value = lower_intrinsic(ctx, intrinsic, block_idx, inst_idx)?;
                        if !value.is_null() {
                            ctx.values
                                .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                        }
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
                    DebugOp::Assign { .. } => {
                        // TODO: no di assign in llvm sys?
                        todo!()
                    }
                },
                Instruction::ConversionOp(conv_op) => match conv_op {
                    irvm::block::ConversionOp::Trunc { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result = core::LLVMBuildTrunc(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::ZExt { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result = core::LLVMBuildZExt(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::SExt { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result = core::LLVMBuildSExt(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::FPTrunc { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildFPTrunc(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::FPExt { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result = core::LLVMBuildFPExt(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::FPToUI { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildFPToUI(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::FPToSI { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildFPToSI(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::UIToFP { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildUIToFP(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::SIToFP { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildSIToFP(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::PtrToInt { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildPtrToInt(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::IntToPtr { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildIntToPtr(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::Bitcast { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildBitCast(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                    irvm::block::ConversionOp::AddrSpaceCast { value, target_ty } => {
                        let val = lower_operand(ctx, value);
                        let ty = lower_type(ctx.ctx, ctx.storage, *target_ty);
                        let result =
                            core::LLVMBuildAddrSpaceCast(ctx.builder, val, ty, null_name.as_ptr());
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), result);
                    }
                },
                Instruction::AggregateOp(agg_op) => match agg_op {
                    irvm::block::AggregateOp::ExtractValue { aggregate, indices } => {
                        let agg_val = lower_operand(ctx, aggregate);
                        let value = if indices.len() == 1 {
                            core::LLVMBuildExtractValue(
                                ctx.builder,
                                agg_val,
                                indices[0],
                                null_name.as_ptr(),
                            )
                        } else {
                            // For multiple indices, we need to chain ExtractValue calls
                            let mut result = agg_val;
                            for &idx in indices {
                                result = core::LLVMBuildExtractValue(
                                    ctx.builder,
                                    result,
                                    idx,
                                    null_name.as_ptr(),
                                );
                            }
                            result
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                    irvm::block::AggregateOp::InsertValue {
                        aggregate,
                        element,
                        indices,
                    } => {
                        let agg_val = lower_operand(ctx, aggregate);
                        let elem_val = lower_operand(ctx, element);
                        let value = if indices.len() == 1 {
                            core::LLVMBuildInsertValue(
                                ctx.builder,
                                agg_val,
                                elem_val,
                                indices[0],
                                null_name.as_ptr(),
                            )
                        } else {
                            // For multiple indices, we need to extract nested aggregates,
                            // insert at the innermost, and rebuild
                            // For simplicity, this implementation only handles single index
                            // A full implementation would need recursive extraction
                            core::LLVMBuildInsertValue(
                                ctx.builder,
                                agg_val,
                                elem_val,
                                indices[0],
                                null_name.as_ptr(),
                            )
                        };
                        ctx.values
                            .insert((block_idx.to_idx(), inst_idx.to_idx()), value);
                    }
                },
            }
        }

        match ctx.func.blocks[block_idx].terminator().clone() {
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
            irvm::block::Terminator::Switch {
                value,
                default_block,
                cases,
                location,
                ..
            } => {
                set_loc(ctx.ctx, ctx.builder, &location, ctx.debug_scope);
                let switch_val = lower_operand(ctx, &value);
                let default_bb = *ctx.blocks.get(&default_block.to_idx()).unwrap();

                let switch_instr =
                    core::LLVMBuildSwitch(ctx.builder, switch_val, default_bb, cases.len() as u32);

                let val_ty = lower_type(ctx.ctx, ctx.storage, value.get_type());

                for case in cases {
                    let case_bb = *ctx.blocks.get(&case.block.to_idx()).unwrap();
                    let case_val = core::LLVMConstInt(val_ty, case.value, 0);
                    core::LLVMAddCase(switch_instr, case_val, case_bb);
                }
            }
            irvm::block::Terminator::Invoke {
                call,
                normal_dest,
                unwind_dest,
                location,
                ..
            } => {
                set_loc(ctx.ctx, ctx.builder, &location, ctx.debug_scope);

                let (target_fn_ptr, fn_ty) = match &call.fn_target {
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
                    }
                };

                let mut args = call
                    .params
                    .iter()
                    .map(|p| lower_operand(ctx, p))
                    .collect_vec();

                let normal_bb = *ctx.blocks.get(&normal_dest.to_idx()).unwrap();
                let unwind_bb = *ctx.blocks.get(&unwind_dest.to_idx()).unwrap();

                core::LLVMBuildInvoke2(
                    ctx.builder,
                    fn_ty,
                    target_fn_ptr,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    normal_bb,
                    unwind_bb,
                    c"".as_ptr(),
                );
            }
            irvm::block::Terminator::Resume { value, location } => {
                set_loc(ctx.ctx, ctx.builder, &location, ctx.debug_scope);
                let val = lower_operand(ctx, &value);
                core::LLVMBuildResume(ctx.builder, val);
            }
            irvm::block::Terminator::Unreachable { location } => {
                set_loc(ctx.ctx, ctx.builder, &location, ctx.debug_scope);
                core::LLVMBuildUnreachable(ctx.builder);
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
            format!("bb{name}")
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

    let ty_ptr = lower_debug_type(datalayout, dibuilder, storage, scope, variable.ty);
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
            Operand::Global(global_idx, _ty) => *ctx
                .globals
                .get(&global_idx.to_idx())
                .expect("global not found"),
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
    module_scope: LLVMMetadataRef,
    type_idx: TypeIdx,
) -> LLVMMetadataRef {
    let ty = storage.get_type_info(type_idx);

    // 1 == address
    // 2 = boolean
    // 4 = float
    // 5 = signed
    // 11 = numeric string
    // https://dwarfstd.org/doc/DWARF5.pdf#section.7.8

    let size_in_bits = datalayout.get_type_size(storage, type_idx);
    let align_in_bits = datalayout.get_type_align(storage, type_idx);

    if let Some(debug_info) = &ty.debug_info {
        let name = CString::new(debug_info.name.clone()).unwrap();
        unsafe {
            match &ty.ty {
                Type::Int(width) => {
                    let mut encoding = DW_ATE_unsigned;
                    if *width == 1 {
                        encoding = DW_ATE_boolean;
                    }
                    debuginfo::LLVMDIBuilderCreateBasicType(
                        builder,
                        name.as_ptr(),
                        name.count_bytes(),
                        size_in_bits as u64,
                        encoding.0 as u32,
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
                    let pointee_ptr =
                        lower_debug_type(datalayout, builder, storage, module_scope, *pointee);

                    if debug_info.is_reference {
                        debuginfo::LLVMDIBuilderCreateReferenceType(
                            builder,
                            DW_TAG_reference_type.0 as u32,
                            pointee_ptr,
                        )
                    } else {
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
                }
                Type::Vector(vector_type) => {
                    let inner_ty_ptr = lower_debug_type(
                        datalayout,
                        builder,
                        storage,
                        module_scope,
                        vector_type.ty,
                    );
                    let size = datalayout.get_type_size(storage, type_idx);
                    let align = datalayout.get_type_align(storage, type_idx);
                    let mut subrange = debuginfo::LLVMDIBuilderGetOrCreateSubrange(
                        builder,
                        0,
                        vector_type.size as i64,
                    );
                    debuginfo::LLVMDIBuilderCreateVectorType(
                        builder,
                        size as u64,
                        align,
                        inner_ty_ptr,
                        &raw mut subrange,
                        1,
                    )
                }
                Type::Array(array_type) => {
                    let inner_ty_ptr =
                        lower_debug_type(datalayout, builder, storage, module_scope, array_type.ty);
                    let size = datalayout.get_type_size(storage, type_idx);
                    let align = datalayout.get_type_align(storage, type_idx);
                    let mut subrange = debuginfo::LLVMDIBuilderGetOrCreateSubrange(
                        builder,
                        0,
                        array_type.size as i64,
                    );
                    debuginfo::LLVMDIBuilderCreateArrayType(
                        builder,
                        size as u64,
                        align,
                        inner_ty_ptr,
                        &raw mut subrange,
                        1,
                    )
                }
                Type::Struct(struct_type) => {
                    let mut fields = Vec::with_capacity(struct_type.fields.len());

                    let difile = get_difile_location(builder, &debug_info.location);
                    let line = debug_info.location.get_line();

                    let mut offset = 0;
                    let mut cur_align = 8;

                    for (i, field) in struct_type.fields.iter().enumerate() {
                        let field_align = datalayout.get_type_align(storage, *field);
                        cur_align = cur_align.max(field_align);

                        if offset % field_align != 0 {
                            let padding = (field_align - (offset % field_align)) % field_align;
                            offset += padding;
                        }

                        let field_size = datalayout.get_type_size(storage, *field);

                        let mut ty =
                            lower_debug_type(datalayout, builder, storage, module_scope, *field);

                        if let Some((field_name, location)) = struct_type.debug_field_names.get(i) {
                            let name = CString::new(field_name.clone()).unwrap();
                            let difile = get_difile_location(builder, location);
                            let line = location.get_line().unwrap_or(0);
                            let size = datalayout.get_type_size(storage, *field);
                            let align = datalayout.get_type_align(storage, *field);
                            ty = debuginfo::LLVMDIBuilderCreateMemberType(
                                builder,
                                module_scope,
                                name.as_ptr(),
                                name.count_bytes(),
                                difile,
                                line,
                                size as u64,
                                align,
                                offset as u64,
                                0,
                                ty,
                            );
                        }

                        offset += field_size;

                        fields.push(ty);
                    }

                    let size = datalayout.get_type_size(storage, type_idx);
                    let align = datalayout.get_type_align(storage, type_idx);

                    debuginfo::LLVMDIBuilderCreateStructType(
                        builder,
                        module_scope,
                        name.as_ptr(),
                        name.count_bytes(),
                        difile,
                        line.unwrap_or(0),
                        size as u64,
                        align,
                        0,
                        null_mut(),
                        fields.as_mut_ptr(),
                        fields.len() as u32,
                        0,
                        null_mut(),
                        name.as_ptr(),
                        name.count_bytes(),
                    )
                }
            }
        }
    } else {
        // No debug info provided - create anonymous debug types
        unsafe {
            match &ty.ty {
                Type::Int(width) => {
                    let name = CString::new(format!("i{}", width)).unwrap();
                    let encoding = if *width == 1 {
                        DW_ATE_boolean
                    } else {
                        DW_ATE_unsigned
                    };
                    debuginfo::LLVMDIBuilderCreateBasicType(
                        builder,
                        name.as_ptr(),
                        name.count_bytes(),
                        size_in_bits as u64,
                        encoding.0 as u32,
                        LLVMDIFlagPublic,
                    )
                }
                Type::Half
                | Type::BFloat
                | Type::Float
                | Type::Double
                | Type::Fp128
                | Type::X86Fp80
                | Type::PpcFp128 => {
                    let name = CString::new(format!("f{}", size_in_bits)).unwrap();
                    debuginfo::LLVMDIBuilderCreateBasicType(
                        builder,
                        name.as_ptr(),
                        name.count_bytes(),
                        size_in_bits as u64,
                        DW_ATE_float.0 as u32,
                        LLVMDIFlagPublic,
                    )
                }
                Type::Ptr {
                    pointee,
                    address_space,
                } => {
                    let pointee_ptr =
                        lower_debug_type(datalayout, builder, storage, module_scope, *pointee);
                    let name = CString::new("ptr").unwrap();
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
                Type::Vector(vector_type) => {
                    let inner_ty_ptr = lower_debug_type(
                        datalayout,
                        builder,
                        storage,
                        module_scope,
                        vector_type.ty,
                    );
                    let mut subrange = debuginfo::LLVMDIBuilderGetOrCreateSubrange(
                        builder,
                        0,
                        vector_type.size as i64,
                    );
                    debuginfo::LLVMDIBuilderCreateVectorType(
                        builder,
                        size_in_bits as u64,
                        align_in_bits,
                        inner_ty_ptr,
                        &raw mut subrange,
                        1,
                    )
                }
                Type::Array(array_type) => {
                    let inner_ty_ptr =
                        lower_debug_type(datalayout, builder, storage, module_scope, array_type.ty);
                    let mut subrange = debuginfo::LLVMDIBuilderGetOrCreateSubrange(
                        builder,
                        0,
                        array_type.size as i64,
                    );
                    debuginfo::LLVMDIBuilderCreateArrayType(
                        builder,
                        size_in_bits as u64,
                        align_in_bits,
                        inner_ty_ptr,
                        &raw mut subrange,
                        1,
                    )
                }
                Type::Struct(struct_type) => {
                    let mut fields = Vec::with_capacity(struct_type.fields.len());
                    let mut offset = 0;

                    for field in struct_type.fields.iter() {
                        let field_align = datalayout.get_type_align(storage, *field);
                        if offset % field_align != 0 {
                            offset += (field_align - (offset % field_align)) % field_align;
                        }
                        let ty =
                            lower_debug_type(datalayout, builder, storage, module_scope, *field);
                        let field_size = datalayout.get_type_size(storage, *field);
                        offset += field_size;
                        fields.push(ty);
                    }

                    let name = CString::new("struct").unwrap();
                    debuginfo::LLVMDIBuilderCreateStructType(
                        builder,
                        module_scope,
                        name.as_ptr(),
                        name.count_bytes(),
                        null_mut(),
                        0,
                        size_in_bits as u64,
                        align_in_bits,
                        0,
                        null_mut(),
                        fields.as_mut_ptr(),
                        fields.len() as u32,
                        0,
                        null_mut(),
                        name.as_ptr(),
                        name.count_bytes(),
                    )
                }
            }
        }
    }
}

/// Lower a constant value for global variable initialization (doesn't need FnCtx).
fn lower_global_constant(
    ctx: LLVMContextRef,
    storage: &TypeStorage,
    value: &ConstValue,
    ty: TypeIdx,
) -> LLVMValueRef {
    unsafe {
        let ty_ptr = lower_type(ctx, storage, ty);

        match value {
            ConstValue::Int(value) => core::LLVMConstInt(ty_ptr, *value, 0_i32),
            ConstValue::Float(value) => core::LLVMConstReal(ty_ptr, *value),
            ConstValue::Array(const_values) => {
                let array_ty = if let Type::Array(array_ty) = &storage.get_type_info(ty).ty {
                    array_ty
                } else {
                    panic!("type mismatch")
                };

                let typtr = lower_type(ctx, storage, array_ty.ty);
                let mut values: Vec<_> = const_values
                    .iter()
                    .map(|v| lower_global_constant(ctx, storage, v, array_ty.ty))
                    .collect();

                core::LLVMConstArray2(typtr, values.as_mut_ptr(), values.len() as u64)
            }
            ConstValue::Vector(const_values) => {
                let vec_ty = if let Type::Vector(vec_ty) = &storage.get_type_info(ty).ty {
                    vec_ty
                } else {
                    panic!("type mismatch")
                };

                let mut values: Vec<_> = const_values
                    .iter()
                    .map(|v| lower_global_constant(ctx, storage, v, vec_ty.ty))
                    .collect();

                core::LLVMConstVector(values.as_mut_ptr(), values.len() as u32)
            }
            ConstValue::Struct(const_values) => {
                let struct_ty = if let Type::Struct(struct_ty) = &storage.get_type_info(ty).ty {
                    &**struct_ty
                } else {
                    panic!("type mismatch")
                };
                let mut const_fields: Vec<_> = const_values
                    .iter()
                    .zip(struct_ty.fields.iter())
                    .map(|(v, field)| lower_global_constant(ctx, storage, v, *field))
                    .collect();
                core::LLVMConstStructInContext(
                    ctx,
                    const_fields.as_mut_ptr(),
                    const_fields.len() as u32,
                    struct_ty.packed as i32,
                )
            }
            ConstValue::NullPtr => core::LLVMConstPointerNull(ty_ptr),
            ConstValue::Undef => core::LLVMGetUndef(ty_ptr),
            ConstValue::Poison => core::LLVMGetPoison(ty_ptr),
        }
    }
}

fn lower_linkage(linkage: &Linkage) -> llvm_sys::LLVMLinkage {
    match linkage {
        Linkage::Private => llvm_sys::LLVMLinkage::LLVMPrivateLinkage,
        Linkage::Internal => llvm_sys::LLVMLinkage::LLVMInternalLinkage,
        Linkage::AvailableExternally => llvm_sys::LLVMLinkage::LLVMAvailableExternallyLinkage,
        Linkage::LinkOnce => llvm_sys::LLVMLinkage::LLVMLinkOnceAnyLinkage,
        Linkage::Weak => llvm_sys::LLVMLinkage::LLVMWeakAnyLinkage,
        Linkage::Common => llvm_sys::LLVMLinkage::LLVMCommonLinkage,
        Linkage::Appending => llvm_sys::LLVMLinkage::LLVMAppendingLinkage,
        Linkage::ExternWeak => llvm_sys::LLVMLinkage::LLVMExternalWeakLinkage,
        Linkage::LinkOnceOdr => llvm_sys::LLVMLinkage::LLVMLinkOnceODRLinkage,
        Linkage::WeakOdr => llvm_sys::LLVMLinkage::LLVMWeakODRLinkage,
        Linkage::External => llvm_sys::LLVMLinkage::LLVMExternalLinkage,
    }
}

fn lower_atomic_ordering(ordering: &AtomicOrdering) -> llvm_sys::LLVMAtomicOrdering {
    match ordering {
        AtomicOrdering::Unordered => llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingUnordered,
        AtomicOrdering::Monotonic => llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic,
        AtomicOrdering::Acquire => llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingAcquire,
        AtomicOrdering::Release => llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingRelease,
        AtomicOrdering::AcqRel => llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingAcquireRelease,
        AtomicOrdering::SeqCst => {
            llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent
        }
    }
}

fn lower_atomic_rmw_op(op: &AtomicRMWOp) -> llvm_sys::LLVMAtomicRMWBinOp {
    match op {
        AtomicRMWOp::Xchg => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXchg,
        AtomicRMWOp::Add => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
        AtomicRMWOp::Sub => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpSub,
        AtomicRMWOp::And => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAnd,
        AtomicRMWOp::Nand => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpNand,
        AtomicRMWOp::Or => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpOr,
        AtomicRMWOp::Xor => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXor,
        AtomicRMWOp::Max => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMax,
        AtomicRMWOp::Min => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMin,
        AtomicRMWOp::UMax => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpUMax,
        AtomicRMWOp::UMin => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpUMin,
        AtomicRMWOp::FAdd => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFAdd,
        AtomicRMWOp::FSub => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFSub,
        AtomicRMWOp::FMax => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFMax,
        AtomicRMWOp::FMin => llvm_sys::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFMin,
    }
}

fn apply_fast_math_flags(value: LLVMValueRef, flags: &FastMathFlags) {
    if !flags.any() {
        return;
    }

    let mut llvm_flags = llvm_sys::LLVMFastMathNone;
    if flags.reassoc {
        llvm_flags |= llvm_sys::LLVMFastMathAllowReassoc;
    }
    if flags.nnan {
        llvm_flags |= llvm_sys::LLVMFastMathNoNaNs;
    }
    if flags.ninf {
        llvm_flags |= llvm_sys::LLVMFastMathNoInfs;
    }
    if flags.nsz {
        llvm_flags |= llvm_sys::LLVMFastMathNoSignedZeros;
    }
    if flags.arcp {
        llvm_flags |= llvm_sys::LLVMFastMathAllowReciprocal;
    }
    if flags.contract {
        llvm_flags |= llvm_sys::LLVMFastMathAllowContract;
    }
    if flags.afn {
        llvm_flags |= llvm_sys::LLVMFastMathApproxFunc;
    }

    unsafe {
        core::LLVMSetFastMathFlags(value, llvm_flags);
    }
}

/// Apply function-level attributes to an LLVM function.
fn apply_function_attrs(
    ctx: LLVMContextRef,
    fn_ptr: LLVMValueRef,
    attrs: &irvm::function::FunctionAttrs,
) {
    // Function attribute index is -1 (LLVMAttributeFunctionIndex)
    const FUNCTION_INDEX: u32 = !0;

    unsafe {
        // Helper to add a string-named attribute
        let add_attr = |name: &[u8]| {
            let kind = core::LLVMGetEnumAttributeKindForName(name.as_ptr().cast(), name.len());
            if kind != 0 {
                let attr = core::LLVMCreateEnumAttribute(ctx, kind, 0);
                core::LLVMAddAttributeAtIndex(fn_ptr, FUNCTION_INDEX, attr);
            }
        };

        if attrs.nounwind {
            add_attr(b"nounwind");
        }
        if attrs.noreturn {
            add_attr(b"noreturn");
        }
        if attrs.cold {
            add_attr(b"cold");
        }
        if attrs.hot {
            add_attr(b"hot");
        }
        if attrs.willreturn {
            add_attr(b"willreturn");
        }
        if attrs.nosync {
            add_attr(b"nosync");
        }
        if attrs.nofree {
            add_attr(b"nofree");
        }
        if attrs.norecurse {
            add_attr(b"norecurse");
        }
        if attrs.readnone {
            add_attr(b"readnone");
        }
        if attrs.readonly {
            add_attr(b"readonly");
        }
        if attrs.writeonly {
            add_attr(b"writeonly");
        }
        if attrs.inlinehint {
            add_attr(b"inlinehint");
        }
        if attrs.alwaysinline {
            add_attr(b"alwaysinline");
        }
        if attrs.noinline {
            add_attr(b"noinline");
        }
        if attrs.minsize {
            add_attr(b"minsize");
        }
        if attrs.optsize {
            add_attr(b"optsize");
        }
    }
}

/// Apply parameter attributes to an LLVM function.
fn apply_parameter_attrs(
    ctx: LLVMContextRef,
    fn_ptr: LLVMValueRef,
    params: &[irvm::function::Parameter],
) {
    unsafe {
        for (idx, param) in params.iter().enumerate() {
            // Parameter indices start at 1 (0 is return value)
            let param_index = (idx + 1) as u32;

            // Helper to add a boolean attribute
            let add_attr = |name: &[u8]| {
                let kind = core::LLVMGetEnumAttributeKindForName(name.as_ptr().cast(), name.len());
                if kind != 0 {
                    let attr = core::LLVMCreateEnumAttribute(ctx, kind, 0);
                    core::LLVMAddAttributeAtIndex(fn_ptr, param_index, attr);
                }
            };

            if param.nocapture {
                add_attr(b"nocapture");
            }
            if param.readonly {
                add_attr(b"readonly");
            }
            if param.writeonly {
                add_attr(b"writeonly");
            }
            if param.noalias {
                add_attr(b"noalias");
            }
            if param.noundef {
                add_attr(b"noundef");
            }
            if param.nonnull {
                add_attr(b"nonnull");
            }
            if param.nofree {
                add_attr(b"nofree");
            }
            if param.nest {
                add_attr(b"nest");
            }
            if param.returned {
                add_attr(b"returned");
            }
            if param.inreg {
                add_attr(b"inreg");
            }
            if param.zeroext {
                add_attr(b"zeroext");
            }
            if param.signext {
                add_attr(b"signext");
            }

            // Int-valued attribute for dereferenceable
            if let Some(deref) = param.dereferenceable {
                let kind = core::LLVMGetEnumAttributeKindForName(
                    b"dereferenceable".as_ptr().cast(),
                    b"dereferenceable".len(),
                );
                if kind != 0 {
                    let attr = core::LLVMCreateEnumAttribute(ctx, kind, deref as u64);
                    core::LLVMAddAttributeAtIndex(fn_ptr, param_index, attr);
                }
            }

            // Int-valued attribute for alignment
            if let Some(align) = param.align {
                let kind =
                    core::LLVMGetEnumAttributeKindForName(b"align".as_ptr().cast(), b"align".len());
                if kind != 0 {
                    let attr = core::LLVMCreateEnumAttribute(ctx, kind, align as u64);
                    core::LLVMAddAttributeAtIndex(fn_ptr, param_index, attr);
                }
            }
        }
    }
}

/// Apply return value attributes to an LLVM function.
fn apply_return_attrs(
    ctx: LLVMContextRef,
    fn_ptr: LLVMValueRef,
    attrs: &irvm::function::ReturnAttrs,
) {
    // Return attribute index is 0
    const RETURN_INDEX: u32 = 0;

    unsafe {
        let add_attr = |name: &[u8]| {
            let kind = core::LLVMGetEnumAttributeKindForName(name.as_ptr().cast(), name.len());
            if kind != 0 {
                let attr = core::LLVMCreateEnumAttribute(ctx, kind, 0);
                core::LLVMAddAttributeAtIndex(fn_ptr, RETURN_INDEX, attr);
            }
        };

        if attrs.noalias {
            add_attr(b"noalias");
        }
        if attrs.noundef {
            add_attr(b"noundef");
        }
        if attrs.nonnull {
            add_attr(b"nonnull");
        }

        // Int-valued attribute for dereferenceable
        if let Some(deref) = attrs.dereferenceable {
            let kind = core::LLVMGetEnumAttributeKindForName(
                b"dereferenceable".as_ptr().cast(),
                b"dereferenceable".len(),
            );
            if kind != 0 {
                let attr = core::LLVMCreateEnumAttribute(ctx, kind, deref as u64);
                core::LLVMAddAttributeAtIndex(fn_ptr, RETURN_INDEX, attr);
            }
        }
    }
}

/// Lower an intrinsic call to LLVM IR.
unsafe fn lower_intrinsic(
    ctx: &FnCtx,
    intrinsic: &irvm::block::Intrinsic,
    _block_idx: BlockIdx,
    _inst_idx: irvm::block::InstIdx,
) -> Result<LLVMValueRef, Error> {
    use irvm::block::Intrinsic;

    unsafe {
        let null_name = c"";
        let module = core::LLVMGetGlobalParent(ctx.fn_ptr);

        match intrinsic {
            // Memory intrinsics
            Intrinsic::Memcpy {
                dest,
                src,
                len,
                is_volatile: _,
            } => {
                let dest_val = lower_operand(ctx, dest);
                let src_val = lower_operand(ctx, src);
                let len_val = lower_operand(ctx, len);
                core::LLVMBuildMemCpy(
                    ctx.builder,
                    dest_val,
                    1, // dest alignment
                    src_val,
                    1, // src alignment
                    len_val,
                );
                Ok(null_mut())
            }
            Intrinsic::Memset {
                dest,
                val,
                len,
                is_volatile: _,
            } => {
                let dest_val = lower_operand(ctx, dest);
                let val_val = lower_operand(ctx, val);
                let len_val = lower_operand(ctx, len);
                core::LLVMBuildMemSet(
                    ctx.builder,
                    dest_val,
                    val_val,
                    len_val,
                    1, // alignment
                );
                Ok(null_mut())
            }
            Intrinsic::Memmove {
                dest,
                src,
                len,
                is_volatile: _,
            } => {
                let dest_val = lower_operand(ctx, dest);
                let src_val = lower_operand(ctx, src);
                let len_val = lower_operand(ctx, len);
                core::LLVMBuildMemMove(
                    ctx.builder,
                    dest_val,
                    1, // dest alignment
                    src_val,
                    1, // src alignment
                    len_val,
                );
                Ok(null_mut())
            }

            // Overflow intrinsics
            Intrinsic::SaddWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.sadd.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::UaddWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.uadd.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::SsubWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.ssub.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::UsubWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.usub.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::SmulWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.smul.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::UmulWithOverflow {
                lhs,
                rhs,
                result_ty: _,
            } => {
                let lhs_val = lower_operand(ctx, lhs);
                let rhs_val = lower_operand(ctx, rhs);
                let lhs_ty = lower_type(ctx.ctx, ctx.storage, lhs.get_type());
                let intrinsic_name = CString::new(format!(
                    "llvm.umul.with.overflow.i{}",
                    core::LLVMGetIntTypeWidth(lhs_ty)
                ))
                .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [lhs_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [lhs_val, rhs_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    null_name.as_ptr(),
                );
                Ok(value)
            }

            // Math intrinsics - unary
            Intrinsic::Sqrt { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.sqrt", value)
            }
            Intrinsic::Sin { value } => lower_unary_float_intrinsic(ctx, module, "llvm.sin", value),
            Intrinsic::Cos { value } => lower_unary_float_intrinsic(ctx, module, "llvm.cos", value),
            Intrinsic::Exp { value } => lower_unary_float_intrinsic(ctx, module, "llvm.exp", value),
            Intrinsic::Exp2 { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.exp2", value)
            }
            Intrinsic::Log { value } => lower_unary_float_intrinsic(ctx, module, "llvm.log", value),
            Intrinsic::Log2 { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.log2", value)
            }
            Intrinsic::Log10 { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.log10", value)
            }
            Intrinsic::Fabs { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.fabs", value)
            }
            Intrinsic::Floor { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.floor", value)
            }
            Intrinsic::Ceil { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.ceil", value)
            }
            Intrinsic::Trunc { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.trunc", value)
            }
            Intrinsic::Round { value } => {
                lower_unary_float_intrinsic(ctx, module, "llvm.round", value)
            }

            // Math intrinsics - binary
            Intrinsic::Pow { base, exp } => {
                lower_binary_float_intrinsic(ctx, module, "llvm.pow", base, exp)
            }
            Intrinsic::Powi { base, exp } => {
                // powi takes float and i32 exponent
                let base_val = lower_operand(ctx, base);
                let exp_val = lower_operand(ctx, exp);
                let base_ty = lower_type(ctx.ctx, ctx.storage, base.get_type());
                let ty_name = get_float_type_suffix(base_ty);
                let intrinsic_name = CString::new(format!("llvm.powi.{}.i32", ty_name)).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [base_ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [base_val, exp_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(value)
            }
            Intrinsic::Copysign { mag, sign } => {
                lower_binary_float_intrinsic(ctx, module, "llvm.copysign", mag, sign)
            }
            Intrinsic::Minnum { a, b } => {
                lower_binary_float_intrinsic(ctx, module, "llvm.minnum", a, b)
            }
            Intrinsic::Maxnum { a, b } => {
                lower_binary_float_intrinsic(ctx, module, "llvm.maxnum", a, b)
            }

            // Math intrinsics - ternary
            Intrinsic::Fma { a, b, c } => {
                let a_val = lower_operand(ctx, a);
                let b_val = lower_operand(ctx, b);
                let c_val = lower_operand(ctx, c);
                let ty = lower_type(ctx.ctx, ctx.storage, a.get_type());
                let ty_name = get_float_type_suffix(ty);
                let intrinsic_name = CString::new(format!("llvm.fma.{}", ty_name)).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [a_val, b_val, c_val];
                let value = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(value)
            }

            // Bit manipulation intrinsics
            Intrinsic::Ctpop { value } => {
                lower_unary_int_intrinsic(ctx, module, "llvm.ctpop", value)
            }
            Intrinsic::Ctlz {
                value,
                is_zero_poison,
            } => {
                let val = lower_operand(ctx, value);
                let ty = lower_type(ctx.ctx, ctx.storage, value.get_type());
                let intrinsic_name =
                    CString::new(format!("llvm.ctlz.i{}", core::LLVMGetIntTypeWidth(ty))).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let poison_val = core::LLVMConstInt(
                    core::LLVMInt1TypeInContext(ctx.ctx),
                    *is_zero_poison as u64,
                    0,
                );
                let mut args = [val, poison_val];
                let result = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(result)
            }
            Intrinsic::Cttz {
                value,
                is_zero_poison,
            } => {
                let val = lower_operand(ctx, value);
                let ty = lower_type(ctx.ctx, ctx.storage, value.get_type());
                let intrinsic_name =
                    CString::new(format!("llvm.cttz.i{}", core::LLVMGetIntTypeWidth(ty))).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let poison_val = core::LLVMConstInt(
                    core::LLVMInt1TypeInContext(ctx.ctx),
                    *is_zero_poison as u64,
                    0,
                );
                let mut args = [val, poison_val];
                let result = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(result)
            }
            Intrinsic::Bitreverse { value } => {
                lower_unary_int_intrinsic(ctx, module, "llvm.bitreverse", value)
            }
            Intrinsic::Bswap { value } => {
                lower_unary_int_intrinsic(ctx, module, "llvm.bswap", value)
            }
            Intrinsic::Fshl { a, b, shift } => {
                let a_val = lower_operand(ctx, a);
                let b_val = lower_operand(ctx, b);
                let shift_val = lower_operand(ctx, shift);
                let ty = lower_type(ctx.ctx, ctx.storage, a.get_type());
                let intrinsic_name =
                    CString::new(format!("llvm.fshl.i{}", core::LLVMGetIntTypeWidth(ty))).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [a_val, b_val, shift_val];
                let result = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(result)
            }
            Intrinsic::Fshr { a, b, shift } => {
                let a_val = lower_operand(ctx, a);
                let b_val = lower_operand(ctx, b);
                let shift_val = lower_operand(ctx, shift);
                let ty = lower_type(ctx.ctx, ctx.storage, a.get_type());
                let intrinsic_name =
                    CString::new(format!("llvm.fshr.i{}", core::LLVMGetIntTypeWidth(ty))).unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [a_val, b_val, shift_val];
                let result = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(result)
            }

            // Other intrinsics
            Intrinsic::Expect { value, expected } => {
                let val = lower_operand(ctx, value);
                let exp = lower_operand(ctx, expected);
                let ty = lower_type(ctx.ctx, ctx.storage, value.get_type());
                let intrinsic_name =
                    CString::new(format!("llvm.expect.i{}", core::LLVMGetIntTypeWidth(ty)))
                        .unwrap();
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.to_bytes().len(),
                );
                let mut param_types = [ty];
                let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
                    module,
                    intrinsic_id,
                    param_types.as_mut_ptr(),
                    param_types.len(),
                );
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [val, exp];
                let result = core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(result)
            }
            Intrinsic::Assume { cond } => {
                let cond_val = lower_operand(ctx, cond);
                let intrinsic_name = c"llvm.assume";
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.count_bytes(),
                );
                let intrinsic_fn =
                    core::LLVMGetIntrinsicDeclaration(module, intrinsic_id, null_mut(), 0);
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                let mut args = [cond_val];
                core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    args.as_mut_ptr(),
                    args.len() as u32,
                    c"".as_ptr(),
                );
                Ok(null_mut())
            }
            Intrinsic::Trap => {
                let intrinsic_name = c"llvm.trap";
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.count_bytes(),
                );
                let intrinsic_fn =
                    core::LLVMGetIntrinsicDeclaration(module, intrinsic_id, null_mut(), 0);
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    null_mut(),
                    0,
                    c"".as_ptr(),
                );
                Ok(null_mut())
            }
            Intrinsic::Debugtrap => {
                let intrinsic_name = c"llvm.debugtrap";
                let intrinsic_id = core::LLVMLookupIntrinsicID(
                    intrinsic_name.as_ptr(),
                    intrinsic_name.count_bytes(),
                );
                let intrinsic_fn =
                    core::LLVMGetIntrinsicDeclaration(module, intrinsic_id, null_mut(), 0);
                let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
                core::LLVMBuildCall2(
                    ctx.builder,
                    fn_ty,
                    intrinsic_fn,
                    null_mut(),
                    0,
                    c"".as_ptr(),
                );
                Ok(null_mut())
            }
        }
    } // end unsafe block
}

/// Helper to lower unary float intrinsics.
unsafe fn lower_unary_float_intrinsic(
    ctx: &FnCtx,
    module: *mut LLVMModule,
    name: &str,
    value: &Operand,
) -> Result<LLVMValueRef, Error> {
    unsafe {
        let val = lower_operand(ctx, value);
        let ty = lower_type(ctx.ctx, ctx.storage, value.get_type());
        let ty_name = get_float_type_suffix(ty);
        let intrinsic_name = CString::new(format!("{}.{}", name, ty_name)).unwrap();
        let intrinsic_id =
            core::LLVMLookupIntrinsicID(intrinsic_name.as_ptr(), intrinsic_name.to_bytes().len());
        let mut param_types = [ty];
        let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
            module,
            intrinsic_id,
            param_types.as_mut_ptr(),
            param_types.len(),
        );
        let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
        let mut args = [val];
        let result = core::LLVMBuildCall2(
            ctx.builder,
            fn_ty,
            intrinsic_fn,
            args.as_mut_ptr(),
            args.len() as u32,
            c"".as_ptr(),
        );
        Ok(result)
    } // end unsafe block
}

/// Helper to lower binary float intrinsics.
unsafe fn lower_binary_float_intrinsic(
    ctx: &FnCtx,
    module: *mut LLVMModule,
    name: &str,
    a: &Operand,
    b: &Operand,
) -> Result<LLVMValueRef, Error> {
    unsafe {
        let a_val = lower_operand(ctx, a);
        let b_val = lower_operand(ctx, b);
        let ty = lower_type(ctx.ctx, ctx.storage, a.get_type());
        let ty_name = get_float_type_suffix(ty);
        let intrinsic_name = CString::new(format!("{}.{}", name, ty_name)).unwrap();
        let intrinsic_id =
            core::LLVMLookupIntrinsicID(intrinsic_name.as_ptr(), intrinsic_name.to_bytes().len());
        let mut param_types = [ty];
        let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
            module,
            intrinsic_id,
            param_types.as_mut_ptr(),
            param_types.len(),
        );
        let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
        let mut args = [a_val, b_val];
        let result = core::LLVMBuildCall2(
            ctx.builder,
            fn_ty,
            intrinsic_fn,
            args.as_mut_ptr(),
            args.len() as u32,
            c"".as_ptr(),
        );
        Ok(result)
    } // end unsafe block
}

/// Helper to lower unary integer intrinsics.
unsafe fn lower_unary_int_intrinsic(
    ctx: &FnCtx,
    module: *mut LLVMModule,
    name: &str,
    value: &Operand,
) -> Result<LLVMValueRef, Error> {
    unsafe {
        let val = lower_operand(ctx, value);
        let ty = lower_type(ctx.ctx, ctx.storage, value.get_type());
        let intrinsic_name =
            CString::new(format!("{}.i{}", name, core::LLVMGetIntTypeWidth(ty))).unwrap();
        let intrinsic_id =
            core::LLVMLookupIntrinsicID(intrinsic_name.as_ptr(), intrinsic_name.to_bytes().len());
        let mut param_types = [ty];
        let intrinsic_fn = core::LLVMGetIntrinsicDeclaration(
            module,
            intrinsic_id,
            param_types.as_mut_ptr(),
            param_types.len(),
        );
        let fn_ty = core::LLVMGlobalGetValueType(intrinsic_fn);
        let mut args = [val];
        let result = core::LLVMBuildCall2(
            ctx.builder,
            fn_ty,
            intrinsic_fn,
            args.as_mut_ptr(),
            args.len() as u32,
            c"".as_ptr(),
        );
        Ok(result)
    } // end unsafe block
}

/// Get the LLVM type suffix for float types.
unsafe fn get_float_type_suffix(ty: LLVMTypeRef) -> &'static str {
    unsafe {
        match core::LLVMGetTypeKind(ty) {
            llvm_sys::LLVMTypeKind::LLVMHalfTypeKind => "f16",
            llvm_sys::LLVMTypeKind::LLVMBFloatTypeKind => "bf16",
            llvm_sys::LLVMTypeKind::LLVMFloatTypeKind => "f32",
            llvm_sys::LLVMTypeKind::LLVMDoubleTypeKind => "f64",
            llvm_sys::LLVMTypeKind::LLVMFP128TypeKind => "f128",
            _ => "f64", // default to f64
        }
    } // end unsafe block
}
