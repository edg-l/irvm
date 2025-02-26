use std::{error::Error, ffi::{CStr, CString}};

use irvm::module::Module;

use llvm_sys::core;


pub fn lower_module(module: &Module) -> Result<(), Box<dyn Error>> {

    unsafe {
        let ctx = core::LLVMContextCreate();
        let module_name = CString::new(module.name.clone())?;
        let module = core::LLVMModuleCreateWithNameInContext(module_name.as_ptr(), ctx);


        core::LLVMDisposeModule(module);
        core::LLVMContextDispose(ctx);
    }

    Ok(())

}
