# IRVM

A IR compiler target with a native Rust friendly API that lowers to LLVM IR.

## How it works

Basically mimic a IR that closely resembles LLVM IR in Rust structures and only interface with LLVM at the time of lowering to LLVM IR / compilation.

Ideally when lowering to LLVM IR the IR in IRVM should be valid due to checks on our side.

## Why?

There are some nice crates to use LLVM from Rust, like [inkwell](https://github.com/TheDan64/inkwell), but due to the need to model the C++ ownership (ffi) in Rust, the API tends to not be so user friendly, even if they try hard, also some functions like [GEP](https://thedan64.github.io/inkwell/inkwell/builder/struct.Builder.html#method.build_gep) are `unsafe` if used incorrectly, this library strives to provide a Rust friendly API thats fully safe.
