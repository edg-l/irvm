# IRVM - Claude Code Context

## Project Overview

IRVM is a Rust library providing an intermediate representation (IR) that lowers to LLVM IR. It offers a type-safe, Rust-friendly API for generating LLVM IR without direct FFI usage.

**Target LLVM version: 20** (llvm-sys 201.0.1)

## Repository Structure

```
irvm/
├── src/                    # Core IR library
│   ├── lib.rs             # Public API exports
│   ├── module.rs          # Module, GlobalVariable, GlobalIdx
│   ├── function.rs        # Function, Parameter, FnIdx
│   ├── block.rs           # Block, Instructions, Terminators (~900 lines)
│   ├── types.rs           # Type system, TypeStorage, TypeIdx
│   ├── value.rs           # Operand, ConstValue
│   ├── common.rs          # Linkage, Visibility, CConv, etc.
│   ├── datalayout.rs      # DataLayout for type sizes/alignment
│   └── error.rs           # Error types
├── irvm-lower/            # LLVM lowering crate
│   └── src/
│       ├── lib.rs         # Public exports
│       └── llvm.rs        # Lowering implementation (~2500 lines)
├── examples/simple/       # Example usage
├── ROADMAP.md            # Feature roadmap and coverage estimates
└── Cargo.toml            # Workspace root
```

## Build & Test

```bash
cargo build --workspace     # Build all crates
cargo test --workspace      # Run all tests
cargo clippy --workspace    # Lint
```

## Key Design Patterns

### Generational Arenas
All indices use `typed_generational_arena::StandardSlabIndex`:
- `TypeIdx` - index into `TypeStorage`
- `BlockIdx` - index into function's blocks
- `InstIdx` - index into block's instructions
- `FnIdx` - index into module's functions
- `GlobalIdx` - index into module's globals

### Instruction Builder Pattern
Instructions are added via methods on `Block`:
```rust
block.instr_add(lhs, rhs, location)        // Returns Operand
block.instr_store(value, ptr, location)    // Returns ()
block.set_terminator(Terminator::Ret(...)) // Sets block terminator
```

### Two-Pass Lowering
In `irvm-lower/src/llvm.rs`, lowering happens in two passes:
1. **Declaration pass**: Create LLVM function/global declarations
2. **Body pass**: Lower instructions with all references available

### Operand System
Values are represented as `Operand` enum:
- `Parameter(nth, type)` - function parameter
- `BlockArgument { block_idx, nth, ty }` - phi-like block argument
- `Value(block, inst, type)` - result of an instruction
- `Constant(ConstValue, type)` - compile-time constant
- `Global(GlobalIdx, type)` - reference to global variable

## Instruction Categories

In `src/block.rs`, instructions are organized by enum:
- `BinaryOp` - add, sub, mul, div, rem (int)
- `BitwiseBinaryOp` - and, or, xor, shl, lshr, ashr
- `ConversionOp` - trunc, zext, sext, fptrunc, fpext, casts
- `VectorOp` - extractelement, insertelement, shufflevector
- `AggregateOp` - extractvalue, insertvalue
- `MemoryOp` - load, store, alloca, GEP, atomics, fence
- `OtherOp` - icmp, fcmp, call, phi, select, landingpad
- `DebugOp` - debug value/declare

Terminators in `Terminator` enum:
- `Ret`, `Br`, `CondBr`, `Switch`, `Invoke`, `Resume`, `Unreachable`

## Common Tasks

### Adding a New Instruction

1. Add variant to appropriate enum in `src/block.rs`
2. Add `instr_*` method to `Block` impl
3. Add lowering in `irvm-lower/src/llvm.rs` in the instruction match

### Adding a New Type

1. Add variant to `Type` enum in `src/types.rs`
2. Add `get_or_create_*` method to `TypeStorage`
3. Add lowering in `lower_type()` function in `llvm.rs`

### Debugging Lowered IR

The test functions use `LLVMDumpModule` or `LLVMPrintModuleToString` to print generated IR. Check `irvm-lower/src/lib.rs` tests for examples.

## API Patterns & Gotchas

### Type Storage Helpers
```rust
// Before using icmp (which returns i1), ensure i1 type exists:
let _i1 = storage.get_or_create_i1();

// Similarly for i64:
let _i64 = storage.get_or_create_i64();
```

### Debug Info
Types don't require debug names - anonymous debug types are auto-generated:
```rust
// Both work fine:
storage.add_type(Type::Int(32), Some("i32"));  // Named
storage.add_type(Type::Int(32), None);          // Anonymous
```

### JIT Testing Limitations
The MCJIT execution engine only supports certain parameter/return types reliably:
- Use `i32` parameters and returns for JIT tests
- Avoid `i8`, `i16`, structs as function parameters in JIT tests
- For testing small types, use `i32` params and trunc/ext internally

## LLVM 20 Specifics

- Use `LLVMConstArray2` (not deprecated `LLVMConstArray`)
- Use typed APIs: `LLVMBuildLoad2`, `LLVMBuildGEP2`, `LLVMBuildCall2`, `LLVMBuildInvoke2`
- `LLVMSetFastMathFlags` is available for applying fast-math flags
- Opaque pointers are the default (no typed pointers)

## Current Coverage

~65-70% of LLVM IR (v0.2.0). See ROADMAP.md for details.

**Implemented:**
- All arithmetic/bitwise operations
- Type casts (trunc, zext, sext, fp casts, ptr↔int, bitcast)
- Memory operations (load, store, alloca, GEP)
- Atomic operations (atomicrmw, cmpxchg, fence)
- Control flow (br, condbr, switch, ret)
- Exception handling (invoke, landingpad, resume, unreachable)
- Vector operations (extract/insert element, shufflevector)
- Aggregate operations (extractvalue, insertvalue)
- Global variables with full attributes
- Fast-math flags
- Debug info (DISubprogram, DILocation, DILocalVariable, etc.)

**Not yet implemented** (see ROADMAP.md):
- Intrinsics system
- Function/parameter attributes
- GEP type inference
- Module metadata
- Inline assembly

## Code Style

- Edition 2024
- No unsafe in `irvm` crate (all unsafe in `irvm-lower`)
- Prefer explicit types over inference for public APIs
- Use `Location` for source mapping on all constructs
- Clone-friendly types (most types derive Clone)
- Use `cargo fmt` after changes
