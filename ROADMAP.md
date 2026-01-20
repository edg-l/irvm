# IRVM Roadmap

This document outlines the planned features and improvements for IRVM.

## Recently Completed (v0.2.0)

### Critical Features
- **Type Casting/Conversion Instructions**: `trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `fptoui`, `fptosi`, `uitofp`, `sitofp`, `ptrtoint`, `inttoptr`, `bitcast`, `addrspacecast`
- **Global Variables & Constants**: Module-level global variable support with initializers, linkage, visibility, and alignment
- **`switch` Terminator**: Multi-way branching based on integer comparison
- **`select` Instruction**: Conditional value selection (ternary operator)

### High Priority Features
- **Fast Math Flags**: `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc` for floating-point operations
- **Vector Operations**: `insertelement`, `shufflevector`
- **Aggregate Operations**: `extractvalue`, `insertvalue` for structs and arrays
- **Atomic Operations**: `atomicrmw`, `cmpxchg`, `fence`, atomic load/store
- **Exception Handling**: `invoke`, `landingpad`, `resume`, `unreachable` terminators
- **FNeg Instruction**: Floating-point negation

---

## Medium Priority (v0.3.0)

### Intrinsics System
- [x] Design intrinsics declaration mechanism
- [x] Implement `llvm.memcpy`, `llvm.memset`, `llvm.memmove`
- [x] Implement overflow intrinsics (`llvm.sadd.with.overflow`, etc.)
- [x] Implement math intrinsics (`llvm.sqrt`, `llvm.sin`, `llvm.cos`, etc.)
- [x] Implement bit manipulation intrinsics (`llvm.ctpop`, `llvm.ctlz`, `llvm.cttz`)

### Function & Call Attributes
- [x] Function attributes: `nounwind`, `noreturn`, `cold`, `hot`, `willreturn`, `nosync`, `nofree`
- [x] Parameter attributes: `nocapture`, `readonly`, `writeonly`, `noalias`, `noundef`, `nonnull`, `nofree`, `nest`, `returned`, `inreg`, `zeroext`, `signext`, `dereferenceable(N)`, `align(N)`
- [x] Return attributes: `noalias`, `noundef`, `nonnull`, `dereferenceable(N)`
- [x] GC name support
- [x] Prefix/prologue data

### GEP Type Inference
- [x] Implement automatic result type computation for GEP
- [x] Add `compute_gep_result_type` helper function
- [x] Update `instr_gep` to infer result type automatically
- [x] Keep `instr_gep_ex` for explicit type specification

### Memory Operation Enhancements
- [x] Add `volatile` flag to load/store
- [x] Alignment validation error type (enforcement pending)
- [ ] Non-temporal hints

---

## Lower Priority (v0.4.0+)

### Module-Level Features
- [ ] Function declarations (external functions without bodies)
- [ ] Module flags metadata
- [ ] Named metadata
- [ ] Comdat groups
- [ ] Module-level inline assembly
- [ ] Section name support for functions/globals

### Enhanced Debug Information
- [ ] Fix `DebugOp::Assign` lowering (currently `todo!()`)
- [ ] DIFile, DILexicalBlock support
- [ ] DICompositeType for classes
- [ ] Variable lifetime tracking
- [ ] Scope/location tracking improvements

### Type System Improvements
- [ ] Explicit `Void` type (currently uses `Option<TypeIdx>`)
- [ ] `Metadata` type
- [ ] `Token` type (for exception handling)
- [ ] Opaque struct types
- [ ] Improved scalable vector support

### Additional Instructions
- [ ] `freeze` instruction (freeze poison/undef values)
- [ ] Vector broadcast operations
- [ ] More vector shuffle patterns

### Error Handling Improvements
- [ ] More detailed type checking errors with source locations
- [ ] Validation errors with context
- [ ] Better error messages for debugging

### Inline Assembly
- [ ] Basic inline assembly support
- [ ] Constraint parsing
- [ ] Clobber handling

---

## Long-Term Goals

### Performance
- [ ] Lazy type lowering
- [ ] Type caching optimizations
- [ ] Parallel lowering for large modules

### Tooling
- [ ] IR pretty printer
- [ ] IR parser (from text format)
- [ ] IR validation pass
- [ ] IR optimization passes (constant folding, dead code elimination)

### Documentation
- [ ] Comprehensive API documentation
- [ ] Tutorial for building a simple language
- [ ] Examples for common patterns

---

## Estimated LLVM IR Coverage

| Version | Coverage | Notes |
|---------|----------|-------|
| v0.1.x  | ~40-45%  | Basic arithmetic, memory, control flow |
| v0.2.0  | ~65-70%  | +casts, globals, atomics, exceptions, vectors |
| v0.3.0  | ~80%     | +intrinsics, attributes, GEP inference |
| v0.4.0+ | ~90%+    | +module features, debug info, inline asm |

---

## Contributing

Contributions are welcome! When working on a feature:

1. Check if there's an existing issue or discussion
2. Follow the existing code patterns (generational arenas, two-pass lowering, etc.)
3. Add tests for new functionality
4. Update this roadmap when features are completed

## Design Principles

- **Safety First**: Unsafe code should be hidden behind safe abstractions
- **Type Safety**: Strong typing to prevent operand misuse
- **Separation of Concerns**: IR definition separate from LLVM lowering
- **LLVM Fidelity**: Model LLVM IR concepts faithfully
- **Ergonomic API**: Rust-friendly API that's pleasant to use
