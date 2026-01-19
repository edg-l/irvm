use typed_generational_arena::{StandardSlab, StandardSlabIndex};

use crate::{
    common::{CConv, Location},
    error::Error,
    function::{DebugVarIdx, FnIdx},
    types::{FunctionType, Type, TypeIdx, TypeStorage},
    value::Operand,
};

/// A Block id.
pub type BlockIdx = StandardSlabIndex<Block>;
/// An instruction id.
pub type InstIdx = StandardSlabIndex<(Location, Instruction)>;

/// A Block that holds instructions executed in a sequence and a terminator for control flow.
///
/// Terminator default to Ret.
#[derive(Debug, Clone)]
pub struct Block {
    // The id is always set, but this is needed because first we need to
    // insert the block into the arena to get an id.
    pub(crate) id: Option<BlockIdx>,
    /// Arguments are made to model phi nodes.
    arguments: Vec<TypeIdx>,
    /// The instructions within this block.
    instructions: StandardSlab<(Location, Instruction)>,
    /// the terminator of this block
    terminator: Terminator,
    last_instr_idx: Option<InstIdx>,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    BinaryOp(BinaryOp),
    BitwiseBinaryOp(BitwiseBinaryOp),
    VectorOp(VectorOp),
    MemoryOp(MemoryOp),
    OtherOp(OtherOp),
    DebugOp(DebugOp),
    ConversionOp(ConversionOp),
    AggregateOp(AggregateOp),
}

/// Type conversion/casting operations.
#[derive(Debug, Clone)]
pub enum ConversionOp {
    /// Truncate integer to smaller type.
    Trunc { value: Operand, target_ty: TypeIdx },
    /// Zero extend integer to larger type.
    ZExt { value: Operand, target_ty: TypeIdx },
    /// Sign extend integer to larger type.
    SExt { value: Operand, target_ty: TypeIdx },
    /// Truncate floating-point to smaller type.
    FPTrunc { value: Operand, target_ty: TypeIdx },
    /// Extend floating-point to larger type.
    FPExt { value: Operand, target_ty: TypeIdx },
    /// Convert floating-point to unsigned integer.
    FPToUI { value: Operand, target_ty: TypeIdx },
    /// Convert floating-point to signed integer.
    FPToSI { value: Operand, target_ty: TypeIdx },
    /// Convert unsigned integer to floating-point.
    UIToFP { value: Operand, target_ty: TypeIdx },
    /// Convert signed integer to floating-point.
    SIToFP { value: Operand, target_ty: TypeIdx },
    /// Convert pointer to integer.
    PtrToInt { value: Operand, target_ty: TypeIdx },
    /// Convert integer to pointer.
    IntToPtr { value: Operand, target_ty: TypeIdx },
    /// Bitwise reinterpretation (same bit width required).
    Bitcast { value: Operand, target_ty: TypeIdx },
    /// Cast pointer to different address space.
    AddrSpaceCast { value: Operand, target_ty: TypeIdx },
}

/// Aggregate (struct/array) operations.
#[derive(Debug, Clone)]
pub enum AggregateOp {
    /// Extract a value from an aggregate (struct or array) at the given indices.
    ExtractValue {
        aggregate: Operand,
        indices: Vec<u32>,
    },
    /// Insert a value into an aggregate (struct or array) at the given indices.
    InsertValue {
        aggregate: Operand,
        element: Operand,
        indices: Vec<u32>,
    },
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Ret((Location, Option<Operand>)),
    Br {
        block: BlockIdx,
        arguments: Vec<Operand>,
        location: Location,
    },
    CondBr {
        then_block: BlockIdx,
        else_block: BlockIdx,
        cond: Operand,
        if_args: Vec<Operand>,
        then_args: Vec<Operand>,
        location: Location,
    },
    /// Multi-way branch based on integer value.
    Switch {
        value: Operand,
        default_block: BlockIdx,
        default_args: Vec<Operand>,
        cases: Vec<SwitchCase>,
        location: Location,
    },
    /// Call that may throw an exception.
    Invoke {
        call: CallOp,
        normal_dest: BlockIdx,
        normal_args: Vec<Operand>,
        unwind_dest: BlockIdx,
        unwind_args: Vec<Operand>,
        location: Location,
    },
    /// Resume propagation of an exception.
    Resume {
        value: Operand,
        location: Location,
    },
    /// Mark code as unreachable.
    Unreachable {
        location: Location,
    },
}

/// A case in a switch statement.
#[derive(Debug, Clone)]
pub struct SwitchCase {
    /// The constant integer value to match.
    pub value: u64,
    /// The target block.
    pub block: BlockIdx,
    /// Arguments to pass to the target block.
    pub arguments: Vec<Operand>,
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Sub {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Mul {
        lhs: Operand,
        rhs: Operand,
        nsw: bool,
        nuw: bool,
    },
    Div {
        lhs: Operand,
        rhs: Operand,
        signed: bool,
        /// If the exact keyword is present,
        /// the result value of the udiv is a poison value if %op1
        /// is not a multiple of %op2 (as such, “((a udiv exact b) mul b) == a”).
        exact: bool,
    },
    Rem {
        lhs: Operand,
        rhs: Operand,
        signed: bool,
    },
    FAdd {
        lhs: Operand,
        rhs: Operand,
        flags: FastMathFlags,
    },
    FSub {
        lhs: Operand,
        rhs: Operand,
        flags: FastMathFlags,
    },
    FMul {
        lhs: Operand,
        rhs: Operand,
        flags: FastMathFlags,
    },
    FDiv {
        lhs: Operand,
        rhs: Operand,
        flags: FastMathFlags,
    },
    FRem {
        lhs: Operand,
        rhs: Operand,
        flags: FastMathFlags,
    },
    /// Floating-point negation.
    FNeg {
        value: Operand,
        flags: FastMathFlags,
    },
}

#[derive(Debug, Clone)]
pub enum BitwiseBinaryOp {
    Shl {
        lhs: Operand,
        rhs: Operand,
    },
    Lshr {
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    },
    Ashr {
        lhs: Operand,
        rhs: Operand,
        exact: bool,
    },
    And {
        lhs: Operand,
        rhs: Operand,
    },
    Or {
        lhs: Operand,
        rhs: Operand,
        disjoint: bool,
    },
    Xor {
        lhs: Operand,
        rhs: Operand,
    },
}

#[derive(Debug, Clone)]
pub enum VectorOp {
    /// Extract an element from a vector at the given index.
    ExtractElement { vector: Operand, idx: Operand },
    /// Insert an element into a vector at the given index.
    InsertElement {
        vector: Operand,
        element: Operand,
        idx: Operand,
    },
    /// Shuffle elements from two vectors using a mask.
    /// Mask values of -1 indicate undef/poison.
    ShuffleVector {
        vec1: Operand,
        vec2: Operand,
        mask: Vec<i32>,
    },
}

/// Fast math flags for floating-point operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FastMathFlags {
    /// Assume no NaNs.
    pub nnan: bool,
    /// Assume no Infs.
    pub ninf: bool,
    /// Allow treating sign of zero as insignificant.
    pub nsz: bool,
    /// Allow reciprocal approximations.
    pub arcp: bool,
    /// Allow floating-point contraction.
    pub contract: bool,
    /// Allow approximations for library functions.
    pub afn: bool,
    /// Allow reassociation of operations.
    pub reassoc: bool,
}

impl FastMathFlags {
    /// Create flags with all fast-math optimizations enabled.
    pub fn fast() -> Self {
        Self {
            nnan: true,
            ninf: true,
            nsz: true,
            arcp: true,
            contract: true,
            afn: true,
            reassoc: true,
        }
    }

    /// Check if any flag is set.
    pub fn any(&self) -> bool {
        self.nnan || self.ninf || self.nsz || self.arcp || self.contract || self.afn || self.reassoc
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtomicOrdering {
    Unordered,
    Monotonic,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Atomic read-modify-write operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicRMWOp {
    Xchg,
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
    FAdd,
    FSub,
    FMax,
    FMin,
}

/// Synchronization scope for atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncScope {
    SingleThread,
    #[default]
    System,
}

#[derive(Debug, Clone)]
pub enum GepIndex {
    Const(usize),
    Value(Operand),
}

#[derive(Debug, Clone)]
pub enum MemoryOp {
    Alloca {
        ty: TypeIdx,
        num_elements: u32,
        inalloca: bool,
        align: Option<u32>,
        addr_space: Option<u32>,
    },
    Load {
        ptr: Operand,
        /// Align in bits.
        align: Option<u32>,
    },
    Store {
        value: Operand,
        ptr: Operand,
        align: Option<u32>,
    },
    GetElementPtr {
        ptr: Operand,
        indices: Vec<GepIndex>,
    },
    /// Atomic load operation.
    AtomicLoad {
        ptr: Operand,
        ordering: AtomicOrdering,
        align: Option<u32>,
        sync_scope: SyncScope,
    },
    /// Atomic store operation.
    AtomicStore {
        value: Operand,
        ptr: Operand,
        ordering: AtomicOrdering,
        align: Option<u32>,
        sync_scope: SyncScope,
    },
    /// Atomic read-modify-write operation.
    AtomicRMW {
        op: AtomicRMWOp,
        ptr: Operand,
        value: Operand,
        ordering: AtomicOrdering,
        sync_scope: SyncScope,
    },
    /// Atomic compare-and-exchange operation.
    CmpXchg {
        ptr: Operand,
        cmp: Operand,
        new_val: Operand,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        weak: bool,
        sync_scope: SyncScope,
    },
    /// Memory fence operation.
    Fence {
        ordering: AtomicOrdering,
        sync_scope: SyncScope,
    },
}

#[derive(Debug, Clone)]
pub enum OtherOp {
    Call(CallOp),
    Icmp {
        cond: IcmpCond,
        lhs: Operand,
        rhs: Operand,
    },
    Fcmp {
        cond: FcmpCond,
        lhs: Operand,
        rhs: Operand,
    },
    /// Conditional value selection (ternary operator).
    Select {
        cond: Operand,
        true_val: Operand,
        false_val: Operand,
    },
    /// Exception landing pad.
    LandingPad {
        result_ty: TypeIdx,
        cleanup: bool,
        clauses: Vec<LandingPadClause>,
    },
}

/// A clause in a landing pad instruction.
#[derive(Debug, Clone)]
pub enum LandingPadClause {
    /// Catch a specific exception type.
    Catch(Operand),
    /// Filter clause (array of type infos).
    Filter(Vec<Operand>),
}

#[derive(Debug, Clone)]
pub struct CallOp {
    pub tail: bool,
    pub musttail: bool,
    pub notail: bool,
    /// Must match the target fn cconv or ub.
    pub cconv: CConv,
    pub params: Vec<Operand>,
    pub ret_ty: TypeIdx,
    pub ret_attrs: Option<CallReturnAttrs>,
    pub addr_space: Option<u32>,
    /// Only needed if its a varargs function.
    pub fn_ty: Option<FunctionType>,
    pub fn_target: CallableValue,
}

#[derive(Debug, Clone, Copy)]
pub enum IcmpCond {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

#[derive(Debug, Clone, Copy)]
pub enum FcmpCond {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

#[derive(Debug, Clone)]
pub enum CallableValue {
    Symbol(FnIdx),
    Pointer(Operand, FunctionType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallReturnAttrs {
    pub zeroext: bool,
    pub signext: bool,
    pub noext: bool,
    pub inreg: bool,
}

/// A debug operation.
#[derive(Debug, Clone)]
pub enum DebugOp {
    /// Provides information about a local element (e.g., variable).
    Declare {
        /// Must be of pointer type. Usually used for variables allocated in a alloca or behind a pointer.
        address: Operand,
        variable: DebugVarIdx,
    },
    /// Provides information when a user source variable is set to a new value.
    ///
    /// Useful for variables whose address can't be taken.
    Value {
        new_value: Operand,
        variable: DebugVarIdx,
    },
    /// This marks the position where a source assignment occurred. It encodes the value of the variable.
    /// It references the store, if any, that performs the assignment, and the destination address.
    Assign {
        address: Operand,
        new_value: Operand,
        store_inst: InstIdx,
        variable: DebugVarIdx,
    },
}

/// Debug info for a variable.
#[derive(Debug, Clone)]
pub struct DebugVariable {
    /// The name.
    pub name: String,
    /// If it's a parameter, the number.
    pub parameter: Option<u32>,
    /// The type of the variable.
    pub ty: TypeIdx,
    /// The source location.
    pub location: Location,
}

macro_rules! binop_float {
    ($name:ident, $name_ex:ident, $variant:ident) => {
        pub fn $name(
            &mut self,
            lhs: Operand,
            rhs: Operand,
            location: Location,
        ) -> Result<Operand, Error> {
            self.$name_ex(lhs, rhs, FastMathFlags::default(), location)
        }

        pub fn $name_ex(
            &mut self,
            lhs: Operand,
            rhs: Operand,
            flags: FastMathFlags,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.add_instr((
                location,
                Instruction::BinaryOp(BinaryOp::$variant { lhs, rhs, flags }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }
    };
}

macro_rules! binop_with_overflow_flags {
    ($name:ident, $name_ex:ident, $variant:ident) => {
        pub fn $name(
            &mut self,
            lhs: &Operand,
            rhs: &Operand,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.add_instr((
                location,
                Instruction::BinaryOp(BinaryOp::$variant {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    nsw: false,
                    nuw: false,
                }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }

        pub fn $name_ex(
            &mut self,
            lhs: &Operand,
            rhs: &Operand,
            nsw: bool,
            nuw: bool,
            location: Location,
        ) -> Result<Operand, Error> {
            if lhs.get_type() != rhs.get_type() {
                return Err(Error::TypeMismatch {
                    found: rhs.get_type(),
                    expected: lhs.get_type(),
                });
            }

            let result_type = lhs.get_type();
            let idx = self.add_instr((
                location,
                Instruction::BinaryOp(BinaryOp::$variant {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    nsw,
                    nuw,
                }),
            ));

            Ok(Operand::Value(self.id(), idx, result_type))
        }
    };
}

impl Block {
    pub(crate) fn new(arguments: &[TypeIdx]) -> Self {
        Self {
            instructions: StandardSlab::new(),
            terminator: Terminator::Ret((Location::Unknown, None)),
            arguments: arguments.to_vec(),
            id: None,
            last_instr_idx: None,
        }
    }

    /// Get the block terminator.
    pub fn terminator(&self) -> &Terminator {
        &self.terminator
    }

    /// Get the instructions within this block.
    pub fn instructions(&self) -> &StandardSlab<(Location, Instruction)> {
        &self.instructions
    }

    fn add_instr(&mut self, value: (Location, Instruction)) -> InstIdx {
        let id = self.instructions.insert(value);

        self.last_instr_idx = Some(id);

        id
    }

    /// Get the id of the block.
    pub fn id(&self) -> BlockIdx {
        self.id.unwrap()
    }

    /// Get the argument at the given index.
    pub fn arg(&self, nth: usize) -> Result<Operand, Error> {
        self.arguments
            .get(nth)
            .map(|x| Operand::BlockArgument {
                block_idx: self.id.unwrap().to_idx(),
                nth,
                ty: *x,
            })
            .ok_or_else(|| Error::BlockArgNotFound {
                block_id: self.id(),
                nth,
            })
    }

    /// Get the block argument type ids.
    pub fn args(&self) -> &[TypeIdx] {
        &self.arguments
    }

    /// Create a return instruction.
    pub fn instr_ret(&mut self, value: Option<&Operand>, location: Location) {
        self.terminator = Terminator::Ret((location, value.cloned()));
    }

    /// Create an unconditional jump/branch instruction.
    pub fn instr_jmp(&mut self, target: BlockIdx, arguments: &[Operand], location: Location) {
        self.terminator = Terminator::Br {
            block: target,
            arguments: arguments.to_vec(),
            location,
        };
    }

    /// Add a conditional jump/branch instruction.
    pub fn instr_cond_jmp(
        &mut self,
        then_block: BlockIdx,
        else_block: BlockIdx,
        cond: &Operand,
        then_block_args: &[Operand],
        else_block_args: &[Operand],
        location: Location,
    ) {
        self.terminator = Terminator::CondBr {
            then_block,
            else_block,
            cond: cond.clone(),
            if_args: then_block_args.to_vec(),
            then_args: else_block_args.to_vec(),
            location,
        };
    }

    binop_with_overflow_flags!(instr_add, instr_add_ex, Add);
    binop_with_overflow_flags!(instr_sub, instr_sub_ex, Sub);
    binop_with_overflow_flags!(instr_mul, instr_mul_ex, Mul);

    pub fn instr_div(
        &mut self,
        lhs: &Operand,
        rhs: &Operand,
        signed: bool,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BinaryOp(BinaryOp::Div {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                signed,
                exact,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_rem(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        signed: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BinaryOp(BinaryOp::Rem { lhs, rhs, signed }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    binop_float!(instr_fadd, instr_fadd_ex, FAdd);
    binop_float!(instr_fsub, instr_fsub_ex, FSub);
    binop_float!(instr_fmul, instr_fmul_ex, FMul);
    binop_float!(instr_fdiv, instr_fdiv_ex, FDiv);
    binop_float!(instr_frem, instr_frem_ex, FRem);

    /// Floating-point negation.
    pub fn instr_fneg(&mut self, value: Operand, location: Location) -> Result<Operand, Error> {
        self.instr_fneg_ex(value, FastMathFlags::default(), location)
    }

    /// Floating-point negation with fast math flags.
    pub fn instr_fneg_ex(
        &mut self,
        value: Operand,
        flags: FastMathFlags,
        location: Location,
    ) -> Result<Operand, Error> {
        let result_type = value.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BinaryOp(BinaryOp::FNeg { value, flags }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_shl(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Shl { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_lshr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Lshr { lhs, rhs, exact }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_ashr(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        exact: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Ashr { lhs, rhs, exact }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_and(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::And { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_or(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        disjoint: bool,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Or { lhs, rhs, disjoint }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_xor(
        &mut self,
        lhs: Operand,
        rhs: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::BitwiseBinaryOp(BitwiseBinaryOp::Xor { lhs, rhs }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    pub fn instr_alloca(
        &mut self,
        pointer_type_idx: TypeIdx,
        num_elements: u32,
        align: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(pointer_type_idx);
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            panic!("invalid pointer type")
        };
        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::Alloca {
                ty: inner,
                num_elements,
                inalloca: false,
                align,
                addr_space: None,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, pointer_type_idx))
    }

    pub fn instr_alloca_ex(
        &mut self,
        pointer_type_idx: TypeIdx,
        num_elements: u32,
        align: Option<u32>,
        inalloca: bool,
        addr_space: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(pointer_type_idx);
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            panic!("invalid pointer type")
        };

        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::Alloca {
                ty: inner,
                num_elements,
                inalloca,
                align,
                addr_space,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, pointer_type_idx))
    }

    pub fn instr_load(
        &mut self,
        ptr: Operand,
        align: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(ptr.get_type());
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            return Err(Error::InvalidType {
                found: pointer_type.clone(),
                expected: "pointer like".to_string(),
            });
        };

        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::Load { ptr, align }),
        ));

        Ok(Operand::Value(self.id(), idx, inner))
    }

    pub fn instr_store(
        &mut self,
        ptr: Operand,
        value: Operand,
        align: Option<u32>,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<(), Error> {
        let pointer_type = type_storage.get_type_info(ptr.get_type());
        let inner = if let Type::Ptr { pointee, .. } = pointer_type.ty {
            pointee
        } else {
            return Err(Error::InvalidType {
                found: pointer_type.clone(),
                expected: "pointer".to_string(),
            });
        };

        if inner != value.get_type() {
            return Err(Error::TypeMismatch {
                expected: inner,
                found: value.get_type(),
            });
        }

        self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::Store { value, ptr, align }),
        ));

        Ok(())
    }

    pub fn instr_gep(
        &mut self,
        ptr: Operand,
        indices: &[GepIndex],
        result_type: TypeIdx,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        let pointer_type = type_storage.get_type_info(ptr.get_type());

        if !matches!(pointer_type.ty, Type::Ptr { .. }) {
            return Err(Error::InvalidType {
                found: pointer_type.clone(),
                expected: "pointer".to_string(),
            });
        }

        // TODO: Find result type automatically.

        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::GetElementPtr {
                ptr,
                indices: indices.to_vec(),
            }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    /// Add a function call instruction.
    pub fn instr_call(
        &mut self,
        fn_idx: FnIdx,
        params: &[Operand],
        ret_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::OtherOp(OtherOp::Call(CallOp {
                tail: false,
                musttail: false,
                notail: false,
                cconv: CConv::default(),
                params: params.to_vec(),
                ret_ty,
                ret_attrs: None,
                addr_space: None,
                fn_ty: None,
                fn_target: CallableValue::Symbol(fn_idx),
            })),
        ));

        Ok(Operand::Value(self.id(), idx, ret_ty))
    }

    pub fn instr_call_ex(&mut self, call_op: CallOp, location: Location) -> Result<Operand, Error> {
        let ret_ty = call_op.ret_ty;
        let idx = self
            .instructions
            .insert((location, Instruction::OtherOp(OtherOp::Call(call_op))));

        Ok(Operand::Value(self.id(), idx, ret_ty))
    }

    pub fn instr_icmp(
        &mut self,
        cond: IcmpCond,
        lhs: Operand,
        rhs: Operand,
        location: Location,
        type_storage: &TypeStorage,
    ) -> Result<Operand, Error> {
        if lhs.get_type() != rhs.get_type() {
            return Err(Error::TypeMismatch {
                found: rhs.get_type(),
                expected: lhs.get_type(),
            });
        }

        let result_type_idx = lhs.get_type();
        let idx = self.add_instr((
            location,
            Instruction::OtherOp(OtherOp::Icmp { cond, lhs, rhs }),
        ));

        let result_type = type_storage.get_type_info(result_type_idx);
        if let Type::Vector(_) = result_type.ty {
            Ok(Operand::Value(self.id(), idx, result_type_idx))
        } else {
            Ok(Operand::Value(
                self.id(),
                idx,
                type_storage
                    .i1_ty
                    .expect("i1 type missing - call storage.get_or_create_i1() first"),
            ))
        }
    }

    pub fn instr_dbg_declare(
        &mut self,
        address: Operand,
        variable: DebugVarIdx,
        location: Location,
    ) -> Result<(), Error> {
        self.add_instr((
            location,
            Instruction::DebugOp(DebugOp::Declare { address, variable }),
        ));

        Ok(())
    }

    pub fn instr_dbg_value(
        &mut self,
        new_value: Operand,
        variable: DebugVarIdx,
        location: Location,
    ) -> Result<(), Error> {
        self.add_instr((
            location,
            Instruction::DebugOp(DebugOp::Value {
                new_value,
                variable,
            }),
        ));

        Ok(())
    }

    pub fn instr_dbg_assign(
        &mut self,
        address: Operand,
        new_value: Operand,
        variable: DebugVarIdx,
        store_instr_idx: InstIdx,
        location: Location,
    ) -> Result<(), Error> {
        self.add_instr((
            location,
            Instruction::DebugOp(DebugOp::Assign {
                address,
                new_value,
                store_inst: store_instr_idx,
                variable,
            }),
        ));

        Ok(())
    }

    pub fn get_last_instr_idx(&self) -> Option<InstIdx> {
        self.last_instr_idx
    }

    // ==================== Conversion/Cast Instructions ====================

    /// Truncate integer to a smaller type.
    pub fn instr_trunc(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::Trunc { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Zero extend integer to a larger type.
    pub fn instr_zext(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::ZExt { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Sign extend integer to a larger type.
    pub fn instr_sext(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::SExt { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Truncate floating-point to a smaller type.
    pub fn instr_fptrunc(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::FPTrunc { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Extend floating-point to a larger type.
    pub fn instr_fpext(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::FPExt { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert floating-point to unsigned integer.
    pub fn instr_fptoui(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::FPToUI { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert floating-point to signed integer.
    pub fn instr_fptosi(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::FPToSI { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert unsigned integer to floating-point.
    pub fn instr_uitofp(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::UIToFP { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert signed integer to floating-point.
    pub fn instr_sitofp(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::SIToFP { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert pointer to integer.
    pub fn instr_ptrtoint(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::PtrToInt { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Convert integer to pointer.
    pub fn instr_inttoptr(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::IntToPtr { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Bitcast (reinterpret bits without changing them).
    pub fn instr_bitcast(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::Bitcast { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    /// Cast pointer to different address space.
    pub fn instr_addrspacecast(
        &mut self,
        value: Operand,
        target_ty: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::ConversionOp(ConversionOp::AddrSpaceCast { value, target_ty }),
        ));
        Ok(Operand::Value(self.id(), idx, target_ty))
    }

    // ==================== Select Instruction ====================

    /// Conditional value selection (ternary operator).
    pub fn instr_select(
        &mut self,
        cond: Operand,
        true_val: Operand,
        false_val: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        if true_val.get_type() != false_val.get_type() {
            return Err(Error::TypeMismatch {
                found: false_val.get_type(),
                expected: true_val.get_type(),
            });
        }

        let result_type = true_val.get_type();
        let idx = self.add_instr((
            location,
            Instruction::OtherOp(OtherOp::Select {
                cond,
                true_val,
                false_val,
            }),
        ));

        Ok(Operand::Value(self.id(), idx, result_type))
    }

    // ==================== Switch Terminator ====================

    /// Create a switch (multi-way branch) terminator.
    pub fn instr_switch(
        &mut self,
        value: Operand,
        default_block: BlockIdx,
        default_args: &[Operand],
        cases: Vec<SwitchCase>,
        location: Location,
    ) {
        self.terminator = Terminator::Switch {
            value,
            default_block,
            default_args: default_args.to_vec(),
            cases,
            location,
        };
    }

    // ==================== Vector Operations ====================

    /// Insert an element into a vector.
    pub fn instr_insertelement(
        &mut self,
        vector: Operand,
        element: Operand,
        idx: Operand,
        location: Location,
    ) -> Result<Operand, Error> {
        let result_type = vector.get_type();
        let inst_idx = self.add_instr((
            location,
            Instruction::VectorOp(VectorOp::InsertElement {
                vector,
                element,
                idx,
            }),
        ));
        Ok(Operand::Value(self.id(), inst_idx, result_type))
    }

    /// Shuffle elements from two vectors using a mask.
    pub fn instr_shufflevector(
        &mut self,
        vec1: Operand,
        vec2: Operand,
        mask: Vec<i32>,
        result_type: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::VectorOp(VectorOp::ShuffleVector { vec1, vec2, mask }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    // ==================== Aggregate Operations ====================

    /// Extract a value from an aggregate (struct or array).
    pub fn instr_extractvalue(
        &mut self,
        aggregate: Operand,
        indices: &[u32],
        result_type: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::AggregateOp(AggregateOp::ExtractValue {
                aggregate,
                indices: indices.to_vec(),
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    /// Insert a value into an aggregate (struct or array).
    pub fn instr_insertvalue(
        &mut self,
        aggregate: Operand,
        element: Operand,
        indices: &[u32],
        location: Location,
    ) -> Result<Operand, Error> {
        let result_type = aggregate.get_type();
        let idx = self.add_instr((
            location,
            Instruction::AggregateOp(AggregateOp::InsertValue {
                aggregate,
                element,
                indices: indices.to_vec(),
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    // ==================== Atomic Operations ====================

    /// Atomic load.
    pub fn instr_atomic_load(
        &mut self,
        ptr: Operand,
        ordering: AtomicOrdering,
        align: Option<u32>,
        result_type: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::AtomicLoad {
                ptr,
                ordering,
                align,
                sync_scope: SyncScope::default(),
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    /// Atomic store.
    pub fn instr_atomic_store(
        &mut self,
        ptr: Operand,
        value: Operand,
        ordering: AtomicOrdering,
        align: Option<u32>,
        location: Location,
    ) {
        self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::AtomicStore {
                value,
                ptr,
                ordering,
                align,
                sync_scope: SyncScope::default(),
            }),
        ));
    }

    /// Atomic read-modify-write operation.
    pub fn instr_atomicrmw(
        &mut self,
        op: AtomicRMWOp,
        ptr: Operand,
        value: Operand,
        ordering: AtomicOrdering,
        location: Location,
    ) -> Result<Operand, Error> {
        let result_type = value.get_type();
        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::AtomicRMW {
                op,
                ptr,
                value,
                ordering,
                sync_scope: SyncScope::default(),
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    /// Atomic compare-and-exchange.
    pub fn instr_cmpxchg(
        &mut self,
        ptr: Operand,
        cmp: Operand,
        new_val: Operand,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        weak: bool,
        result_type: TypeIdx,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::CmpXchg {
                ptr,
                cmp,
                new_val,
                success_ordering,
                failure_ordering,
                weak,
                sync_scope: SyncScope::default(),
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_type))
    }

    /// Memory fence.
    pub fn instr_fence(&mut self, ordering: AtomicOrdering, location: Location) {
        self.add_instr((
            location,
            Instruction::MemoryOp(MemoryOp::Fence {
                ordering,
                sync_scope: SyncScope::default(),
            }),
        ));
    }

    // ==================== Exception Handling ====================

    /// Invoke a function that may throw an exception.
    pub fn instr_invoke(
        &mut self,
        call: CallOp,
        normal_dest: BlockIdx,
        normal_args: &[Operand],
        unwind_dest: BlockIdx,
        unwind_args: &[Operand],
        location: Location,
    ) {
        self.terminator = Terminator::Invoke {
            call,
            normal_dest,
            normal_args: normal_args.to_vec(),
            unwind_dest,
            unwind_args: unwind_args.to_vec(),
            location,
        };
    }

    /// Resume propagation of an exception.
    pub fn instr_resume(&mut self, value: Operand, location: Location) {
        self.terminator = Terminator::Resume { value, location };
    }

    /// Mark code as unreachable.
    pub fn instr_unreachable(&mut self, location: Location) {
        self.terminator = Terminator::Unreachable { location };
    }

    /// Landing pad for exception handling.
    pub fn instr_landingpad(
        &mut self,
        result_ty: TypeIdx,
        cleanup: bool,
        clauses: Vec<LandingPadClause>,
        location: Location,
    ) -> Result<Operand, Error> {
        let idx = self.add_instr((
            location,
            Instruction::OtherOp(OtherOp::LandingPad {
                result_ty,
                cleanup,
                clauses,
            }),
        ));
        Ok(Operand::Value(self.id(), idx, result_ty))
    }
}
