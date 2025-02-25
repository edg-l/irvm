#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Linkage {
    #[default]
    Private,
    Internal,
    AvailableExternally,
    LinkOnce,
    Weak,
    Common,
    Appending,
    ExternWeak,
    LinkOnceOdr,
    WeakOdr,
    External,
}

#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CConv {
    #[default]
    Ccc,
    FastCc,
    ColdCc,
    GhCcc,
    Cc11,
    Anyregcc,
    PreserveMostCc,
    PreserveAllCc,
    PreserveNoneCc,
    CxxFastTlsCc,
    TailCc,
    SwiftCc,
    CfGuardCheckCc,
}

#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Visibility {
    #[default]
    Default,
    Hidden,
    Protected
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DllStorageClass {
    Import,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThreadLocalStorageModel {
    LocalDynamic,
    InitialExec,
    LocalExec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimePreemption {
    DsoPreemptable,
    DsoLocal,
}
