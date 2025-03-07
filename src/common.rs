use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

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
    Protected,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    Unknown,
    File(FileLocation),
}

impl Location {
    pub fn unknown() -> Self {
        Self::Unknown
    }

    pub fn file(file: &Path, line: u32, col: u32) -> Self {
        Self::File(FileLocation::new(file, line, col))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLocation {
    pub file: Arc<PathBuf>,
    pub line: u32,
    pub col: u32,
}

impl FileLocation {
    pub fn new(file: &Path, line: u32, col: u32) -> Self {
        Self {
            file: file.to_path_buf().into(),
            line,
            col,
        }
    }
}
