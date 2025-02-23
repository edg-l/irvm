use target_lexicon::Triple;

use crate::datalayout::DataLayout;

#[derive(Debug, Clone)]
pub struct Module {
    pub data_layout: DataLayout,
    pub target_triple: Triple,
}
