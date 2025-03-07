use crate::{
    module::TypeIdx,
    types::{Type, TypeStorage},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataLayout {
    pub endianess: Endianess,
    /// Specifies the natural alignment of the stack in bits.
    /// Alignment promotion of stack variables is limited to the natural stack alignment to avoid dynamic stack realignment.
    /// The stack alignment must be a multiple of 8-bits.
    /// If omitted, the natural stack alignment defaults to “unspecified”, which does not prevent any alignment promotions.
    pub stack_alignment: Option<u64>,
    /// Specifies the address space that corresponds to program memory.
    /// Harvard architectures can use this to specify what space LLVM should place things such as functions into.
    /// If omitted, the program memory space defaults to the default address space of 0,
    /// which corresponds to a Von Neumann architecture that has code and data in the same space.
    pub address_space: Option<u64>,
    /// Specifies the address space to be used by default when creating global variables.
    /// If omitted, the globals address space defaults to the default address space 0.
    /// Note: variable declarations without an address space are always created in address space 0,
    /// this property only affects the default value to be used when creating globals without additional contextual information (e.g. in LLVM passes).
    pub globals_address_space: Option<u64>,
    /// Specifies the address space of objects created by ‘alloca’. Defaults to the default address space of 0.
    pub alloca_address_space: Option<u64>,
    /// Size, alignment, abi, etc info for types.
    pub type_infos: Vec<TypeLayout>,
    /// If present, specifies that llvm names are mangled in the output.
    /// Symbols prefixed with the mangling escape character \01 are passed through directly to the assembler without the escape character.
    pub mangling: Option<Mangling>,
    /// This specifies a set of native integer widths for the target CPU in bits.
    /// For example, it might contain n32 for 32-bit PowerPC, n32:64 for PowerPC 64, or n8:16:32:64 for X86-64.
    /// Elements of this set are considered to support most general arithmetic operations efficiently.
    pub native_integer_widths: Vec<u32>,
    /// This specifies pointer types with the specified address spaces as Non-Integral Pointer Types.
    /// The 0 address space cannot be specified as non-integral.
    pub non_integral_address_spaces: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeLayout {
    Pointer {
        /// <size> The size of the type (bits).
        size: u32,
        /// <abi> The abi alignment (bits).
        abi: u32,
        /// <pref> The preferred alignment (bits).
        prefered: Option<u32>,
        /// [n]. Only used for pointer types.
        address_space: Option<u32>,
        index_size: Option<u32>,
    },
    FunctionPointer {
        /// True: The alignment of function pointers is independent of the alignment of functions, and is a multiple of <abi>.
        ///
        /// False: The alignment of function pointers is a multiple of the explicit alignment specified on the function, and is a multiple of <abi>.
        align_independent: bool,
        /// <abi> The abi alignment.
        abi: u32,
    },
    Int {
        /// <size> The size of the type (bits).
        size: u32,
        /// <abi> The abi alignment (bits).
        abi: u32,
        /// <pref> The preferred alignment (bits).
        prefered: Option<u32>,
    },
    Vector {
        /// <size> The size of the type (bits).
        size: u32,
        /// <abi> The abi alignment (bits).
        abi: u32,
        /// <pref> The preferred alignment (bits).
        prefered: Option<u32>,
    },
    Float {
        /// <size> The size of the type (bits).
        size: u32,
        /// <abi> The abi alignment (bits).
        abi: u32,
        /// <pref> The preferred alignment (bits).
        prefered: Option<u32>,
    },
    Aggregate {
        /// <abi> The abi alignment (bits).
        abi: u32,
        /// <pref> The preferred alignment (bits).
        prefered: Option<u32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Endianess {
    #[default]
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mangling {
    Elf,
    Goff,
    Mips,
    MachO,
    Windowsx86Coff,
    WindowsCoff,
    XCoff,
}

impl Default for DataLayout {
    fn default() -> Self {
        Self {
            endianess: Endianess::Little,
            stack_alignment: None,
            address_space: None,
            alloca_address_space: None,
            globals_address_space: None,
            mangling: None,
            native_integer_widths: Vec::new(),
            non_integral_address_spaces: Vec::new(),

            type_infos: vec![
                TypeLayout::Pointer {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                    address_space: None,
                    index_size: None,
                },
                TypeLayout::Int {
                    size: 1,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 8,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Int {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Int {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Float {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Float {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Float {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Float {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Vector {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Vector {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Aggregate {
                    abi: 0,
                    prefered: Some(64),
                },
            ],
        }
    }
}

impl DataLayout {
    pub fn default_host() -> Self {
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Self::default_linux_x86_64()
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Self::default_macos_aarch64()
        } else {
            Self::default()
        }
    }

    pub fn default_linux_x86_64() -> Self {
        // example linux x86_64 "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
        Self {
            endianess: Endianess::Little,
            stack_alignment: Some(128),
            address_space: None,
            alloca_address_space: None,
            globals_address_space: None,
            mangling: Some(Mangling::Elf),
            native_integer_widths: vec![8, 16, 32, 64],
            non_integral_address_spaces: Vec::new(),

            type_infos: vec![
                TypeLayout::Pointer {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                    address_space: None,
                    index_size: None,
                },
                TypeLayout::Pointer {
                    size: 32,
                    abi: 32,
                    prefered: None,
                    address_space: Some(270),
                    index_size: None,
                },
                TypeLayout::Pointer {
                    size: 32,
                    abi: 32,
                    prefered: None,
                    address_space: Some(271),
                    index_size: None,
                },
                TypeLayout::Pointer {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                    address_space: Some(272),
                    index_size: None,
                },
                TypeLayout::Int {
                    size: 1,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 8,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Int {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Int {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Int {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Float {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Float {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Float {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Float {
                    size: 80,
                    abi: 128,
                    prefered: None,
                },
                TypeLayout::Float {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Vector {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Vector {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Aggregate {
                    abi: 0,
                    prefered: Some(64),
                },
            ],
        }
    }

    pub fn default_macos_aarch64() -> Self {
        // example macos arm "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
        Self {
            endianess: Endianess::Little,
            stack_alignment: Some(128),
            address_space: None,
            alloca_address_space: None,
            globals_address_space: None,
            mangling: Some(Mangling::MachO),
            native_integer_widths: vec![32, 64],
            non_integral_address_spaces: Vec::new(),

            type_infos: vec![
                TypeLayout::Pointer {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                    address_space: None,
                    index_size: None,
                },
                TypeLayout::Int {
                    size: 1,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 8,
                    abi: 8,
                    prefered: Some(8),
                },
                TypeLayout::Int {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Int {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Int {
                    size: 64,
                    abi: 64,
                    prefered: None,
                },
                TypeLayout::Int {
                    size: 128,
                    abi: 128,
                    prefered: None,
                },
                TypeLayout::Float {
                    size: 16,
                    abi: 16,
                    prefered: Some(16),
                },
                TypeLayout::Float {
                    size: 32,
                    abi: 32,
                    prefered: Some(32),
                },
                TypeLayout::Float {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Float {
                    size: 80,
                    abi: 128,
                    prefered: None,
                },
                TypeLayout::Float {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Vector {
                    size: 64,
                    abi: 64,
                    prefered: Some(64),
                },
                TypeLayout::Vector {
                    size: 128,
                    abi: 128,
                    prefered: Some(128),
                },
                TypeLayout::Aggregate {
                    abi: 0,
                    prefered: Some(64),
                },
                TypeLayout::FunctionPointer {
                    align_independent: false,
                    abi: 32,
                },
            ],
        }
    }

    pub fn get_type_size(&self, storage: &TypeStorage, ty: TypeIdx) -> u32 {
        let ty = &storage.types[ty];
        match &ty.ty {
            Type::Int(bits) => {
                let mut closest_found_lower = None;

                for found in &self.type_infos {
                    if let TypeLayout::Int { size, abi, .. } = *found {
                        if size == *bits {
                            return size.max(abi);
                        }

                        if size > *bits
                            && (closest_found_lower.is_none() || Some(size) < closest_found_lower)
                        {
                            closest_found_lower = Some(size.max(abi));
                        }
                    }
                }

                if let Some(closest_found_lower) = closest_found_lower {
                    closest_found_lower
                } else {
                    bits.next_power_of_two()
                }
            }
            Type::Half => 16,
            Type::BFloat => 16,
            Type::Float => 32,
            Type::Double => 64,
            Type::Fp128 => 128,
            Type::X86Fp80 => 80,
            Type::PpcFp128 => 128,
            Type::Ptr(addr_space) => {
                let mut found_value = None;
                for found in &self.type_infos {
                    if let TypeLayout::Pointer {
                        size,
                        address_space,
                        ..
                    } = *found
                    {
                        if address_space == *addr_space {
                            return size;
                        }

                        if addr_space.is_none() && address_space == Some(0) {
                            return size;
                        }

                        if address_space.is_none() || address_space == Some(0) {
                            found_value = Some(size);
                        }
                    }
                }
                found_value.expect("should have a ptr type")
            }
            Type::Vector(vector_type) => {
                let align = self.get_type_align(storage, vector_type.ty);
                align * vector_type.size
            }
            Type::Array(array_type) => {
                let align = self.get_type_align(storage, array_type.ty);
                align * array_type.size as u32
            }
            Type::Struct(struct_type) => {
                let mut size = 0;
                let mut align = 1;

                for field in &struct_type.fields {
                    let field_align = self.get_type_align(storage, *field);
                    align = align.max(field_align);

                    if size % field_align != 0 {
                        let padding = (field_align - (size % field_align)) % field_align;
                        size += padding;
                    }

                    let field_size = self.get_type_size(storage, *field);

                    size += field_size;
                }

                if size % align == 0 {
                    size
                } else {
                    let padding = (align - (size % align)) % align;
                    size += padding;
                    size
                }
            }
        }
    }

    pub fn get_type_abi_align(&self, storage: &TypeStorage, ty: TypeIdx) -> u32 {
        let ty = &storage.types[ty];
        match &ty.ty {
            Type::Int(bits) => {
                let mut closest_found_lower = None;
                let mut closest_found_lower_abi = None;

                for found in &self.type_infos {
                    if let TypeLayout::Int {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == *bits {
                            return abi;
                        }

                        if size > *bits
                            && (closest_found_lower.is_none() || Some(size) < closest_found_lower)
                        {
                            closest_found_lower = Some(size);
                            closest_found_lower_abi = Some(abi);
                        }
                    }
                }

                closest_found_lower_abi.unwrap()
            }
            Type::Half | Type::BFloat => {
                let bits = 16;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(abi);
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Float => {
                let bits = 32;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(abi);
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Double => {
                let bits = 64;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(abi);
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Fp128 | Type::PpcFp128 => {
                let bits = 128;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(abi);
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::X86Fp80 => {
                let bits = 80;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered: _,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(abi);
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Ptr(addr_space) => {
                let mut found_value = None;
                for found in &self.type_infos {
                    if let TypeLayout::Pointer {
                        address_space, abi, ..
                    } = *found
                    {
                        if address_space == *addr_space {
                            return abi;
                        }

                        if addr_space.is_none() && address_space == Some(0) {
                            return abi;
                        }

                        if address_space.is_none() || address_space == Some(0) {
                            found_value = Some(abi);
                        }
                    }
                }
                found_value.expect("should have a ptr type")
            }
            Type::Vector(vector_type) => self.get_type_abi_align(storage, vector_type.ty),
            Type::Array(array_type) => self.get_type_abi_align(storage, array_type.ty),
            Type::Struct(struct_type) => {
                if struct_type.packed {
                    1
                } else {
                    let mut max_align = 1;

                    for field in &struct_type.fields {
                        max_align = max_align.max(self.get_type_abi_align(storage, *field));
                    }

                    max_align
                }
            }
        }
    }

    /// Returns preferred align if there is one.
    pub fn get_type_align(&self, storage: &TypeStorage, ty: TypeIdx) -> u32 {
        let ty = storage.get_type_info(ty);
        match &ty.ty {
            Type::Int(bits) => {
                let mut closest_found_lower = None;
                let mut closest_found_lower_align = None;
                let mut max_align = 1;

                for found in &self.type_infos {
                    if let TypeLayout::Int {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == *bits {
                            return abi;
                        }

                        if size > *bits
                            && (closest_found_lower.is_none() || Some(size) < closest_found_lower)
                        {
                            closest_found_lower = Some(size);
                            closest_found_lower_align = Some(prefered.unwrap_or(abi));
                        }

                        max_align = max_align.max(abi);
                    }
                }

                closest_found_lower_align.unwrap_or(max_align)
            }
            Type::Half | Type::BFloat => {
                let bits = 16;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return prefered.unwrap_or(abi);
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(prefered.unwrap_or(abi));
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Float => {
                let bits = 32;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return prefered.unwrap_or(abi);
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(prefered.unwrap_or(abi));
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Double => {
                let bits = 64;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return prefered.unwrap_or(abi);
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(prefered.unwrap_or(abi));
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Fp128 | Type::PpcFp128 => {
                let bits = 128;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return prefered.unwrap_or(abi);
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(prefered.unwrap_or(abi));
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::X86Fp80 => {
                let bits = 80;
                let mut closest_found_abi = None;
                let mut closest_found = None;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return prefered.unwrap_or(abi);
                        }

                        if size > bits && (closest_found.is_none() || Some(size) < closest_found) {
                            closest_found = Some(size);
                            closest_found_abi = Some(prefered.unwrap_or(abi));
                        }
                    }
                }

                closest_found_abi.unwrap()
            }
            Type::Ptr(addr_space) => {
                let mut found_value = None;
                for found in &self.type_infos {
                    if let TypeLayout::Pointer {
                        address_space,
                        abi,
                        prefered,
                        ..
                    } = *found
                    {
                        if address_space == *addr_space {
                            return prefered.unwrap_or(abi);
                        }

                        if addr_space.is_none() && address_space == Some(0) {
                            return prefered.unwrap_or(abi);
                        }

                        if address_space.is_none() || address_space == Some(0) {
                            found_value = Some(prefered.unwrap_or(abi));
                        }
                    }
                }
                found_value.expect("should have a ptr type")
            }
            Type::Vector(vector_type) => self.get_type_align(storage, vector_type.ty),
            Type::Array(array_type) => self.get_type_align(storage, array_type.ty),
            Type::Struct(struct_type) => {
                if struct_type.packed {
                    1
                } else {
                    let mut max_align = 1;

                    for field in &struct_type.fields {
                        max_align = max_align.max(self.get_type_align(storage, *field));
                    }

                    max_align
                }
            }
        }
    }

    pub fn to_llvm_string(&self) -> String {
        // example linux x86_64 "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
        // example macos arm "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
        let mut target = String::new();

        match self.endianess {
            Endianess::Little => target.push('e'),
            Endianess::Big => target.push('E'),
        }

        if let Some(mangling) = self.mangling {
            match mangling {
                Mangling::Elf => target.push_str("-m:e"),
                Mangling::Goff => target.push_str("-m:l"),
                Mangling::Mips => target.push_str("-m:m"),
                Mangling::MachO => target.push_str("-m:o"),
                Mangling::Windowsx86Coff => target.push_str("-m:x"),
                Mangling::WindowsCoff => target.push_str("-m:w"),
                Mangling::XCoff => target.push_str("-m:a"),
            }
        }

        for ty in &self.type_infos {
            match ty {
                TypeLayout::Pointer {
                    size,
                    abi,
                    prefered,
                    address_space,
                    index_size,
                } => {
                    target.push_str("-p");

                    if let Some(address_space) = address_space {
                        target.push_str(&address_space.to_string());
                    }

                    target.push_str(&format!(":{size}"));
                    target.push_str(&format!(":{abi}"));

                    if let Some(prefered) = prefered {
                        target.push_str(&format!(":{prefered}"));
                    }

                    if let Some(index_size) = index_size {
                        target.push_str(&format!(":{index_size}"));
                    }
                }
                TypeLayout::FunctionPointer {
                    align_independent,
                    abi,
                } => {
                    target.push_str("-F");

                    if *align_independent {
                        target.push('i');
                    } else {
                        target.push('n');
                    }

                    target.push_str(&format!("{abi}"));
                }
                TypeLayout::Int {
                    size,
                    abi,
                    prefered,
                } => {
                    target.push_str("-i");

                    target.push_str(&format!("{size}"));
                    target.push_str(&format!(":{abi}"));

                    if let Some(prefered) = prefered {
                        target.push_str(&format!(":{prefered}"));
                    }
                }
                TypeLayout::Vector {
                    size,
                    abi,
                    prefered,
                } => {
                    target.push_str("-v");

                    target.push_str(&format!("{size}"));
                    target.push_str(&format!(":{abi}"));

                    if let Some(prefered) = prefered {
                        target.push_str(&format!(":{prefered}"));
                    }
                }
                TypeLayout::Float {
                    size,
                    abi,
                    prefered,
                } => {
                    target.push_str("-f");

                    target.push_str(&format!("{size}"));
                    target.push_str(&format!(":{abi}"));

                    if let Some(prefered) = prefered {
                        target.push_str(&format!(":{prefered}"));
                    }
                }
                TypeLayout::Aggregate { abi, prefered } => {
                    target.push_str("-a");

                    target.push_str(&format!(":{abi}"));

                    if let Some(prefered) = prefered {
                        target.push_str(&format!(":{prefered}"));
                    }
                }
            }
        }

        if let Some(stack_align) = self.stack_alignment {
            target.push_str(&format!("-S{stack_align}"));
        }

        if let Some(address_space) = self.address_space {
            target.push_str(&format!("-P{address_space}"));
        }

        if let Some(globals_address_space) = self.globals_address_space {
            target.push_str(&format!("-G{globals_address_space}"));
        }

        if let Some(alloca_address_space) = self.alloca_address_space {
            target.push_str(&format!("-A{alloca_address_space}"));
        }

        if !self.native_integer_widths.is_empty() {
            target.push_str("-n");
            let mut iter = self.native_integer_widths.iter();
            target.push_str(&iter.next().unwrap().to_string());
            for native in iter {
                target.push_str(&format!(":{native}"));
            }
        }

        if !self.non_integral_address_spaces.is_empty() {
            target.push_str("-ni");
            let mut iter = self.non_integral_address_spaces.iter();
            target.push_str(&iter.next().unwrap().to_string());
            for native in iter {
                target.push_str(&format!(":{native}"));
            }
        }

        target
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        common::Location,
        types::{StructType, Type, TypeStorage},
    };

    use super::DataLayout;

    #[test]
    fn test_get_type_size() {
        let mut storage = TypeStorage::new();
        let datalayout = DataLayout::default();

        let i1_ty = storage.add_type(Type::Int(1), Location::Unknown, None);
        let i2_ty = storage.add_type(Type::Int(2), Location::Unknown, None);
        let i4_ty = storage.add_type(Type::Int(4), Location::Unknown, None);
        let i8_ty = storage.add_type(Type::Int(8), Location::Unknown, None);
        let i32_ty = storage.add_type(Type::Int(32), Location::Unknown, None);
        let i64_ty = storage.add_type(Type::Int(64), Location::Unknown, None);
        assert_eq!(datalayout.get_type_size(&storage, i1_ty), 8);
        assert_eq!(datalayout.get_type_size(&storage, i2_ty), 8);
        assert_eq!(datalayout.get_type_size(&storage, i4_ty), 8);
        assert_eq!(datalayout.get_type_size(&storage, i8_ty), 8);
        /*
        assert_eq!(datalayout.get_type_size(&Type::Int(16)), 16);
        assert_eq!(datalayout.get_type_size(&Type::Int(24)), 32);
        assert_eq!(datalayout.get_type_size(&Type::Int(32)), 32);
        assert_eq!(datalayout.get_type_size(&Type::Int(36)), 64);
        assert_eq!(datalayout.get_type_size(&Type::Int(64)), 64);
        assert_eq!(datalayout.get_type_size(&Type::Int(80)), 128);
        assert_eq!(datalayout.get_type_size(&Type::Int(128)), 128);
        assert_eq!(datalayout.get_type_size(&Type::Int(130)), 256);

        assert_eq!(datalayout.get_type_size(&Type::Half), 16);
        assert_eq!(datalayout.get_type_size(&Type::BFloat), 16);
        assert_eq!(datalayout.get_type_size(&Type::Float), 32);
        assert_eq!(datalayout.get_type_size(&Type::Double), 64);
        assert_eq!(datalayout.get_type_size(&Type::X86Fp80), 80);
        assert_eq!(datalayout.get_type_size(&Type::PpcFp128), 128);
         */

        let ptr_ty = storage.add_type(Type::Ptr(None), Location::Unknown, None);

        assert_eq!(datalayout.get_type_size(&storage, ptr_ty), 64);
        // assert_eq!(datalayout.get_type_size(&Type::Ptr(Some(2))), 64);

        let struct_ty = storage.add_type(
            Type::Struct(Arc::new(StructType {
                packed: false,
                ident: None,
                fields: vec![i64_ty, i32_ty, i32_ty],
            })),
            Location::Unknown,
            None,
        );

        assert_eq!(datalayout.get_type_size(&storage, struct_ty), 128);

        let struct_ty = storage.add_type(
            Type::Struct(Arc::new(StructType {
                packed: false,
                ident: None,
                fields: vec![i64_ty, i32_ty, i32_ty, i32_ty],
            })),
            Location::Unknown,
            None,
        );

        assert_eq!(datalayout.get_type_size(&storage, struct_ty), 192);
    }

    #[test]
    fn test_get_type_align() {
        let mut type_storage = TypeStorage::new();
        let datalayout = DataLayout::default();

        let i1_ty = type_storage.add_type(Type::Int(1), Location::Unknown, None);
        let i2_ty = type_storage.add_type(Type::Int(2), Location::Unknown, None);
        let i4_ty = type_storage.add_type(Type::Int(4), Location::Unknown, None);
        let i8_ty = type_storage.add_type(Type::Int(8), Location::Unknown, None);
        let i32_ty = type_storage.add_type(Type::Int(32), Location::Unknown, None);
        let i64_ty = type_storage.add_type(Type::Int(64), Location::Unknown, None);

        assert_eq!(datalayout.get_type_align(&type_storage, i1_ty), 8);
        assert_eq!(datalayout.get_type_align(&type_storage, i2_ty), 8);
        assert_eq!(datalayout.get_type_align(&type_storage, i4_ty), 8);
        assert_eq!(datalayout.get_type_align(&type_storage, i8_ty), 8);
        /*
        assert_eq!(datalayout.get_type_align(&Type::Int(16)), 16);
        assert_eq!(datalayout.get_type_align(&Type::Int(24)), 32);
        assert_eq!(datalayout.get_type_align(&Type::Int(32)), 32);
        assert_eq!(datalayout.get_type_align(&Type::Int(36)), 64);
        assert_eq!(datalayout.get_type_align(&Type::Int(64)), 64);
        assert_eq!(datalayout.get_type_align(&Type::Int(80)), 64);
        assert_eq!(datalayout.get_type_align(&Type::Int(128)), 64);
        assert_eq!(datalayout.get_type_align(&Type::Int(130)), 64);

        assert_eq!(datalayout.get_type_align(&Type::Half), 16);
        assert_eq!(datalayout.get_type_align(&Type::BFloat), 16);
        assert_eq!(datalayout.get_type_align(&Type::Float), 32);
        assert_eq!(datalayout.get_type_align(&Type::Double), 64);
        assert_eq!(datalayout.get_type_align(&Type::X86Fp80), 128);
        assert_eq!(datalayout.get_type_align(&Type::PpcFp128), 128);

        assert_eq!(datalayout.get_type_align(&Type::Ptr(None)), 64);
        assert_eq!(datalayout.get_type_align(&Type::Ptr(Some(2))), 64);
        */

        let struct_ty = type_storage.add_type(
            Type::Struct(Arc::new(StructType {
                packed: false,
                ident: None,
                fields: vec![i64_ty, i32_ty, i32_ty],
            })),
            Location::Unknown,
            None,
        );

        assert_eq!(datalayout.get_type_align(&type_storage, struct_ty), 64);

        let struct_ty = type_storage.add_type(
            Type::Struct(Arc::new(StructType {
                packed: false,
                ident: None,
                fields: vec![i64_ty, i32_ty, i32_ty, i32_ty],
            })),
            Location::Unknown,
            None,
        );

        assert_eq!(datalayout.get_type_align(&type_storage, struct_ty), 64);
    }

    #[test]
    fn test_datalayout_string() {
        let datalayout = DataLayout::default();

        assert_eq!(
            datalayout.to_llvm_string(),
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a:0:64"
        );
    }
}
