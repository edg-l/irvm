use std::u32;

use crate::types::Type;

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
    pub fn get_type_size(&self, ty: &Type) -> u32 {
        match ty {
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

                        dbg!(closest_found_lower);
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
            Type::X86Fp80 => 128,
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
            Type::Vector(vector_type) => todo!(),
            Type::Array(array_type) => todo!(),
            Type::Struct(struct_type) => todo!(),
            Type::Opaque(_) => todo!(),
        }
    }

    pub fn get_type_abi_align(&self, ty: &Type) -> u32 {
        match ty {
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
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = abi;
                        }
                    }
                }

                closest_found_abi
            }
            Type::Float => {
                let bits = 32;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = abi;
                        }
                    }
                }

                closest_found_abi
            }
            Type::Double => {
                let bits = 64;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = abi;
                        }
                    }
                }

                closest_found_abi
            }
            Type::Fp128 | Type::PpcFp128 => {
                let bits = 128;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = abi;
                        }
                    }
                }

                closest_found_abi
            }
            Type::X86Fp80 => {
                let bits = 80;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
                for found in &self.type_infos {
                    if let TypeLayout::Float {
                        size,
                        abi,
                        prefered,
                    } = *found
                    {
                        if size == bits {
                            return abi;
                        }

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = abi;
                        }
                    }
                }

                closest_found_abi
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
            Type::Vector(vector_type) => todo!(),
            Type::Array(array_type) => todo!(),
            Type::Struct(struct_type) => todo!(),
            Type::Opaque(_) => todo!(),
        }
    }

    /// Returns preferred align if there is one.
    pub fn get_type_align(&self, ty: &Type) -> u32 {
        match ty {
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
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
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

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = prefered.unwrap_or(abi);
                        }
                    }
                }

                closest_found_abi
            }
            Type::Float => {
                let bits = 32;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
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

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = prefered.unwrap_or(abi);
                        }
                    }
                }

                closest_found_abi
            }
            Type::Double => {
                let bits = 64;
                let mut closest_found_abi = 0;
                let mut closest_found: u32 = 0;
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

                        if bits.abs_diff(size) <= closest_found.abs_diff(size) && bits >= size {
                            closest_found = size;
                            closest_found_abi = prefered.unwrap_or(abi);
                        }
                    }
                }

                closest_found_abi
            }
            Type::Fp128 | Type::PpcFp128 => 128,
            Type::X86Fp80 => 128,
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
            Type::Vector(vector_type) => todo!(),
            Type::Array(array_type) => todo!(),
            Type::Struct(struct_type) => todo!(),
            Type::Opaque(_) => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::types::Type;

    use super::DataLayout;

    #[test]
    fn test_get_type_size() {
        let datalayout = DataLayout::default();

        assert_eq!(datalayout.get_type_size(&Type::Int(1)), 8);
        assert_eq!(datalayout.get_type_size(&Type::Int(2)), 8);
        assert_eq!(datalayout.get_type_size(&Type::Int(4)), 8);
        assert_eq!(datalayout.get_type_size(&Type::Int(8)), 8);
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
        assert_eq!(datalayout.get_type_size(&Type::X86Fp80), 128);
        assert_eq!(datalayout.get_type_size(&Type::PpcFp128), 128);

        assert_eq!(datalayout.get_type_size(&Type::Ptr(None)), 64);
        assert_eq!(datalayout.get_type_size(&Type::Ptr(Some(2))), 64);
    }

    #[test]
    fn test_get_type_align() {
        let datalayout = DataLayout::default();

        assert_eq!(datalayout.get_type_align(&Type::Int(1)), 8);
        assert_eq!(datalayout.get_type_align(&Type::Int(2)), 8);
        assert_eq!(datalayout.get_type_align(&Type::Int(4)), 8);
        assert_eq!(datalayout.get_type_align(&Type::Int(8)), 8);
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
    }
}
