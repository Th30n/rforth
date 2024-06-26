use clap::Parser;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::io::Write;

/// Cell size in bytes.
pub const FORTH_CELL_SIZE: ForthUCell = 2; // 16 bit

/// Cell type which fits FORTH_CELL_SIZE.
pub type ForthCell = i16;

/// Unsigned cell type which fits FORTH_CELL_SIZE.
pub type ForthUCell = u16;

/// Double cell type.
pub type ForthDCell = i32;

pub fn unpack_forth_d_cell(d: ForthDCell) -> (ForthCell, ForthCell) {
    let b = (d >> (FORTH_CELL_SIZE * 8)) as ForthCell;
    let a = d as ForthCell;
    (a, b)
}

pub fn pack_forth_d_cell(a: ForthCell, b: ForthCell) -> ForthDCell {
    // Treat `a` as bits, so cast to ForthUCell before casting to ForthDCell to avoid sign
    // extension.
    ((b as ForthDCell) << (FORTH_CELL_SIZE * 8)) | a as ForthUCell as ForthDCell
}

/// Forth address into data space.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ForthPtr(ForthUCell);

impl ForthPtr {
    pub fn null() -> ForthPtr {
        ForthPtr(0)
    }

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn checked_add(self, offset: ForthUCell) -> Option<ForthPtr> {
        self.0.checked_add(offset).map(ForthPtr)
    }

    pub fn checked_sub(self, offset: ForthUCell) -> Option<ForthPtr> {
        self.0.checked_sub(offset).map(ForthPtr)
    }

    pub fn byte_offset(self, offset: isize) -> ForthPtr {
        ForthPtr((self.0 as isize + offset).try_into().unwrap())
    }

    pub fn byte_offset_from(self, origin: ForthPtr) -> isize {
        self.0 as isize - origin.0 as isize
    }

    pub fn align(self, alignment: ForthUCell) -> ForthPtr {
        ForthPtr(
            align_addr(self.0 as usize, alignment as usize)
                .try_into()
                .unwrap(),
        )
    }
}

impl std::fmt::Debug for ForthPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ForthPtr {{ {:#x } }}", self.0)
    }
}

fn is_aligned(addr: usize, alignment: usize) -> bool {
    addr % alignment == 0
}

fn align_addr(addr: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two());
    let aligned_addr = (addr + (alignment - 1)) & (!alignment + 1);
    assert!(is_aligned(aligned_addr, alignment));
    assert!(aligned_addr >= addr);
    aligned_addr
}

/// Return true if `val` is within [lower, upper).
fn is_within<T: Ord>(val: &T, lower: &T, upper: &T) -> bool {
    assert!(lower <= upper);
    lower <= val && val < upper
}

/// Raw memory buffer with stack based allocation scheme.
#[derive(Debug)]
pub struct Memory {
    layout: std::alloc::Layout,
    ptr: *mut u8,
    current: *mut u8,
}

impl Memory {
    pub fn with_size(bytes: usize) -> Self {
        assert!(bytes > 0);
        let layout = std::alloc::Layout::from_size_align(bytes, 8).unwrap();
        let ptr;
        unsafe {
            ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
        }
        Memory {
            layout,
            ptr,
            current: ptr,
        }
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn unused(&self) -> usize {
        let end = self.ptr as usize + self.size();
        end - self.current as usize
    }

    pub fn is_valid_ptr<T>(&self, ptr: *const T) -> bool {
        let end = self.current as usize - std::mem::size_of::<T>() + 1;
        is_aligned(ptr as usize, std::mem::align_of::<T>())
            && is_within(&(ptr as usize), &(self.ptr as usize), &end)
    }

    #[must_use]
    pub fn align(&mut self, alignment: usize) -> bool {
        let aligned_addr = align_addr(self.current as usize, alignment);
        let end = self.ptr as usize + self.size();
        if aligned_addr >= end {
            return false;
        }
        self.current = aligned_addr as *mut u8;
        true
    }

    // NOTE: Previously, `alloc` would return `Option<&mut [u8]>`. But this
    // causes UB when *reading* or *writing* any other previously returned
    // allocation. In other words, `&mut` provides exclusive access into `self`,
    // and thus returning `&mut [u8]` invalidates old pointers.
    //
    // https://doc.rust-lang.org/std/slice/fn.from_raw_parts_mut.html
    //
    //   The memory referenced by the returned slice must not be accessed
    //   through any other pointer (not derived from the return value) for the
    //   duration of lifetime 'a. Both read and write accesses are forbidden.
    //
    // Running the original code through MIRI would trigger an error
    //
    //   error: Undefined Behavior: trying to retag from <255807> for
    //          SharedReadOnly permission at alloc61163[0x529], but that tag
    //          does not exist in the borrow stack for this location
    //
    // And MIRI would point for further information here:
    // https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
    pub fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        if bytes > self.unused() {
            None
        } else {
            let ret = self.current;
            self.current = unsafe { self.current.add(bytes) };
            Some(ret)
        }
    }

    #[must_use]
    pub fn dealloc(&mut self, bytes: usize) -> bool {
        if bytes > self.current as usize - self.ptr as usize {
            false
        } else {
            self.current = unsafe { self.current.sub(bytes) };
            true
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

#[test]
fn test_memory_align() {
    const RUST_PTR_SIZE: usize = std::mem::size_of::<usize>();
    let mut memory = Memory::with_size(1024);
    assert_eq!(memory.unused(), 1024);
    assert!(memory.align(RUST_PTR_SIZE));
    assert_eq!(memory.unused(), 1024);
    memory.alloc(8);
    assert_eq!(memory.unused(), 1024 - 8);
    assert!(memory.align(RUST_PTR_SIZE));
    assert_eq!(memory.unused(), 1024 - 8);
    memory.alloc(1);
    assert_eq!(memory.unused(), 1024 - 9);
    assert!(memory.align(RUST_PTR_SIZE));
    assert_eq!(memory.unused(), 1024 - 16);
    memory.alloc(7);
    assert_eq!(memory.unused(), 1024 - 16 - 7);
    assert!(memory.align(RUST_PTR_SIZE));
    assert_eq!(memory.unused(), 1024 - 16 - 8);
}

// Mask for fitting dict entry's name length, this can fit up to 63 bytes.
// Note this mask is used in rforth.fs
const DICT_ENTRY_LEN_MASK: u8 = 0x3f;
// Flags must not overlap with the len mask.
#[repr(u8)]
enum WordFlag {
    Hidden = 0x40,
    Immediate = 0x80,
}

/// Entry in a Forth dictionary
///
/// Memory layout of a dict entry is the following.
///
/// | prev: ptr | flags|len: u8 | name: bytes | pad | definition...
///
/// Definition list consists of | codeword | definition addr...
///
/// The codeword is the native address to a function, so we call that directly.
/// It is either a `DOCOL_IX`, `DOCREATE_IX` or an index into `BUILTIN_WORDS`.
/// Remaining `definition addr` are pointers to the start of a `definition` in
/// some other dict entry, i.e. word.
#[derive(Copy, Clone)]
pub struct DictEntryRef<'a> {
    data_space: &'a DataSpace,
    /// Points to start of the entry in DataSpace.
    ptr: ForthPtr,
}

impl<'a> DictEntryRef<'a> {
    const FLAGS_AND_NAME_LEN_OFFSET: usize = FORTH_CELL_SIZE as usize;
    const NAME_OFFSET: usize = Self::FLAGS_AND_NAME_LEN_OFFSET + 1;

    pub fn prev(&self) -> Option<DictEntryRef<'a>> {
        let prev_ptr = ForthPtr(self.data_space.read_cell(self.ptr).unwrap() as ForthUCell);
        if prev_ptr.is_null() {
            None
        } else {
            // TODO: Make this assertion an actionable error, as it is possible
            // to overwrite the prev_addr to something invalid in Forth.
            assert!(self.data_space.is_valid_ptr(prev_ptr));
            Some(DictEntryRef {
                data_space: self.data_space,
                ptr: prev_ptr,
            })
        }
    }

    fn flags_and_name_len(&self) -> u8 {
        self.data_space
            .bytes_at(
                self.ptr
                    .checked_add(Self::FLAGS_AND_NAME_LEN_OFFSET as ForthUCell)
                    .unwrap(),
                1,
            )
            .unwrap()[0]
    }

    fn name_len(&self) -> u8 {
        self.flags_and_name_len() & DICT_ENTRY_LEN_MASK
    }

    pub fn flags(&self) -> u8 {
        self.flags_and_name_len() & !DICT_ENTRY_LEN_MASK
    }

    fn set_flags(&mut self, flags: u8) {
        assert_eq!(flags & DICT_ENTRY_LEN_MASK, 0);
        self.data_space
            .write_byte(
                self.ptr
                    .checked_add(Self::FLAGS_AND_NAME_LEN_OFFSET as ForthUCell)
                    .unwrap(),
                self.name_len() | flags,
            )
            .unwrap();
    }

    pub fn name(&self) -> &str {
        let len = self.name_len() as usize;
        let name_buf = self
            .data_space
            .bytes_at(
                self.ptr
                    .checked_add(Self::NAME_OFFSET as ForthUCell)
                    .unwrap(),
                len,
            )
            .unwrap();
        std::str::from_utf8(name_buf).unwrap()
    }

    pub fn definition_addr(&self) -> ForthPtr {
        self.ptr
            .byte_offset(Self::NAME_OFFSET as isize + self.name_len() as isize)
            .align(FORTH_CELL_SIZE as ForthUCell)
    }

    // Size of the entry up to the definition_addr
    fn header_size(&self) -> usize {
        self.definition_addr().byte_offset_from(self.ptr) as usize
    }

    // Return true if now hidden.
    pub fn toggle_hidden(&mut self) -> bool {
        let new_flags = self.flags() ^ WordFlag::Hidden as u8;
        self.set_flags(new_flags);
        self.is_hidden()
    }

    pub fn is_hidden(&self) -> bool {
        (self.flags() & WordFlag::Hidden as u8) != 0
    }

    // Return true if now immediate.
    pub fn toggle_immediate(&mut self) -> bool {
        let new_flags = self.flags() ^ WordFlag::Immediate as u8;
        self.set_flags(new_flags);
        self.is_immediate()
    }

    pub fn is_immediate(&self) -> bool {
        (self.flags() & WordFlag::Immediate as u8) != 0
    }
}

impl std::fmt::Debug for DictEntryRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DictEntryRef {{ ptr: {:?}, end_of_data_space: {:?}, prev: {:?}, name: {}, flags: {:#x}, definition_addr: {:?} }}",
            self.ptr,
            self.data_space.end(),
            self.prev().map_or(ForthPtr::null(), |e| e.ptr),
            self.name(),
            self.flags(),
            self.definition_addr(),
        )
    }
}

/// Memory for Forth's data space.
///
/// Data space includes dictionary entries and user allocations.
#[derive(Debug)]
pub struct DataSpace {
    memory: Memory,
    dict_head: ForthPtr,
}

impl DataSpace {
    pub fn with_size(bytes: usize) -> Self {
        assert!(
            bytes <= (ForthUCell::MAX - FORTH_CELL_SIZE) as usize,
            "trying to allocate bytes the ForthPtr cannot address"
        );
        let memory = Memory::with_size(bytes);
        assert!(is_aligned(
            memory.current as usize,
            FORTH_CELL_SIZE as usize
        ));
        DataSpace {
            memory,
            dict_head: ForthPtr::null(),
        }
    }

    fn memory_ptr_to_forth_ptr(&self, ptr: *mut u8) -> ForthPtr {
        if ptr.is_null() {
            ForthPtr::null()
        } else {
            let offset = unsafe { ptr.offset_from(self.memory.ptr) };
            assert!(offset >= 0);
            // Additionally offset by FORTH_CELL_SIZE, so that 0 can be used as null.
            ForthPtr(
                offset
                    .checked_add(FORTH_CELL_SIZE as isize)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            )
        }
    }

    fn forth_ptr_to_memory_ptr(&self, forth_ptr: ForthPtr) -> *mut u8 {
        if forth_ptr.is_null() {
            std::ptr::null_mut()
        } else {
            let offset = (forth_ptr.0 as isize)
                .checked_sub(FORTH_CELL_SIZE as isize)
                .unwrap();
            assert!(offset >= 0);
            unsafe { self.memory.ptr.offset(offset) }
        }
    }

    pub fn bytes_at(&self, ptr: ForthPtr, len: usize) -> Result<&[u8], ForthError> {
        assert!(len > 0);
        if !self.is_valid_ptr(ptr) || !self.is_valid_ptr(ptr.byte_offset(len as isize - 1)) {
            return Err(ForthError::InvalidPointer(ptr));
        }
        let ptr = self.forth_ptr_to_memory_ptr(ptr) as *const u8;
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    pub fn bytes_mut_at(&self, ptr: ForthPtr, len: usize) -> Result<&mut [u8], ForthError> {
        assert!(len > 0);
        if !self.is_valid_ptr(ptr) || !self.is_valid_ptr(ptr.byte_offset(len as isize - 1)) {
            return Err(ForthError::InvalidPointer(ptr));
        }
        let ptr = self.forth_ptr_to_memory_ptr(ptr);
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    pub fn write_bytes(&self, ptr: ForthPtr, bytes: &[u8]) -> Result<(), ForthError> {
        let buf = self.bytes_mut_at(ptr, bytes.len())?;
        let mut cursor = std::io::Cursor::new(buf);
        cursor.write_all(bytes).unwrap();
        Ok(())
    }

    pub fn write_byte(&self, ptr: ForthPtr, val: u8) -> Result<(), ForthError> {
        self.bytes_mut_at(ptr, 1)?[0] = val;
        Ok(())
    }

    pub fn read_cell(&self, ptr: ForthPtr) -> Result<ForthCell, ForthError> {
        let bytes = <[u8; FORTH_CELL_SIZE as usize]>::try_from(
            self.bytes_at(ptr, FORTH_CELL_SIZE as usize)?,
        )
        .unwrap();
        Ok(ForthCell::from_ne_bytes(bytes))
    }

    pub fn write_cell(&self, ptr: ForthPtr, val: ForthCell) -> Result<(), ForthError> {
        self.write_bytes(ptr, &val.to_ne_bytes())
    }

    pub fn size(&self) -> usize {
        self.memory.size()
    }

    /// Past the end address of DataSpace
    pub fn end(&self) -> ForthPtr {
        self.memory_ptr_to_forth_ptr(self.memory.ptr)
            .byte_offset(self.size() as isize)
    }

    pub fn here(&self) -> ForthPtr {
        self.memory_ptr_to_forth_ptr(self.memory.current)
    }

    pub fn unused(&self) -> usize {
        self.memory.unused()
    }

    pub fn is_valid_ptr(&self, ptr: ForthPtr) -> bool {
        let ptr = self.forth_ptr_to_memory_ptr(ptr);
        self.memory.is_valid_ptr(ptr)
    }

    #[must_use]
    pub fn align(&mut self) -> bool {
        self.memory.align(FORTH_CELL_SIZE as usize)
    }

    pub fn alloc(&mut self, bytes: usize) -> Option<ForthPtr> {
        self.memory
            .alloc(bytes)
            .map(|p| self.memory_ptr_to_forth_ptr(p))
    }

    #[must_use]
    pub fn dealloc(&mut self, bytes: usize) -> bool {
        assert!(bytes > 0);
        let new_here = self.here().byte_offset(-(bytes as isize));
        let mut new_dict_head = ForthPtr::null();
        let mut maybe_entry = self.latest_entry();
        while let Some(entry) = maybe_entry {
            if entry.ptr.byte_offset(entry.header_size() as isize) <= new_here {
                new_dict_head = entry.ptr;
                break;
            } else {
                maybe_entry = entry.prev();
            }
        }
        let succ = self.memory.dealloc(bytes);
        if succ {
            assert_eq!(self.here(), new_here);
            self.dict_head = new_dict_head;
        }
        assert!(self.dict_head.is_null() || self.is_valid_ptr(self.dict_head));
        succ
    }

    /// Allocate a new entry in the dictionary with given `name`.
    /// The entry is padded, so that the next allocation is aligned.
    #[must_use]
    pub fn push_dict_entry(&mut self, name: &str, flags: u8) -> bool {
        assert_eq!(flags & DICT_ENTRY_LEN_MASK, 0);
        assert!(name.len() <= DICT_ENTRY_LEN_MASK as usize);
        let prev_addr = self.dict_head;
        let entry_size = DictEntryRef::NAME_OFFSET + name.len();
        match self.alloc(entry_size) {
            None => false,
            Some(entry_ptr) => {
                let buf = self.bytes_mut_at(entry_ptr, entry_size).unwrap();
                let mut cursor = std::io::Cursor::new(buf);
                cursor.write_all(&prev_addr.0.to_ne_bytes()).unwrap();
                let flags_and_name_len = flags | (name.len() as u8);
                cursor.write_all(&[flags_and_name_len]).unwrap();
                cursor.write_all(name.as_bytes()).unwrap();
                if !self.align() {
                    assert!(self.dealloc(entry_size));
                    false
                } else {
                    self.dict_head = entry_ptr;
                    true
                }
            }
        }
    }

    pub fn latest_entry(&self) -> Option<DictEntryRef<'_>> {
        if self.dict_head.is_null() {
            None
        } else {
            assert!(self.is_valid_ptr(self.dict_head));
            Some(DictEntryRef {
                data_space: self,
                ptr: self.dict_head,
            })
        }
    }

    pub fn find_entry(&self, name: &str) -> Option<DictEntryRef<'_>> {
        let mut maybe_entry = self.latest_entry();
        while let Some(entry) = maybe_entry {
            if !entry.is_hidden() && entry.name() == name {
                return maybe_entry;
            } else {
                maybe_entry = entry.prev();
            }
        }
        None
    }

    pub fn find_entry_by_def_addr(&self, def_addr: ForthPtr) -> Option<DictEntryRef<'_>> {
        let mut maybe_entry = self.latest_entry();
        while let Some(entry) = maybe_entry {
            if entry.definition_addr() == def_addr {
                return maybe_entry;
            } else {
                maybe_entry = entry.prev();
            }
        }
        None
    }

    pub fn push_builtin_word(&mut self, name: &str, flags: u8, word_ix: usize) {
        assert!(self.push_dict_entry(name, flags));
        let word_ix: ForthUCell = (word_ix + SPECIAL_CODEWORDS.len()).try_into().unwrap();
        let ptr = self.alloc(FORTH_CELL_SIZE as usize).unwrap();
        let buf = self.bytes_mut_at(ptr, FORTH_CELL_SIZE as usize).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        cursor.write_all(&word_ix.to_ne_bytes()).unwrap();
    }
}

#[test]
fn test_data_space_find_entry() {
    let mut data_space = DataSpace::with_size(1024);
    assert!(data_space.find_entry("DUP").is_none());
    assert!(data_space.push_dict_entry("DUP", 0));
    {
        let dup = data_space.find_entry("DUP").unwrap();
        assert_eq!(dup.name(), "DUP");
        let latest = data_space.latest_entry().unwrap();
        assert_eq!(dup.ptr, latest.ptr);
    }
    assert!(data_space.push_dict_entry("SWAP", 0));
    {
        let dup = data_space.find_entry("DUP").unwrap();
        assert_eq!(dup.name(), "DUP");
        let latest = data_space.latest_entry().unwrap();
        assert_ne!(dup.ptr, latest.ptr);
        assert_eq!(dup.ptr, latest.prev().unwrap().ptr);
    }
    assert!(data_space.push_dict_entry("DUP", 0));
    {
        let dup = data_space.find_entry("DUP").unwrap();
        assert_eq!(dup.name(), "DUP");
        let latest = data_space.latest_entry().unwrap();
        assert_eq!(dup.ptr, latest.ptr);
        let orig_dup = data_space
            .latest_entry()
            .and_then(|e| e.prev())
            .and_then(|e| e.prev())
            .unwrap();
        assert_eq!(dup.name(), orig_dup.name());
        assert_ne!(dup.ptr, orig_dup.ptr);
        let dup_header_size = dup.header_size();
        let orig_dup_ptr = orig_dup.ptr;
        assert!(data_space.dealloc(dup_header_size));
        assert_eq!(data_space.find_entry("DUP").unwrap().ptr, orig_dup_ptr)
    }
}

const FORTH_TRUE: ForthCell = -1;
const FORTH_FALSE: ForthCell = 0;
const INPUT_BUFFER_SIZE: usize = 256;

#[derive(Debug)]
struct StackImpl {
    start: ForthPtr,
    end: ForthPtr,
    max_elements: usize,
}

impl StackImpl {
    fn alloc(data_space: &mut DataSpace, max_elements: usize) -> StackImpl {
        assert!(max_elements > 0);
        assert!(data_space.align());
        let start = data_space
            .alloc(max_elements * FORTH_CELL_SIZE as usize)
            .unwrap();
        Self {
            start,
            end: start,
            max_elements,
        }
    }
}

pub struct Stack<'a> {
    data_space: &'a DataSpace,
    stack: &'a mut StackImpl,
}

impl<'a> Stack<'a> {
    pub fn len(&self) -> usize {
        assert!(self.data_space.is_valid_ptr(self.stack.start));
        let byte_offset = self.stack.end.byte_offset_from(self.stack.start);
        assert_eq!(byte_offset % FORTH_CELL_SIZE as isize, 0);
        let offset = byte_offset / FORTH_CELL_SIZE as isize;
        assert!(offset >= 0);
        offset as usize
    }
    pub fn is_empty(&self) -> bool {
        self.stack.end == self.stack.start
    }
    pub fn clear(&mut self) {
        self.stack.end = self.stack.start
    }
    #[must_use]
    pub fn push(&mut self, val: ForthCell) -> Option<()> {
        if self.len() < self.stack.max_elements {
            self.data_space.write_cell(self.stack.end, val).unwrap();
            self.stack.end = self.stack.end.checked_add(FORTH_CELL_SIZE).unwrap();
            Some(())
        } else {
            None
        }
    }
    #[must_use]
    pub fn pop(&mut self) -> Option<ForthCell> {
        if self.is_empty() {
            None
        } else {
            let last_ptr = self.stack.end.checked_sub(FORTH_CELL_SIZE).unwrap();
            let val = self.data_space.read_cell(last_ptr).unwrap();
            self.stack.end = last_ptr;
            Some(val)
        }
    }
    pub fn last(&mut self) -> Option<ForthCell> {
        if self.is_empty() {
            None
        } else {
            let last_ptr = self.stack.end.checked_sub(FORTH_CELL_SIZE).unwrap();
            let val = self.data_space.read_cell(last_ptr).unwrap();
            Some(val)
        }
    }
    pub fn iter(&self) -> StackIterator {
        StackIterator::new(self)
    }
}

pub struct StackIterator<'a> {
    stack: &'a Stack<'a>,
    current: ForthPtr,
}
impl<'a> StackIterator<'a> {
    fn new(stack: &'a Stack<'a>) -> Self {
        Self {
            stack,
            current: stack.stack.start,
        }
    }
}
impl Iterator for StackIterator<'_> {
    type Item = ForthCell;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.stack.stack.end {
            None
        } else {
            let val = self.stack.data_space.read_cell(self.current).unwrap();
            self.current = self.current.checked_add(FORTH_CELL_SIZE).unwrap();
            Some(val)
        }
    }
}
impl<'a> IntoIterator for &'a Stack<'a> {
    type Item = ForthCell;
    type IntoIter = StackIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        StackIterator::new(self)
    }
}

impl std::fmt::Debug for Stack<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut elements = Vec::with_capacity(self.len());
        let mut ptr = self.stack.start;
        while self.data_space.is_valid_ptr(ptr) && ptr < self.stack.end {
            elements.push(self.data_space.read_cell(ptr).unwrap());
            ptr = ptr.checked_add(FORTH_CELL_SIZE).unwrap();
        }
        write!(f, "Stack {{ {:?}, elements: {:?} }}", self.stack, elements,)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ForthState {
    Immediate = 0,
    Compile = 1,
}

#[derive(Debug)]
pub enum ForthError {
    DataStackEmpty,
    DataStackLimitReached,
    ReturnStackEmpty,
    ReturnStackLimitReached,
    InputBufferLimitReached,
    AllocFailed,
    DeallocFailed,
    InvalidPointer(ForthPtr),
    AddressOverflow(ForthPtr),
    NoLatestEntry,
    CreateMissingWord,
    InvalidStringEncoding,
    ParseNum(String),
    DivByZero,
}

#[derive(Debug)]
pub struct ForthMachine {
    pub data_space: DataSpace,
    data_stack: StackImpl,
    return_stack: StackImpl,
    curr_def_addr: ForthPtr,
    instruction_addr: ForthPtr,
    curr_input_ix_ptr: ForthPtr,
    curr_input_len: ForthUCell,
    // Stdin line parse area inside `data_space`; length is INPUT_BUFFER_SIZE.
    input_buffer_ptr: ForthPtr,
    state: ForthState,
    // Backtrace of word definition_addr when error happens.
    backtrace: Vec<ForthPtr>,
}

impl ForthMachine {
    pub fn new(
        mut data_space: DataSpace,
        data_stack_size: usize,
        return_stack_size: usize,
    ) -> Self {
        let curr_input_ix_ptr = push_variable(&mut data_space, ">IN", 0);
        assert!(ForthUCell::MAX as usize >= INPUT_BUFFER_SIZE);
        let input_buffer_ptr = data_space.alloc(INPUT_BUFFER_SIZE).unwrap();
        assert!(data_space.align());
        let data_stack = StackImpl::alloc(&mut data_space, data_stack_size);
        let return_stack = StackImpl::alloc(&mut data_space, return_stack_size);
        assert!(data_space.align());
        for (ix, (name, flags, _fun)) in BUILTIN_WORDS.iter().enumerate() {
            data_space.push_builtin_word(name, *flags, ix);
        }
        Self {
            data_space,
            data_stack,
            return_stack,
            instruction_addr: ForthPtr::null(),
            curr_def_addr: ForthPtr::null(),
            curr_input_ix_ptr,
            curr_input_len: 0,
            input_buffer_ptr,
            state: ForthState::Immediate,
            backtrace: vec![],
        }
    }

    fn curr_input_ix(&self) -> Result<ForthUCell, ForthError> {
        self.data_space
            .read_cell(self.curr_input_ix_ptr)
            .map(|v| v as ForthUCell)
    }
    fn set_curr_input_ix(&mut self, ix: ForthUCell) -> Result<(), ForthError> {
        self.data_space
            .write_cell(self.curr_input_ix_ptr, ix as ForthCell)
    }
    fn input_buffer(&mut self) -> &mut [u8] {
        self.data_space
            .bytes_mut_at(self.input_buffer_ptr, INPUT_BUFFER_SIZE)
            .unwrap()
    }

    pub fn data_stack(&mut self) -> Stack {
        Stack {
            data_space: &self.data_space,
            stack: &mut self.data_stack,
        }
    }

    pub fn return_stack(&mut self) -> Stack {
        Stack {
            data_space: &self.data_space,
            stack: &mut self.return_stack,
        }
    }

    pub fn bye(&mut self) {
        bye_builtin(self).unwrap();
    }

    pub fn push_variable(&mut self, name: &str, val: ForthCell) -> ForthPtr {
        push_variable(&mut self.data_space, name, val)
    }
    pub fn push_buffer(&mut self, name: &str, bytes: usize) -> ForthPtr {
        push_buffer(&mut self.data_space, name, bytes)
    }
}

fn ds_pop(forth: &mut ForthMachine) -> Result<ForthCell, ForthError> {
    forth.data_stack().pop().ok_or(ForthError::DataStackEmpty)
}
fn ds_push(forth: &mut ForthMachine, val: ForthCell) -> Result<(), ForthError> {
    forth
        .data_stack()
        .push(val)
        .ok_or(ForthError::DataStackLimitReached)
}
fn ds_last(forth: &mut ForthMachine) -> Result<ForthCell, ForthError> {
    forth.data_stack().last().ok_or(ForthError::DataStackEmpty)
}

fn rs_pop(forth: &mut ForthMachine) -> Result<ForthCell, ForthError> {
    forth
        .return_stack()
        .pop()
        .ok_or(ForthError::ReturnStackEmpty)
}
fn rs_push(forth: &mut ForthMachine, val: ForthCell) -> Result<(), ForthError> {
    forth
        .return_stack()
        .push(val)
        .ok_or(ForthError::ReturnStackLimitReached)
}
fn rs_last(forth: &mut ForthMachine) -> Result<ForthCell, ForthError> {
    forth
        .return_stack()
        .last()
        .ok_or(ForthError::ReturnStackEmpty)
}

fn bye_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.instruction_addr = ForthPtr::null();
    Ok(())
}

fn drop_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    ds_pop(forth)?;
    Ok(())
}

fn nip_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let val = ds_pop(forth)?;
    ds_pop(forth)?;
    ds_push(forth, val)
}

fn dup_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let val = ds_last(forth)?;
    ds_push(forth, val)
}

fn swap_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let a = ds_pop(forth)?;
    let b = ds_pop(forth)?;
    ds_push(forth, a)?;
    ds_push(forth, b)
}

fn rot_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let a = ds_pop(forth)?;
    let b = ds_pop(forth)?;
    let c = ds_pop(forth)?;
    ds_push(forth, b)?;
    ds_push(forth, a)?;
    ds_push(forth, c)
}

// DOCOL is special, it is a codeword and as such its direct address should be
// used when creating new words (via DOCOL_IX).
// TODO: Maybe make a DOCOL word (constant) that pushes DOCOL_IX on stack.
fn docol(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.backtrace.push(forth.curr_def_addr);
    let ret_addr = forth.instruction_addr.0 as ForthCell;
    rs_push(forth, ret_addr)?;
    forth.instruction_addr = forth
        .curr_def_addr
        .checked_add(FORTH_CELL_SIZE)
        .ok_or(ForthError::AddressOverflow(forth.curr_def_addr))?;
    Ok(())
}

fn exit_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.instruction_addr = ForthPtr(rs_pop(forth)? as ForthUCell);
    forth.backtrace.pop();
    Ok(())
}

fn refill_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let mut line = String::with_capacity(INPUT_BUFFER_SIZE);
    if let Ok(num_read) = std::io::stdin().read_line(&mut line) {
        if num_read > INPUT_BUFFER_SIZE {
            return Err(ForthError::InputBufferLimitReached);
        }
        assert!(num_read <= INPUT_BUFFER_SIZE);
        if num_read > 0 {
            assert_eq!(
                forth.input_buffer().write(line.as_bytes()).unwrap(),
                num_read
            );
            forth.curr_input_len = num_read
                .try_into()
                .expect("num_read should fit into ForthUCell because INPUT_BUFFER_SIZE fits.");
            forth.set_curr_input_ix(0)?;
            return ds_push(forth, FORTH_TRUE);
        }
    }
    ds_push(forth, FORTH_FALSE)
}

fn source_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = forth.input_buffer_ptr;
    ds_push(forth, ptr.0 as ForthCell)?;
    let len = forth.curr_input_len;
    ds_push(forth, len as ForthCell)
}

fn read_input_byte(forth: &mut ForthMachine) -> Option<u8> {
    let curr_ix = forth.curr_input_ix().unwrap();
    if curr_ix >= forth.curr_input_len {
        None
    } else {
        assert!(curr_ix < forth.curr_input_len);
        assert!(forth.curr_input_len as usize <= INPUT_BUFFER_SIZE);
        let byte = forth.input_buffer()[curr_ix as usize];
        forth.set_curr_input_ix(curr_ix + 1).unwrap();
        Some(byte)
    }
}

fn emit_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let v = ds_pop(forth)? as u8;
    assert_eq!(std::io::stdout().write(&[v]).unwrap(), 1);
    Ok(())
}

// ( "ccc<eol>" -- )
// Parses and discards \ comments
fn backslash_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.set_curr_input_ix(forth.curr_input_len)
}

const BLANK_CHARS: [char; 3] = [' ', '\t', '\n'];

// ( "<spaces>name<space>" -- c-addr u )
//
// Skip leading white space and parse `name` delimited by a white space.
// NOTE: Behaves like standard PARSE-NAME, not like WORD.
fn word_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let (ptr, len) = match do_word_builtin(forth) {
        None => (
            forth
                .input_buffer_ptr
                .checked_add(INPUT_BUFFER_SIZE as ForthUCell)
                .unwrap(),
            0,
        ),
        Some((ptr, len)) => (ptr, len),
    };
    ds_push(forth, ptr.0 as ForthCell)?;
    ds_push(forth, len as ForthCell)
}
fn do_word_builtin(forth: &mut ForthMachine) -> Option<(ForthPtr, ForthUCell)> {
    let mut byte = read_input_byte(forth)?;
    // Skip blanks
    while BLANK_CHARS.contains(&byte.into()) {
        byte = read_input_byte(forth)?;
    }
    let curr_ix = forth.curr_input_ix().unwrap();
    assert!(curr_ix > 0);
    assert!(curr_ix as usize <= INPUT_BUFFER_SIZE);
    let start_ptr = forth.input_buffer_ptr.checked_add(curr_ix - 1).unwrap();
    let mut len = 0;
    while !BLANK_CHARS.contains(&byte.into()) {
        len += 1;
        match read_input_byte(forth) {
            None => break,
            Some(b) => byte = b,
        }
    }
    assert!(forth.curr_input_ix().unwrap() as usize <= INPUT_BUFFER_SIZE);
    assert!(len > 0);
    Some((start_ptr, len.try_into().unwrap()))
}

fn store_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let val = ds_pop(forth)?;
    forth.data_space.write_cell(ptr, val)
}
fn store_byte_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let val = ds_pop(forth)? as u8;
    forth.data_space.write_byte(ptr, val)
}

fn fetch_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let val = forth.data_space.read_cell(ptr)?;
    ds_push(forth, val)
}
fn fetch_byte_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let val = forth.data_space.bytes_at(ptr, 1)?[0];
    ds_push(forth, val as ForthCell)
}

fn parse_num(src: &str, radix: u32) -> Result<ForthCell, std::num::ParseIntError> {
    if let Some(src) = src.strip_prefix('#') {
        ForthCell::from_str_radix(src, 10)
    } else if let Some(src) = src.strip_prefix('$') {
        ForthCell::from_str_radix(src, 16)
    } else if let Some(src) = src.strip_prefix('%') {
        ForthCell::from_str_radix(src, 2)
    } else if src.starts_with('\'') && src.ends_with('\'') && src.chars().count() == 3 {
        Ok(src.chars().nth(1).unwrap() as ForthCell)
    } else {
        ForthCell::from_str_radix(src, radix)
    }
}

// Converts string to number, flag indicates success.
// (addr u - n f)
// NOTE: Based on Gforth's S>NUMBER?, but we return a single cell number.
fn s_to_number_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let byte_len = ds_pop(forth)? as ForthUCell;
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    if byte_len == 0 {
        ds_push(forth, 0)?;
        return ds_push(forth, FORTH_FALSE);
    }
    let bytes = forth.data_space.bytes_at(ptr, byte_len as usize)?;
    let name = std::str::from_utf8(bytes).map_err(|_| ForthError::InvalidStringEncoding)?;
    // TODO: Use BASE variable for parsing.
    let base = 10;
    match parse_num(name, base) {
        Ok(num) => {
            ds_push(forth, num)?;
            ds_push(forth, FORTH_TRUE)
        }
        _ => {
            ds_push(forth, 0)?;
            ds_push(forth, FORTH_FALSE)
        }
    }
}

// Return name token (start of dict entry) or 0.
// (addr u - nt | 0)
fn find_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let byte_len = ds_pop(forth)? as ForthUCell as usize;
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let bytes = forth.data_space.bytes_at(ptr, byte_len)?;
    let name = std::str::from_utf8(bytes).map_err(|_| ForthError::InvalidStringEncoding)?;
    match forth.data_space.find_entry(name) {
        None => ds_push(forth, 0),
        Some(dict_entry_ref) => {
            let dict_entry_addr = dict_entry_ref.ptr.0 as ForthCell;
            ds_push(forth, dict_entry_addr)
        }
    }
}

fn latest_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let dict_head_addr = forth.data_space.dict_head.0 as ForthCell;
    ds_push(forth, dict_head_addr)
}

// Return Code Field Address (i.e. definition address of a dict entry).
fn to_cfa_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let dict_entry_ref = DictEntryRef {
        data_space: &forth.data_space,
        ptr,
    };
    let def_addr = dict_entry_ref.definition_addr().0 as ForthCell;
    ds_push(forth, def_addr)
}

fn hidden_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr = ForthPtr(ds_pop(forth)? as ForthUCell);
    let mut dict_entry_ref = DictEntryRef {
        data_space: &forth.data_space,
        ptr,
    };
    dict_entry_ref.toggle_hidden();
    Ok(())
}

fn immediate_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth
        .data_space
        .latest_entry()
        .ok_or(ForthError::NoLatestEntry)?
        .toggle_immediate();
    Ok(())
}

fn branch_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr_to_offset = forth.instruction_addr;
    let offset = forth.data_space.read_cell(ptr_to_offset)?;
    if offset >= 0 {
        forth.instruction_addr = forth
            .instruction_addr
            .checked_add(offset as ForthUCell)
            .ok_or(ForthError::AddressOverflow(forth.instruction_addr))?;
        Ok(())
    } else {
        forth.instruction_addr = forth
            .instruction_addr
            .checked_sub(offset.unsigned_abs())
            .ok_or(ForthError::AddressOverflow(forth.instruction_addr))?;
        Ok(())
    }
}

fn zbranch_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let val = ds_pop(forth)?;
    if val == 0 {
        branch_builtin(forth)
    } else {
        forth.instruction_addr = forth
            .instruction_addr
            .checked_add(FORTH_CELL_SIZE)
            .ok_or(ForthError::AddressOverflow(forth.instruction_addr))?;
        Ok(())
    }
}

fn lit_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ptr_to_val = forth.instruction_addr;
    let val = forth.data_space.read_cell(ptr_to_val)?;
    // println!("val: {:#x}", val);
    forth.instruction_addr = forth
        .instruction_addr
        .checked_add(FORTH_CELL_SIZE)
        .ok_or(ForthError::AddressOverflow(forth.instruction_addr))?;
    ds_push(forth, val)
}

fn allot_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let bytes = ds_pop(forth)?;
    if bytes > 0 {
        forth
            .data_space
            .alloc(bytes as usize)
            .ok_or(ForthError::AllocFailed)?;
    } else if bytes < 0 && !forth.data_space.dealloc(bytes.unsigned_abs() as usize) {
        return Err(ForthError::DeallocFailed);
    }
    Ok(())
}

// `CREATE` is a bit special, it will allocate a dict entry, but will set the
// codeword to be `docreate` and append a `ptr` for runtime semantics of the
// newly created word. This ptr is set to 0 initially, but can be overwritten
// with `DOES>`. What follows the allocated dict entry is the space for data
// fields. This address can be obtained via `>BODY`.
// NOTE: `DOES>` & `>BODY` are implemented in rforth.fs
fn create_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let (ptr, byte_len) = do_word_builtin(forth).ok_or(ForthError::CreateMissingWord)?;
    // TODO: Figure out a way to avoid `to_vec`.
    let bytes = forth.data_space.bytes_at(ptr, byte_len as usize)?.to_vec();
    let name = std::str::from_utf8(&bytes).map_err(|_| ForthError::InvalidStringEncoding)?;
    // println!("Pushing '{}' to dict", name);
    // TODO: These should have AllocFailed errors
    assert!(forth.data_space.push_dict_entry(name, 0));
    push_instruction(&mut forth.data_space, DOCREATE_IX as ForthUCell);
    push_instruction(&mut forth.data_space, 0);
    Ok(())
}

// DOCREATE is a special codeword, it will push the address of the start of the
// dict entry's data fields on the data stack. If the `ptr` after the codeword
// is not 0, it will then set the next instruction to be found at that `ptr`.
// This way we implement run-time semantics of a CREATEd word that has DOES>.
fn docreate(forth: &mut ForthMachine) -> Result<(), ForthError> {
    // Sanity check that forth.curr_def_addr is indeed the dict entry we are
    // executing in.
    let fun_addr = forth.data_space.read_cell(forth.curr_def_addr)? as ForthUCell as usize;
    assert_eq!(fun_addr, DOCREATE_IX);

    let instr_addr_ptr = forth
        .curr_def_addr
        .checked_add(FORTH_CELL_SIZE)
        .ok_or(ForthError::AddressOverflow(forth.curr_def_addr))?;
    let instr_addr = ForthPtr(forth.data_space.read_cell(instr_addr_ptr)? as ForthUCell);
    let data_addr = instr_addr_ptr
        .checked_add(FORTH_CELL_SIZE)
        .ok_or(ForthError::AddressOverflow(instr_addr_ptr))?;
    ds_push(forth, data_addr.0 as ForthCell)?;
    if !instr_addr.is_null() {
        forth.backtrace.push(forth.curr_def_addr);
        let ret_addr = forth.instruction_addr.0 as ForthCell;
        rs_push(forth, ret_addr)?;
        forth.instruction_addr = instr_addr;
    }
    Ok(())
}

fn comma_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let def_addr = ds_pop(forth)?;
    push_instruction(&mut forth.data_space, def_addr as ForthUCell);
    Ok(())
}

fn lbrac_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.state = ForthState::Immediate;
    Ok(())
}

fn rbrac_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.state = ForthState::Compile;
    Ok(())
}

// Interpret a *single* word in the parse area (if available).
fn interpret_single_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let (ptr, byte_len) = match do_word_builtin(forth) {
        None => return Ok(()),
        Some((ptr, byte_len)) => (ptr, byte_len),
    };
    let bytes = forth.data_space.bytes_at(ptr, byte_len as usize)?;
    let name = std::str::from_utf8(bytes).map_err(|_| ForthError::InvalidStringEncoding)?;
    if let Some(dict_entry_ref) = forth.data_space.find_entry(name) {
        // dbg!(dict_entry_ref);
        if dict_entry_ref.is_immediate() || forth.state == ForthState::Immediate {
            exec_fun_indirect(dict_entry_ref.definition_addr(), forth)
        } else {
            // Compiling
            let def_addr = dict_entry_ref.definition_addr();
            push_instruction(&mut forth.data_space, def_addr.0);
            Ok(())
        }
    } else {
        // TODO: Use BASE variable for parsing.
        let base = 10;
        let num = parse_num(name, base).map_err(|_| ForthError::ParseNum(name.to_string()))?;
        match forth.state {
            ForthState::Immediate => ds_push(forth, num),
            ForthState::Compile => {
                let lit = forth
                    .data_space
                    .find_entry("LIT")
                    .unwrap()
                    .definition_addr();
                push_instruction(&mut forth.data_space, lit.0);
                push_instruction(&mut forth.data_space, num as ForthUCell);
                Ok(())
            }
        }
    }
}

fn rs_clear_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.return_stack().clear();
    Ok(())
}

fn execute_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let def_addr = ForthPtr(ds_pop(forth)? as ForthUCell);
    exec_fun_indirect(def_addr, forth)
}

fn here_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let here_addr = forth.data_space.here();
    ds_push(forth, here_addr.0 as ForthCell)
}

fn align_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    if !forth.data_space.align() {
        return Err(ForthError::AllocFailed);
    }
    Ok(())
}

fn unused_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let unused = forth.data_space.unused() as ForthCell;
    ds_push(forth, unused)
}

fn cells_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let n = ds_pop(forth)?;
    ds_push(forth, n.wrapping_mul(FORTH_CELL_SIZE as ForthCell))
}

fn add_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    ds_push(forth, a.wrapping_add(b))
}

fn sub_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    ds_push(forth, a.wrapping_sub(b))
}

fn mul_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    ds_push(forth, a.wrapping_mul(b))
}

// ( n1 n2 -- d )
fn m_mul_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let n2 = ds_pop(forth)? as ForthDCell;
    let n1 = ds_pop(forth)? as ForthDCell;
    let d = n1 * n2;
    let (a, b) = unpack_forth_d_cell(d);
    ds_push(forth, a)?;
    ds_push(forth, b)
}

// ( d1 n1 -- n2 n3 )
//
// d1 = n3 * n1 + n2 n3 is the symmetric quotient, n2 is the remainder.
// This is the same behavior as /MOD (slash_mod_builtin).
fn sm_slash_rem_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let n1 = ds_pop(forth)? as ForthDCell;
    let d1_b = ds_pop(forth)?;
    let d1_a = ds_pop(forth)?;
    let d1 = pack_forth_d_cell(d1_a, d1_b);
    ds_push(forth, (d1 % n1) as ForthCell)?;
    ds_push(forth, (d1 / n1) as ForthCell)
}

fn slash_mod_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    if b == 0 {
        return Err(ForthError::DivByZero);
    }
    ds_push(forth, a % b)?;
    ds_push(forth, a / b)
}

fn s_to_d_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let a = ds_last(forth)?;
    let b = if a < 0 { -1 } else { 0 };
    ds_push(forth, b)
}

fn abs_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let a = ds_pop(forth)?;
    ds_push(forth, a.unsigned_abs() as ForthCell)
}

fn invert_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let a = ds_pop(forth)?;
    ds_push(forth, !a)
}

fn and_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, (a & b) as ForthCell)
}

fn or_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, (a | b) as ForthCell)
}

fn xor_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, (a ^ b) as ForthCell)
}

fn lshift_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, (a << b) as ForthCell)
}

fn rshift_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, (a >> b) as ForthCell)
}

fn eq_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    ds_push(forth, if a == b { FORTH_TRUE } else { FORTH_FALSE })
}

fn greater_than_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)?;
    let a = ds_pop(forth)?;
    ds_push(forth, if a > b { FORTH_TRUE } else { FORTH_FALSE })
}

fn u_greater_than_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let b = ds_pop(forth)? as ForthUCell;
    let a = ds_pop(forth)? as ForthUCell;
    ds_push(forth, if a > b { FORTH_TRUE } else { FORTH_FALSE })
}

fn to_rstack_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ret_addr = ds_pop(forth)?;
    rs_push(forth, ret_addr)
}

fn from_rstack_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ret_addr = rs_pop(forth)?;
    ds_push(forth, ret_addr)
}

fn rstack_fetch_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let ret_addr = rs_last(forth)?;
    ds_push(forth, ret_addr)
}

fn print_data_stack_builtin(forth: &mut ForthMachine) -> Result<(), ForthError> {
    let data_stack = forth.data_stack();
    print!("<{}> ", data_stack.len());
    data_stack.iter().for_each(|v| {
        print!("{} ", v);
    });
    println!();
    Ok(())
}

const DOCOL_IX: usize = 0;
const DOCREATE_IX: usize = 1;
const SPECIAL_CODEWORDS: [(&str, fn(&mut ForthMachine) -> Result<(), ForthError>); 2] =
    [("DOCOL", docol), ("DOCREATE", docreate)];

const BUILTIN_WORDS: [(&str, u8, fn(&mut ForthMachine) -> Result<(), ForthError>); 58] = [
    // Stack manipulation
    (".S", 0, print_data_stack_builtin),
    ("DROP", 0, drop_builtin),
    ("DUP", 0, dup_builtin),
    ("NIP", 0, nip_builtin),
    ("ROT", 0, rot_builtin),
    ("SWAP", 0, swap_builtin),
    // Arithmetic
    ("*", 0, mul_builtin),
    ("+", 0, add_builtin),
    ("-", 0, sub_builtin),
    ("/MOD", 0, slash_mod_builtin),
    ("=", 0, eq_builtin),
    (">", 0, greater_than_builtin),
    ("ABS", 0, abs_builtin),
    ("M*", 0, m_mul_builtin),
    ("S>D", 0, s_to_d_builtin),
    ("SM/REM", 0, sm_slash_rem_builtin),
    ("U>", 0, u_greater_than_builtin),
    // Bit operations
    ("AND", 0, and_builtin),
    ("INVERT", 0, invert_builtin),
    ("LSHIFT", 0, lshift_builtin),
    ("OR", 0, or_builtin),
    ("RSHIFT", 0, rshift_builtin),
    ("XOR", 0, xor_builtin),
    // Input buffer & parse area
    // `>IN` is a variable defined in `ForthMachine::new`
    ("REFILL", 0, refill_builtin),
    ("SOURCE", 0, source_builtin),
    ("WORD", 0, word_builtin),
    ("\\", WordFlag::Immediate as u8, backslash_builtin),
    // Output
    ("EMIT", 0, emit_builtin),
    // Memory
    ("!", 0, store_builtin),
    (",", 0, comma_builtin),
    ("@", 0, fetch_builtin),
    ("ALIGN", 0, align_builtin),
    ("ALLOT", 0, allot_builtin),
    ("C!", 0, store_byte_builtin),
    ("C@", 0, fetch_byte_builtin),
    ("CELLS", 0, cells_builtin),
    ("HERE", 0, here_builtin),
    ("UNUSED", 0, unused_builtin),
    // Dictionary
    (">CFA", 0, to_cfa_builtin),
    ("CREATE", 0, create_builtin),
    ("EXECUTE", 0, execute_builtin),
    ("FIND", 0, find_builtin),
    ("HIDDEN", 0, hidden_builtin),
    ("IMMEDIATE", WordFlag::Immediate as u8, immediate_builtin),
    ("LATEST", 0, latest_builtin),
    // Control flow
    ("BRANCH", 0, branch_builtin),
    ("0BRANCH", 0, zbranch_builtin),
    // Return stack manipulation
    ("RS-CLEAR", 0, rs_clear_builtin),
    (">R", 0, to_rstack_builtin),
    ("R>", 0, from_rstack_builtin),
    ("R@", 0, rstack_fetch_builtin),
    // Misc
    ("BYE", 0, bye_builtin),
    ("EXIT", 0, exit_builtin),
    ("INTERPRET-SINGLE", 0, interpret_single_builtin),
    ("LIT", 0, lit_builtin),
    ("S>NUMBER?", 0, s_to_number_builtin),
    ("[", WordFlag::Immediate as u8, lbrac_builtin),
    ("]", 0, rbrac_builtin),
];

fn exec_fun_indirect(def_addr: ForthPtr, forth: &mut ForthMachine) -> Result<(), ForthError> {
    forth.curr_def_addr = def_addr;
    // println!("forth.curr_def_addr: {:#x}", forth.curr_def_addr);
    let word_ix = forth.data_space.read_cell(def_addr)? as ForthUCell as usize;
    let maybe_builtin = if word_ix < SPECIAL_CODEWORDS.len() {
        Some(SPECIAL_CODEWORDS[word_ix])
    } else if word_ix - SPECIAL_CODEWORDS.len() < BUILTIN_WORDS.len() {
        let (name, _flags, fun) = BUILTIN_WORDS[word_ix - SPECIAL_CODEWORDS.len()];
        Some((name, fun))
    } else {
        None
    };
    // TODO: Setup debugging & tracing facilities.
    // match maybe_builtin {
    //     None => println!("Executing builtin fun at index {}", word_ix),
    //     Some((name, _)) => println!("Executing builtin '{}' (index {})", name, word_ix),
    // }
    let builtin = maybe_builtin.ok_or(ForthError::InvalidPointer(def_addr))?;
    let fun = builtin.1;
    match fun(forth) {
        e @ Err(..) => {
            // TODO: This doesn't work well when recursive `exec_fun_indirect` is combined with
            // `next`. For example, builtins will show `INTERPRET-SINGLE` at incorrect place, while
            // `docol` will omit `INTERPRET-SINGLE` from the backtrace.
            forth.backtrace.push(def_addr);
            e
        }
        r => r,
    }
}

fn next(forth: &mut ForthMachine) -> Result<(), ForthError> {
    if forth.instruction_addr.is_null() {
        return Ok(());
    }
    let def_addr = ForthPtr(forth.data_space.read_cell(forth.instruction_addr)? as ForthUCell);
    forth.instruction_addr = forth
        .instruction_addr
        .checked_add(FORTH_CELL_SIZE)
        .ok_or(ForthError::AddressOverflow(forth.instruction_addr))?;
    // println!("curr_def_addr at {:#x}", def_addr);
    // println!("instruction_addr at {:#x}", forth.instruction_addr);
    exec_fun_indirect(def_addr, forth)
}

// Store instruction at `def_addr` and return the address in `data_space` where
// it was stored.
fn push_instruction(data_space: &mut DataSpace, def_addr: ForthUCell) -> ForthPtr {
    assert!(is_aligned(
        data_space.memory.current as usize,
        FORTH_CELL_SIZE as usize
    ));
    let ptr = data_space.alloc(FORTH_CELL_SIZE as usize).unwrap();
    data_space.write_cell(ptr, def_addr as ForthCell).unwrap();
    ptr
}

fn push_word<'a, I>(data_space: &mut DataSpace, name: &str, flags: u8, words: I)
where
    I: IntoIterator<Item = &'a str>,
{
    assert!(data_space.push_dict_entry(name, flags));
    push_instruction(data_space, DOCOL_IX as ForthUCell);
    for word in words {
        match data_space.find_entry(word) {
            Some(entry) => {
                push_instruction(data_space, entry.definition_addr().0);
            }
            None => {
                let base = 10;
                let num = parse_num(word, base).unwrap();
                // push_instruction(
                //     data_space,
                //     data_space.find_entry("LIT").unwrap().definition_addr().0,
                // );
                push_instruction(data_space, num as ForthUCell);
            }
        };
    }
    let def_addr = data_space.find_entry("EXIT").unwrap().definition_addr();
    push_instruction(data_space, def_addr.0);
}

fn push_variable(data_space: &mut DataSpace, name: &str, val: ForthCell) -> ForthPtr {
    assert!(data_space.push_dict_entry(name, 0));
    // TODO: Add a builtin `dovar` to save extra space by avoiding the offset ptr for
    // `docreate`. (See how `create_builtin` works).
    push_instruction(data_space, DOCREATE_IX as ForthUCell);
    push_instruction(data_space, 0);
    push_instruction(data_space, val as ForthUCell)
}

fn push_buffer(data_space: &mut DataSpace, name: &str, bytes: usize) -> ForthPtr {
    assert!(data_space.push_dict_entry(name, 0));
    // TODO: Add a builtin `dovar` to save extra space by avoiding the offset ptr for
    // `docreate`. (See how `create_builtin` works).
    push_instruction(data_space, DOCREATE_IX as ForthUCell);
    push_instruction(data_space, 0);
    let ptr = data_space
        .alloc(bytes)
        .expect("data space should have enough memory");
    assert!(data_space.align());
    ptr
}

fn set_instruction(forth: &mut ForthMachine, word: &str) {
    assert!(forth.data_space.align());
    let def_addr = forth.data_space.find_entry(word).unwrap().definition_addr();
    forth.instruction_addr = push_instruction(&mut forth.data_space, def_addr.0);
}

fn parse_memsize(val: &str) -> clap::error::Result<usize> {
    let last_char = val.chars().last().ok_or_else(|| {
        clap::error::Error::raw(clap::error::ErrorKind::ValueValidation, "empty value")
    })?;
    let (unit, val) = match last_char {
        'b' => (1, &val[..val.len() - 1]),
        'k' => (1024, &val[..val.len() - 1]),
        'M' => (1024 * 1024, &val[..val.len() - 1]),
        'G' => (1024 * 1024 * 1024, &val[..val.len() - 1]),
        // Default is 'k'
        _ => (1024, val),
    };
    let num: usize = val
        .parse()
        .map_err(|e| clap::error::Error::raw(clap::error::ErrorKind::ValueValidation, e))?;
    Ok(num * unit)
}

fn fmt_memsize(bytes: usize) -> String {
    let chars = bytes
        .to_string()
        .chars()
        .rev()
        .enumerate()
        .fold(Vec::new(), |mut res, (i, ch)| {
            res.push(ch);
            if (i + 1) % 3 == 0 {
                res.push(',');
            }
            res
        });
    if chars.last().unwrap() == &',' {
        chars.iter().rev().skip(1).collect()
    } else {
        chars.iter().rev().collect()
    }
}

#[derive(Parser)]
#[command(name = "rForth")]
/// Simple Forth Virtual Machine
///
/// To pass a forth source code file, invoke with
///
/// ```sh
/// $ cat FORTH_FILE | rforth
/// ```
///
/// If you want to continue running the interpreter, use
///
/// ```sh
/// $ cat FORTH_FILE - | rforth
/// ```
struct CliArgs {
    /// Size of total memory used for Forth's data space.
    ///
    /// You may suffix the number with one of 'b', 'k', 'M' & 'G' to specify
    /// the size unit. Omitting the suffix defaults to 'k', i.e. kilobytes.
    #[arg(long, default_value = "8k", value_parser = parse_memsize)]
    data_space_size: usize,
    /// Maximum number of cells on the data stack.
    #[arg(long, default_value_t = 64)]
    data_stack_size: usize,
    /// Maximum number of cells on the return stack.
    #[arg(long, default_value_t = 64)]
    return_stack_size: usize,
}

pub fn with_cli_args() -> ForthMachine {
    let cli_args = CliArgs::parse();
    let mut forth = ForthMachine::new(
        DataSpace::with_size(cli_args.data_space_size),
        cli_args.data_stack_size,
        cli_args.return_stack_size,
    );
    push_word(
        &mut forth.data_space,
        ":",
        0,
        [
            "CREATE", // This will push `docreate` as codeword, and 0 ptr.
            "LIT",
            &(-(FORTH_CELL_SIZE as ForthCell)).to_string(),
            "ALLOT", // Deallocate the 0 ptr.
            "LIT",
            &DOCOL_IX.to_string(),
            "LATEST", /* "@" */
            ">CFA",
            "!", // Overwrite `docreate` codeword with docol
            "LATEST",
            /* "@", */ "HIDDEN",
            "]",
        ],
    );
    push_word(
        &mut forth.data_space,
        ";",
        WordFlag::Immediate as u8,
        ["LIT", "EXIT", ",", "LATEST", /* "@", */ "HIDDEN", "["],
    );
    push_word(
        &mut forth.data_space,
        "INTERPRET",
        0,
        [
            "INTERPRET-SINGLE",
            "SOURCE", // ( c-addr len )
            "NIP",    // ( len )
            ">IN",
            "@",
            // ( len offset )
            "U>",
            "0BRANCH",
            &(3 * FORTH_CELL_SIZE).to_string(), // If false jump to exit.
            "BRANCH",                           // If len > offset, repeat interpret.
            &(-9 * FORTH_CELL_SIZE as ForthCell).to_string(),
        ],
    );
    push_word(
        &mut forth.data_space,
        "QUIT",
        0,
        [
            "RS-CLEAR",
            "REFILL",
            "0BRANCH",
            &(4 * FORTH_CELL_SIZE).to_string(),
            "INTERPRET",
            "BRANCH",
            &(-5 * FORTH_CELL_SIZE as ForthCell).to_string(),
            "BYE",
        ],
    );
    forth
}

pub fn run(forth: &mut ForthMachine) {
    run_with(forth, |_| {});
}

pub fn run_with(forth: &mut ForthMachine, mut fun: impl FnMut(&mut ForthMachine)) {
    println!("Welcome to rForth");
    println!(
        "data_space_size = {} bytes",
        fmt_memsize(forth.data_space.size())
    );
    println!(
        "data_stack_size = {} cells ({} bytes)",
        forth.data_stack.max_elements,
        fmt_memsize(forth.data_stack.max_elements * FORTH_CELL_SIZE as usize)
    );
    println!(
        "return_stack_size = {} cells ({} bytes)",
        forth.return_stack.max_elements,
        fmt_memsize(forth.return_stack.max_elements * FORTH_CELL_SIZE as usize)
    );
    // This needs to allocate a PTR_SIZE to store indirect address of QUIT.
    set_instruction(forth, "QUIT");
    // We will reuse the allocated indirect address to reset the state on error.
    let quit_instruction_addr = forth.instruction_addr;
    println!("unused {} bytes", fmt_memsize(forth.data_space.unused()));
    while !forth.instruction_addr.is_null() {
        if let Err(e) = next(forth) {
            println!("ERROR: {:?}", e);
            println!("  Backtrace");
            for (ix, def_addr) in forth.backtrace.iter().rev().enumerate() {
                let word_name = forth
                    .data_space
                    .find_entry_by_def_addr(*def_addr)
                    .map(|entry| entry.name().to_string())
                    .unwrap_or_else(|| "<UNKOWN-WORD>".to_string());
                println!("    {}: {}", ix, word_name);
            }
            forth.backtrace.clear();
            // Reset the instruction_addr back to QUIT.
            // TODO: This allows for redefined QUIT, but that brings additional problems. Perhaps
            // hard code it back to original QUIT.
            let quit_def_addr = forth
                .data_space
                .find_entry("QUIT")
                .unwrap()
                .definition_addr();
            forth
                .data_space
                .write_cell(quit_instruction_addr, quit_def_addr.0 as ForthCell)
                .unwrap();
            forth.instruction_addr = quit_instruction_addr;
        }
        fun(forth);
    }
    println!("Bye!");
}
