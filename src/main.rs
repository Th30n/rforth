use clap::Parser;
use std::convert::TryFrom;
use std::io::{Read, Write};

const PTR_SIZE: usize = std::mem::size_of::<usize>();

fn is_aligned(addr: usize, alignment: usize) -> bool {
    addr % alignment == 0
}

fn align_addr(addr: usize) -> usize {
    assert_eq!(std::mem::align_of::<usize>(), PTR_SIZE);
    let aligned_addr = (addr + (PTR_SIZE - 1)) & (!PTR_SIZE + 1);
    assert!(is_aligned(aligned_addr, PTR_SIZE));
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
    pub fn align(&mut self) -> bool {
        let aligned_addr = align_addr(self.current as usize);
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
    let mut memory = Memory::with_size(1024);
    assert_eq!(memory.unused(), 1024);
    assert!(memory.align());
    assert_eq!(memory.unused(), 1024);
    memory.alloc(8);
    assert_eq!(memory.unused(), 1024 - 8);
    assert!(memory.align());
    assert_eq!(memory.unused(), 1024 - 8);
    memory.alloc(1);
    assert_eq!(memory.unused(), 1024 - 9);
    assert!(memory.align());
    assert_eq!(memory.unused(), 1024 - 16);
    memory.alloc(7);
    assert_eq!(memory.unused(), 1024 - 16 - 7);
    assert!(memory.align());
    assert_eq!(memory.unused(), 1024 - 16 - 8);
}

enum WordFlag {
    Hidden = 0x20,
    Immediate = 0x80,
}

/// Entry in a Forth dictionary
///
/// Memory layout of a dict entry is the following.
///
/// | prev: ptr | flags: usize | len: usize | name: bytes | pad | definition...
///
/// Definition list consists of | codeword | definition addr...
///
/// The codeword is the native address to a function, so we call that directly.
/// It is either a `docol` or some of the `*_builtin` functions.
/// Remaining `definition addr` are pointers to the start of a `definition` in
/// some other dict entry, i.e. word.
#[derive(Copy, Clone)]
struct DictEntryRef<'a> {
    data_space: &'a DataSpace,
    /// Points to start of the entry in DataSpace.
    ptr: *mut u8,
}

impl<'a> DictEntryRef<'a> {
    fn ptr_as_slice(&self) -> &[u8] {
        let ptr = self.ptr as *const u8;
        assert!(self.data_space.is_valid_ptr(ptr));
        unsafe { std::slice::from_raw_parts(ptr, self.data_space.end() - ptr as usize) }
    }

    fn ptr_as_slice_mut(&mut self) -> &mut [u8] {
        assert!(self.data_space.is_valid_ptr(self.ptr));
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.data_space.end() - self.ptr as usize)
        }
    }

    pub fn prev(&self) -> Option<DictEntryRef<'a>> {
        let buf_slice = self.ptr_as_slice();
        let prev_bytes = <[u8; PTR_SIZE]>::try_from(&buf_slice[0..PTR_SIZE]).unwrap();
        let prev_ptr = usize::from_ne_bytes(prev_bytes) as *mut u8;
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

    pub fn flags(&self) -> usize {
        let flags_buf = &self.ptr_as_slice()[PTR_SIZE..][..PTR_SIZE];
        usize::from_ne_bytes(<[u8; PTR_SIZE]>::try_from(flags_buf).unwrap())
    }

    fn set_flags(&mut self, flags: usize) {
        let flags_buf = &mut self.ptr_as_slice_mut()[PTR_SIZE..][..PTR_SIZE];
        let mut cursor = std::io::Cursor::new(flags_buf);
        cursor.write_all(&flags.to_ne_bytes()).unwrap();
    }

    pub fn name(&self) -> &str {
        let (len_buf, name_buf) = self.ptr_as_slice()[2 * PTR_SIZE..].split_at(PTR_SIZE);
        let len = usize::from_ne_bytes(<[u8; PTR_SIZE]>::try_from(len_buf).unwrap());
        std::str::from_utf8(&name_buf[0..len]).unwrap()
    }

    pub fn definition_addr(&self) -> usize {
        let addr = align_addr(self.name().as_ptr() as usize + self.name().len());
        assert!(self.data_space.is_valid_ptr(addr as *const usize));
        addr
    }

    // Return true if now hidden.
    pub fn toggle_hidden(&mut self) -> bool {
        let new_flags = self.flags() ^ WordFlag::Hidden as usize;
        self.set_flags(new_flags);
        self.is_hidden()
    }

    pub fn is_hidden(&self) -> bool {
        (self.flags() & WordFlag::Hidden as usize) != 0
    }

    // Return true if now immediate.
    pub fn toggle_immediate(&mut self) -> bool {
        let new_flags = self.flags() ^ WordFlag::Immediate as usize;
        self.set_flags(new_flags);
        self.is_immediate()
    }

    pub fn is_immediate(&self) -> bool {
        (self.flags() & WordFlag::Immediate as usize) != 0
    }
}

impl std::fmt::Debug for DictEntryRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DictEntryRef {{ ptr: {:p}, end_of_data_space: {:p}, prev: {:p}, name: {}, flags: {:#x}, definition_addr: {:p} }}",
            self.ptr as *const u8,
            self.data_space.end() as *const u8,
            self.prev().map_or(std::ptr::null(), |e| e.ptr),
            self.name(),
            self.flags(),
            self.definition_addr() as *const u8,
        )
    }
}

/// Memory for Forth's data space.
///
/// Data space includes dictionary entries and user allocations.
#[derive(Debug)]
struct DataSpace {
    memory: Memory,
    dict_head: *mut u8,
    builtin_addrs: Vec<(usize, String)>,
}

impl DataSpace {
    pub fn with_size(bytes: usize) -> Self {
        let mut builtin_addrs = Vec::with_capacity(64);
        builtin_addrs.push((docol as usize, "DOCOL".to_string()));
        DataSpace {
            memory: Memory::with_size(bytes),
            dict_head: std::ptr::null_mut(),
            builtin_addrs,
        }
    }

    pub fn size(&self) -> usize {
        self.memory.size()
    }

    /// Past the end adress of DataSpace
    pub fn end(&self) -> usize {
        self.memory.ptr as usize + self.size()
    }

    pub fn here(&self) -> usize {
        self.memory.current as usize
    }

    pub fn unused(&self) -> usize {
        self.memory.unused()
    }

    pub fn is_valid_ptr<T>(&self, ptr: *const T) -> bool {
        self.memory.is_valid_ptr(ptr)
    }

    #[must_use]
    pub fn align(&mut self) -> bool {
        self.memory.align()
    }

    pub fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        self.memory.alloc(bytes)
    }

    #[must_use]
    pub fn dealloc(&mut self, bytes: usize) -> bool {
        // TODO: Remove obsolete entries
        let succ = self.memory.dealloc(bytes);
        assert!(self.dict_head.is_null() || self.is_valid_ptr(self.dict_head));
        succ
    }

    /// Allocate a new entry in the dictionary with given `name`.
    /// The entry is padded, so that the next allocation is aligned.
    #[must_use]
    pub fn push_dict_entry(&mut self, name: &str, flags: usize) -> bool {
        let prev_addr = self.dict_head as usize;
        let entry_size = PTR_SIZE + PTR_SIZE + PTR_SIZE + name.len();
        match self.alloc(entry_size) {
            None => false,
            Some(entry_ptr) => {
                let buf = unsafe { std::slice::from_raw_parts_mut(entry_ptr, entry_size) };
                let mut cursor = std::io::Cursor::new(buf);
                cursor.write_all(&prev_addr.to_ne_bytes()).unwrap();
                cursor.write_all(&flags.to_ne_bytes()).unwrap();
                cursor.write_all(&name.len().to_ne_bytes()).unwrap();
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

    pub fn is_builtin_addr(&self, addr: usize) -> bool {
        self.builtin_addrs.iter().any(|(a, _)| *a == addr)
    }

    pub fn push_builtin_word(&mut self, name: &str, flags: usize, word_fn: fn(&mut ForthMachine)) {
        assert!(self.push_dict_entry(name, flags));
        let fn_addr = word_fn as usize;
        if !self.is_builtin_addr(fn_addr) {
            self.builtin_addrs.push((fn_addr, name.to_string()));
        }
        let ptr = self.alloc(PTR_SIZE).unwrap();
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, PTR_SIZE) };
        let mut cursor = std::io::Cursor::new(buf);
        cursor.write_all(&fn_addr.to_ne_bytes()).unwrap();
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
        assert!(std::ptr::eq(dup.ptr, latest.ptr));
    }
    assert!(data_space.push_dict_entry("SWAP", 0));
    {
        let dup = data_space.find_entry("DUP").unwrap();
        assert_eq!(dup.name(), "DUP");
        let latest = data_space.latest_entry().unwrap();
        assert!(!std::ptr::eq(dup.ptr, latest.ptr));
        assert!(std::ptr::eq(dup.ptr, latest.prev().unwrap().ptr));
    }
    assert!(data_space.push_dict_entry("DUP", 0));
    {
        let dup = data_space.find_entry("DUP").unwrap();
        assert_eq!(dup.name(), "DUP");
        let latest = data_space.latest_entry().unwrap();
        assert!(std::ptr::eq(dup.ptr, latest.ptr));
        let orig_dup = data_space
            .latest_entry()
            .and_then(|e| e.prev())
            .and_then(|e| e.prev())
            .unwrap();
        assert_eq!(dup.name(), orig_dup.name());
        assert!(!std::ptr::eq(dup.ptr, orig_dup.ptr));
    }
}

const FORTH_TRUE: isize = -1;
const FORTH_FALSE: isize = 0;
const INPUT_BUFFER_SIZE: usize = 4096;
const WORD_BUFFER_SIZE: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ForthState {
    Immediate = 0,
    Compile = 1,
}

#[derive(Debug)]
struct ForthMachine {
    data_space: DataSpace,
    data_stack: Vec<isize>,
    return_stack: Vec<isize>,
    curr_def_addr: usize,
    instruction_addr: usize,
    // Input bufer and vars for KEY word.
    input_buffer: [u8; INPUT_BUFFER_SIZE],
    curr_input_ix: usize,
    curr_input_len: usize,
    // Buffer inside `data_space` for WORD word; length is WORD_BUFFER_SIZE
    word_buffer_ptr: *mut u8,
    // TODO: Move state to DataSpace
    state: ForthState,
}

impl ForthMachine {
    pub fn new(
        mut data_space: DataSpace,
        data_stack: Vec<isize>,
        return_stack: Vec<isize>,
    ) -> Self {
        let word_buffer_ptr = data_space.alloc(WORD_BUFFER_SIZE).unwrap();
        assert!(data_space.align());
        add_builtins(&mut data_space);
        Self {
            data_space,
            data_stack,
            return_stack,
            instruction_addr: 0,
            curr_def_addr: 0,
            input_buffer: [0; INPUT_BUFFER_SIZE],
            curr_input_ix: 0,
            curr_input_len: 0,
            word_buffer_ptr,
            state: ForthState::Immediate,
        }
    }

    fn word_buffer(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.word_buffer_ptr, WORD_BUFFER_SIZE) }
    }
}

fn bye_builtin(forth: &mut ForthMachine) {
    forth.instruction_addr = 0;
}

fn drop_builtin(forth: &mut ForthMachine) {
    forth.data_stack.pop().unwrap();
}

fn dup_builtin(forth: &mut ForthMachine) {
    forth.data_stack.push(*forth.data_stack.last().unwrap());
}

fn swap_builtin(forth: &mut ForthMachine) {
    let a = forth.data_stack.pop().unwrap();
    let b = forth.data_stack.pop().unwrap();
    forth.data_stack.push(a);
    forth.data_stack.push(b);
}

// DOCOL is special, it is a codeword and as such its direct address should be
// used when creating new words.
// TODO: Maybe make a DOCOL word (constant) that pushes this address on stack.
fn docol(forth: &mut ForthMachine) {
    forth.return_stack.push(forth.instruction_addr as isize);
    forth.instruction_addr = forth.curr_def_addr.checked_add(PTR_SIZE).unwrap();
}

fn exit_builtin(forth: &mut ForthMachine) {
    forth.instruction_addr = forth.return_stack.pop().unwrap() as usize;
}

// Return Ok(None) on EOF.
fn read_stdin_byte(forth: &mut ForthMachine) -> std::io::Result<Option<u8>> {
    if forth.curr_input_ix == forth.curr_input_len {
        let num_read = std::io::stdin().read(&mut forth.input_buffer)?;
        // Handle EOF
        if num_read == 0 {
            return Ok(None);
        }
        forth.curr_input_len = num_read;
        forth.curr_input_ix = 0;
    }
    assert!(forth.curr_input_ix < forth.curr_input_len);
    assert!(forth.curr_input_len <= forth.input_buffer.len());
    let byte = forth.input_buffer[forth.curr_input_ix];
    forth.curr_input_ix += 1;
    Ok(Some(byte))
}

fn key_builtin(forth: &mut ForthMachine) {
    if let Some(byte) = read_stdin_byte(forth).unwrap() {
        forth.data_stack.push(byte as isize)
    } else {
        bye_builtin(forth)
    }
}

fn emit_builtin(forth: &mut ForthMachine) {
    let v = forth.data_stack.pop().unwrap() as u8;
    assert_eq!(std::io::stdout().write(&[v]).unwrap(), 1);
}

const BLANK_CHARS: [char; 3] = [' ', '\t', '\n'];

fn do_word_builtin(forth: &mut ForthMachine) -> Option<()> {
    let mut byte = read_stdin_byte(forth).unwrap()?;
    loop {
        // Skip blanks
        while BLANK_CHARS.contains(&byte.into()) {
            byte = read_stdin_byte(forth).unwrap()?;
        }
        // Skip comment until newline
        if byte == b'\\' {
            while byte != b'\n' {
                byte = read_stdin_byte(forth).unwrap()?;
            }
        } else {
            break;
        }
    }
    let mut word_buffer_ix = 0;
    while !BLANK_CHARS.contains(&byte.into()) {
        assert!(word_buffer_ix < WORD_BUFFER_SIZE);
        forth.word_buffer()[word_buffer_ix] = byte;
        byte = read_stdin_byte(forth).unwrap()?;
        word_buffer_ix += 1;
    }
    forth.data_stack.push(forth.word_buffer_ptr as isize);
    forth.data_stack.push(word_buffer_ix as isize); // len
    Some(())
}

fn word_builtin(forth: &mut ForthMachine) {
    if do_word_builtin(forth).is_none() {
        bye_builtin(forth)
    }
}

fn store_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *mut isize;
    assert!(forth.data_space.is_valid_ptr(ptr));
    let val = forth.data_stack.pop().unwrap();
    let addr_ref = unsafe { ptr.as_mut().unwrap() };
    *addr_ref = val
}

fn fetch_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *const isize;
    assert!(forth.data_space.is_valid_ptr(ptr));
    let addr_ref = unsafe { ptr.as_ref().unwrap() };
    forth.data_stack.push(*addr_ref)
}

fn store_byte_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *mut u8;
    assert!(forth.data_space.is_valid_ptr(ptr));
    let val = forth.data_stack.pop().unwrap() as u8;
    let addr_ref = unsafe { ptr.as_mut().unwrap() };
    *addr_ref = val
}

fn fetch_byte_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *const u8;
    assert!(forth.data_space.is_valid_ptr(ptr));
    let addr_ref = unsafe { ptr.as_ref().unwrap() };
    forth.data_stack.push(*addr_ref as isize)
}

// Converts string to number, flag indicates success.
// (addr u - d f)
fn number_builtin(forth: &mut ForthMachine) {
    let byte_len = forth.data_stack.pop().unwrap() as usize;
    let ptr = forth.data_stack.pop().unwrap() as *const u8;
    if byte_len == 0 {
        forth.data_stack.push(0);
        forth.data_stack.push(FORTH_FALSE);
        return;
    }
    assert!(forth.data_space.is_valid_ptr(ptr));
    assert!(forth
        .data_space
        .is_valid_ptr((ptr as usize + byte_len - 1) as *const u8));
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    // TODO: Use BASE variable for parsing.
    match std::str::from_utf8(bytes).unwrap().parse() {
        Ok(num) => {
            forth.data_stack.push(num);
            forth.data_stack.push(FORTH_TRUE);
        }
        _ => {
            forth.data_stack.push(0);
            forth.data_stack.push(FORTH_FALSE);
        }
    }
}

// Return name token (start of dict entry) or 0.
// (addr u - nt | 0)
fn find_builtin(forth: &mut ForthMachine) {
    let byte_len = forth.data_stack.pop().unwrap() as usize;
    let ptr = forth.data_stack.pop().unwrap() as *const u8;
    assert!(forth.data_space.is_valid_ptr(ptr));
    assert!(forth
        .data_space
        .is_valid_ptr((ptr as usize + byte_len - 1) as *const u8));
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    let name = std::str::from_utf8(bytes).unwrap();
    match forth.data_space.find_entry(name) {
        None => forth.data_stack.push(0),
        Some(dict_entry_ref) => forth
            .data_stack
            .push(dict_entry_ref.ptr as *const u8 as isize),
    }
}

fn latest_builtin(forth: &mut ForthMachine) {
    forth.data_stack.push(forth.data_space.dict_head as isize)
}

// Return Code Field Address (i.e. definition address of a dict entry).
fn to_cfa_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *mut u8;
    let dict_entry_ref = DictEntryRef {
        data_space: &forth.data_space,
        ptr,
    };
    forth
        .data_stack
        .push(dict_entry_ref.definition_addr() as isize);
}

fn hidden_builtin(forth: &mut ForthMachine) {
    let ptr = forth.data_stack.pop().unwrap() as *mut u8;
    let mut dict_entry_ref = DictEntryRef {
        data_space: &forth.data_space,
        ptr,
    };
    dict_entry_ref.toggle_hidden();
}

fn immediate_builtin(forth: &mut ForthMachine) {
    forth.data_space.latest_entry().unwrap().toggle_immediate();
}

fn branch_builtin(forth: &mut ForthMachine) {
    let ptr_to_offset = forth.instruction_addr as *const isize;
    assert!(forth.data_space.is_valid_ptr(ptr_to_offset));
    let offset = unsafe { *ptr_to_offset };
    if offset >= 0 {
        forth.instruction_addr = forth.instruction_addr.checked_add(offset as usize).unwrap();
    } else {
        forth.instruction_addr = forth
            .instruction_addr
            .checked_sub(offset.unsigned_abs())
            .unwrap();
    }
}

fn zbranch_builtin(forth: &mut ForthMachine) {
    let val = forth.data_stack.pop().unwrap();
    if val == 0 {
        branch_builtin(forth);
    } else {
        forth.instruction_addr = forth.instruction_addr.checked_add(PTR_SIZE).unwrap();
    }
}

fn lit_builtin(forth: &mut ForthMachine) {
    let ptr_to_val = forth.instruction_addr as *const isize;
    assert!(forth.data_space.is_valid_ptr(ptr_to_val));
    let val = unsafe { *ptr_to_val };
    // println!("val: {:#x}", val);
    forth.instruction_addr = forth.instruction_addr.checked_add(PTR_SIZE).unwrap();
    forth.data_stack.push(val);
}

fn create_builtin(forth: &mut ForthMachine) {
    let byte_len = forth.data_stack.pop().unwrap() as usize;
    let ptr = forth.data_stack.pop().unwrap() as *const u8;
    assert!(forth.data_space.is_valid_ptr(ptr));
    assert!(forth
        .data_space
        .is_valid_ptr((ptr as usize + byte_len - 1) as *const u8));
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    let name = std::str::from_utf8(bytes).unwrap();
    assert!(forth.data_space.push_dict_entry(name, 0));
}

fn comma_builtin(forth: &mut ForthMachine) {
    let def_addr = forth.data_stack.pop().unwrap();
    push_instruction(&mut forth.data_space, def_addr as usize);
}

fn lbrac_builtin(forth: &mut ForthMachine) {
    forth.state = ForthState::Immediate;
}

fn rbrac_builtin(forth: &mut ForthMachine) {
    forth.state = ForthState::Compile;
}

fn interpret_builtin(forth: &mut ForthMachine) {
    if do_word_builtin(forth).is_none() {
        return bye_builtin(forth);
    }
    let byte_len = forth.data_stack.pop().unwrap() as usize;
    let ptr = forth.data_stack.pop().unwrap() as *const u8;
    assert!(forth.data_space.is_valid_ptr(ptr));
    assert!(forth
        .data_space
        .is_valid_ptr((ptr as usize + byte_len - 1) as *const u8));
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    let name = std::str::from_utf8(bytes).unwrap();
    if let Some(dict_entry_ref) = forth.data_space.find_entry(name) {
        // dbg!(dict_entry_ref);
        if dict_entry_ref.is_immediate() || forth.state == ForthState::Immediate {
            exec_fun_indirect(dict_entry_ref.definition_addr(), forth);
        } else {
            // Compiling
            let def_addr = dict_entry_ref.definition_addr();
            push_instruction(&mut forth.data_space, def_addr);
        }
    } else {
        // TODO: Use BASE variable for parsing.
        let num: isize = name
            .parse()
            .unwrap_or_else(|_| panic!("Unable to parse '{}' as number", name));
        match forth.state {
            ForthState::Immediate => forth.data_stack.push(num),
            ForthState::Compile => {
                let lit = forth
                    .data_space
                    .find_entry("LIT")
                    .unwrap()
                    .definition_addr();
                push_instruction(&mut forth.data_space, lit);
                push_instruction(&mut forth.data_space, num as usize);
            }
        }
    }
}

fn rs_clear_builtin(forth: &mut ForthMachine) {
    forth.return_stack.clear();
}

fn tick_builtin(forth: &mut ForthMachine) {
    let ptr_to_def_addr = forth.instruction_addr as *const isize;
    assert!(forth.data_space.is_valid_ptr(ptr_to_def_addr));
    let def_addr = unsafe { *ptr_to_def_addr };
    forth.instruction_addr = forth.instruction_addr.checked_add(PTR_SIZE).unwrap();
    forth.data_stack.push(def_addr);
}

fn execute_builtin(forth: &mut ForthMachine) {
    let def_addr = forth.data_stack.pop().unwrap() as usize;
    exec_fun_indirect(def_addr, forth);
}

fn here_builtin(forth: &mut ForthMachine) {
    // TODO: Make this put HERE variable address on stack.
    // Now it behaves like a constant.
    forth.data_stack.push(forth.data_space.here() as isize);
}

fn sub_builtin(forth: &mut ForthMachine) {
    let b = forth.data_stack.pop().unwrap();
    let a = forth.data_stack.last_mut().unwrap();
    *a = a.wrapping_sub(b);
}

fn add_builtins(data_space: &mut DataSpace) {
    data_space.push_builtin_word("BYE", 0, bye_builtin);
    data_space.push_builtin_word("DROP", 0, drop_builtin);
    data_space.push_builtin_word("DUP", 0, dup_builtin);
    data_space.push_builtin_word("SWAP", 0, swap_builtin);
    data_space.push_builtin_word("EXIT", 0, exit_builtin);
    data_space.push_builtin_word("KEY", 0, key_builtin);
    data_space.push_builtin_word("EMIT", 0, emit_builtin);
    data_space.push_builtin_word("WORD", 0, word_builtin);
    data_space.push_builtin_word("!", 0, store_builtin);
    data_space.push_builtin_word("@", 0, fetch_builtin);
    data_space.push_builtin_word("C!", 0, store_byte_builtin);
    data_space.push_builtin_word("C@", 0, fetch_byte_builtin);
    data_space.push_builtin_word("S>NUMBER?", 0, number_builtin);
    data_space.push_builtin_word("FIND", 0, find_builtin);
    data_space.push_builtin_word("LATEST", 0, latest_builtin);
    data_space.push_builtin_word(">CFA", 0, to_cfa_builtin);
    data_space.push_builtin_word("HIDDEN", 0, hidden_builtin);
    data_space.push_builtin_word("IMMEDIATE", WordFlag::Immediate as usize, immediate_builtin);
    data_space.push_builtin_word("BRANCH", 0, branch_builtin);
    data_space.push_builtin_word("0BRANCH", 0, zbranch_builtin);
    data_space.push_builtin_word("LIT", 0, lit_builtin);
    data_space.push_builtin_word("CREATE", 0, create_builtin);
    data_space.push_builtin_word(",", 0, comma_builtin);
    data_space.push_builtin_word("[", WordFlag::Immediate as usize, lbrac_builtin);
    data_space.push_builtin_word("]", 0, rbrac_builtin);
    data_space.push_builtin_word("INTERPRET", 0, interpret_builtin);
    data_space.push_builtin_word("RS-CLEAR", 0, rs_clear_builtin);
    data_space.push_builtin_word("'", 0, tick_builtin);
    data_space.push_builtin_word("HERE", 0, here_builtin);
    data_space.push_builtin_word("-", 0, sub_builtin);
    data_space.push_builtin_word("EXECUTE", 0, execute_builtin);
}

fn exec_fun_indirect(addr: usize, forth: &mut ForthMachine) {
    forth.curr_def_addr = addr;
    // println!("forth.curr_def_addr: {:#x}", forth.curr_def_addr);
    assert!(
        forth.data_space.is_valid_ptr(addr as *const usize),
        "'{addr:#x} ({addr})' is not a valid ptr",
        addr = addr,
    );
    let fun_addr = unsafe { *(addr as *const usize) };
    let maybe_builtin = forth
        .data_space
        .builtin_addrs
        .iter()
        .find(|(a, _)| *a == fun_addr);
    // TODO: Setup debugging & tracing facilities.
    // match maybe_builtin {
    //     None => println!("Executing fun at {:#x}", fun_addr),
    //     Some((_, name)) => println!("Executing '{}' ({:#x})", name, fun_addr),
    // }
    assert!(maybe_builtin.is_some());
    let fun: fn(&mut ForthMachine) = unsafe { std::mem::transmute(fun_addr as *const u8) };
    fun(forth)
}

fn next(forth: &mut ForthMachine) {
    if forth.instruction_addr == 0 {
        return;
    }
    assert!(forth
        .data_space
        .is_valid_ptr(forth.instruction_addr as *const usize));
    let def_addr = unsafe { *(forth.instruction_addr as *const usize) };
    forth.instruction_addr = forth.instruction_addr.checked_add(PTR_SIZE).unwrap();
    // println!("curr_def_addr at {:#x}", def_addr);
    // println!("instruction_addr at {:#x}", forth.instruction_addr);
    exec_fun_indirect(def_addr, forth);
}

fn push_instruction(data_space: &mut DataSpace, def_addr: usize) -> usize {
    assert!(is_aligned(data_space.memory.current as usize, PTR_SIZE));
    let ptr = data_space.alloc(PTR_SIZE).unwrap();
    let instruction_buf = unsafe { std::slice::from_raw_parts_mut(ptr, PTR_SIZE) };
    let mut cursor = std::io::Cursor::new(instruction_buf);
    cursor.write_all(&def_addr.to_ne_bytes()).unwrap();
    cursor.into_inner().as_ptr() as usize
}

fn push_word<'a, I>(data_space: &mut DataSpace, name: &str, flags: usize, words: I)
where
    I: IntoIterator<Item = &'a str>,
{
    assert!(data_space.push_dict_entry(name, flags));
    push_instruction(data_space, docol as usize);
    for word in words {
        match data_space.find_entry(word) {
            Some(entry) => {
                push_instruction(data_space, entry.definition_addr());
            }
            None => {
                let num: isize = word.parse().unwrap();
                // push_instruction(
                //     data_space,
                //     data_space.find_entry("LIT").unwrap().definition_addr(),
                // );
                push_instruction(data_space, num as usize);
            }
        };
    }
    let def_addr = data_space.find_entry("EXIT").unwrap().definition_addr();
    push_instruction(data_space, def_addr);
}

fn set_instructions<'a, I>(forth: &mut ForthMachine, words: I)
where
    I: IntoIterator<Item = &'a str>,
{
    assert!(forth.data_space.align());
    for word in words {
        let def_addr = forth.data_space.find_entry(word).unwrap().definition_addr();
        let instruction_addr = push_instruction(&mut forth.data_space, def_addr);
        if forth.instruction_addr == 0 {
            forth.instruction_addr = instruction_addr;
        }
    }
    let def_addr = forth
        .data_space
        .find_entry("BYE")
        .unwrap()
        .definition_addr();
    push_instruction(&mut forth.data_space, def_addr);
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
///     $ cat FORTH_FILE | rforth
///
/// If you want to continue running the interpreter, use
///
///     $ cat FORTH_FILE - | rforth
struct CliArgs {
    /// Size of total memory used for Forth's data space.
    ///
    /// You may suffix the number with one of 'b', 'k', 'M' & 'G' to specify
    /// the size unit. Omitting the suffix defaults to 'k', i.e. kilobytes.
    #[arg(long, default_value = "4k", value_parser = parse_memsize)]
    data_space_size: usize,
    /// Maximum number of cells on the data stack.
    #[arg(long, default_value_t = 256)]
    data_stack_size: usize,
    /// Maximum number of cells on the return stack.
    #[arg(long, default_value_t = 256)]
    return_stack_size: usize,
}

fn main() {
    let cli_args = CliArgs::parse();
    println!("Welcome to rForth");
    println!(
        "data_space_size = {} bytes",
        fmt_memsize(cli_args.data_space_size)
    );
    println!("data_stack_size = {} cells", cli_args.data_stack_size);
    println!("return_stack_size = {} cells", cli_args.return_stack_size);
    let mut forth = ForthMachine::new(
        DataSpace::with_size(cli_args.data_space_size),
        Vec::with_capacity(cli_args.data_stack_size),
        Vec::with_capacity(cli_args.return_stack_size),
    );
    push_word(
        &mut forth.data_space,
        ":",
        0,
        [
            "WORD",
            "CREATE",
            "LIT",
            &(docol as usize).to_string(),
            ",",
            "LATEST",
            /* "@", */ "HIDDEN",
            "]",
        ],
    );
    push_word(
        &mut forth.data_space,
        ";",
        WordFlag::Immediate as usize,
        ["LIT", "EXIT", ",", "LATEST", /* "@", */ "HIDDEN", "["],
    );
    push_word(
        &mut forth.data_space,
        "QUIT",
        0,
        ["RS-CLEAR", "INTERPRET", "BRANCH", "-16"],
    );
    set_instructions(&mut forth, ["QUIT"]);
    while forth.instruction_addr != 0 {
        next(&mut forth);
    }
    println!("\nBye!");
    dbg!(forth.data_stack);
    dbg!(forth.return_stack);
}
