use std::convert::TryFrom;
use std::io::Write;

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

#[derive(Copy, Clone, PartialEq)]
/// Entry in a Forth dictionary
struct DictEntryRef<'a> {
    /// Points to start of the entry in DataSpace.
    ptr: &'a u8,
    /// Past the end adress of DataSpace
    end_of_data_space: usize,
}

impl<'a> DictEntryRef<'a> {
    fn ptr_as_slice(&self) -> Option<&[u8]> {
        let ptr = self.ptr as *const u8;
        // TODO: Check if ptr is below DataSpace
        if ptr as usize >= self.end_of_data_space {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(ptr, self.end_of_data_space - ptr as usize) })
        }
    }

    pub fn prev(&self) -> Option<DictEntryRef<'a>> {
        let buf_slice = self.ptr_as_slice().unwrap();
        let prev_bytes = <[u8; PTR_SIZE]>::try_from(&buf_slice[0..PTR_SIZE]).unwrap();
        let prev_addr = usize::from_ne_bytes(prev_bytes);
        let maybe_prev = unsafe { (prev_addr as *const u8).as_ref() };
        maybe_prev.map(|prev| DictEntryRef {
            ptr: prev,
            end_of_data_space: self.end_of_data_space,
        })
    }

    pub fn name(&self) -> &str {
        let (len_buf, name_buf) = self.ptr_as_slice().unwrap()[PTR_SIZE..].split_at(PTR_SIZE);
        let len = usize::from_ne_bytes(<[u8; PTR_SIZE]>::try_from(len_buf).unwrap());
        std::str::from_utf8(&name_buf[0..len]).unwrap()
    }

    pub fn definition_addr(&self) -> usize {
        let addr = align_addr(self.name().as_ptr() as usize + self.name().len());
        assert!(addr < self.end_of_data_space);
        addr
    }
}

impl std::fmt::Debug for DictEntryRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DictEntryRef {{ ptr: {:p}, end_of_data_space: {:p}, prev: {:p}, name: {} }}",
            self.ptr as *const u8,
            self.end_of_data_space as *const u8,
            self.prev().map_or(std::ptr::null(), |e| e.ptr),
            self.name()
        )
    }
}

/// Buffer for Forth's data space.
///
/// Data space includes dictionary entries and user allocations.
#[derive(Debug)]
struct DataSpace {
    layout: std::alloc::Layout,
    ptr: *mut u8,
    current: *mut u8,
    dict_head: *const u8,
    builtin_addrs: Vec<usize>,
}

impl DataSpace {
    pub fn with_size(bytes: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, 8).unwrap();
        let ptr;
        unsafe {
            ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
        }
        let mut builtin_addrs = Vec::with_capacity(64);
        builtin_addrs.push(docol as usize);
        DataSpace {
            layout: layout,
            ptr: ptr,
            current: ptr,
            dict_head: std::ptr::null(),
            builtin_addrs: builtin_addrs,
        }
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn unused(&self) -> usize {
        let end = self.ptr as usize + self.size();
        end - self.current as usize
    }

    pub fn is_allocd_addr(&self, addr: usize) -> bool {
        is_within(&addr, &(self.ptr as usize), &(self.current as usize))
    }

    pub fn is_valid_ptr<T>(&self, ptr: *const T) -> bool {
        let end = self.current as usize - std::mem::size_of::<T>() + 1;
        is_aligned(ptr as usize, std::mem::align_of::<T>())
            && is_within(&(ptr as usize), &(self.ptr as usize), &end)
    }

    pub fn align(&mut self) {
        let aligned_addr = align_addr(self.current as usize);
        assert!(aligned_addr - self.current as usize <= self.unused());
        self.current = aligned_addr as *mut u8;
    }

    pub fn alloc<'a>(&'a mut self, bytes: usize) -> Option<&'a mut [u8]> {
        if bytes > self.unused() {
            None
        } else {
            unsafe {
                let ret = std::slice::from_raw_parts_mut(self.current, bytes);
                self.current = self.current.add(bytes);
                Some(ret)
            }
        }
    }

    pub fn push_dict_entry(&mut self, name: &str) -> bool {
        let prev_addr = self.dict_head as usize;
        match self.alloc(PTR_SIZE + PTR_SIZE + name.len()) {
            None => false,
            Some(buf) => {
                let mut cursor = std::io::Cursor::new(buf);
                cursor.write_all(&prev_addr.to_ne_bytes()).unwrap();
                cursor.write_all(&name.len().to_ne_bytes()).unwrap();
                cursor.write_all(name.as_bytes()).unwrap();
                self.dict_head = cursor.into_inner().as_ptr();
                true
            }
        }
    }

    pub fn latest_entry<'a>(&'a self) -> Option<DictEntryRef<'a>> {
        let maybe_entry = unsafe { self.dict_head.as_ref() };
        maybe_entry.map(|ptr| DictEntryRef {
            ptr: ptr,
            end_of_data_space: self.ptr as usize + self.size(),
        })
    }

    pub fn find_entry<'a>(&'a self, name: &str) -> Option<DictEntryRef<'a>> {
        let mut maybe_entry = self.latest_entry();
        while let Some(entry) = maybe_entry {
            if entry.name() == name {
                return maybe_entry;
            } else {
                maybe_entry = entry.prev();
            }
        }
        None
    }

    pub fn is_builtin_addr(&self, addr: usize) -> bool {
        self.builtin_addrs.contains(&addr)
    }

    pub fn push_builtin_word(&mut self, name: &str, word_fn: fn(&mut ForthMachine)) {
        self.push_dict_entry(name);
        self.align();
        {
            let fn_addr = word_fn as usize;
            if !self.is_builtin_addr(fn_addr) {
                self.builtin_addrs.push(fn_addr);
            }
            let bytes = self.alloc(PTR_SIZE).unwrap();
            let mut cursor = std::io::Cursor::new(bytes);
            cursor.write_all(&fn_addr.to_ne_bytes()).unwrap();
        }
    }
}

impl Drop for DataSpace {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

#[test]
fn test_data_space_align() {
    let mut data_space = DataSpace::with_size(1024);
    assert_eq!(data_space.unused(), 1024);
    data_space.align();
    assert_eq!(data_space.unused(), 1024);
    data_space.alloc(8);
    assert_eq!(data_space.unused(), 1024 - 8);
    data_space.align();
    assert_eq!(data_space.unused(), 1024 - 8);
    data_space.alloc(1);
    assert_eq!(data_space.unused(), 1024 - 9);
    data_space.align();
    assert_eq!(data_space.unused(), 1024 - 16);
    data_space.alloc(7);
    assert_eq!(data_space.unused(), 1024 - 16 - 7);
    data_space.align();
    assert_eq!(data_space.unused(), 1024 - 16 - 8);
}

#[test]
fn test_data_space_find_entry() {
    let mut data_space = DataSpace::with_size(1024);
    assert!(data_space.find_entry("DUP").is_none());
    data_space.push_dict_entry("DUP");
    {
        let maybe_dup = data_space.find_entry("DUP");
        assert!(maybe_dup.is_some());
        assert_eq!(maybe_dup.unwrap().name(), "DUP");
        assert_eq!(maybe_dup, data_space.latest_entry());
    }
    data_space.push_dict_entry("SWAP");
    {
        let maybe_dup = data_space.find_entry("DUP");
        assert!(maybe_dup.is_some());
        assert_eq!(maybe_dup.unwrap().name(), "DUP");
        assert_eq!(maybe_dup, data_space.latest_entry().unwrap().prev());
    }
    data_space.push_dict_entry("DUP");
    {
        let maybe_dup = data_space.find_entry("DUP");
        assert!(maybe_dup.is_some());
        assert_eq!(maybe_dup.unwrap().name(), "DUP");
        assert_eq!(maybe_dup, data_space.latest_entry());
        let orig_dup = data_space.latest_entry().unwrap().prev().unwrap().prev();
        assert_eq!(maybe_dup.unwrap().name(), orig_dup.unwrap().name());
        assert_ne!(maybe_dup, orig_dup);
    }
}

#[derive(Debug)]
struct ForthMachine {
    data_space: DataSpace,
    data_stack: Vec<isize>,
    return_stack: Vec<isize>,
    curr_def_addr: usize,
    instruction_addr: usize,
}

fn bye_builtin(forth: &mut ForthMachine) {
    forth.instruction_addr = 0;
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

fn docol(forth: &mut ForthMachine) {
    forth.return_stack.push(forth.instruction_addr as isize);
    forth.instruction_addr = forth.curr_def_addr.checked_add(PTR_SIZE).unwrap();
}

fn exit_builtin(forth: &mut ForthMachine) {
    forth.instruction_addr = forth.return_stack.pop().unwrap() as usize;
}

fn add_builtins(data_space: &mut DataSpace) {
    data_space.push_builtin_word("BYE", bye_builtin);
    data_space.push_builtin_word("DUP", dup_builtin);
    data_space.push_builtin_word("SWAP", swap_builtin);
    data_space.push_builtin_word("EXIT", exit_builtin);
}

impl ForthMachine {
    pub fn new(
        mut data_space: DataSpace,
        data_stack: Vec<isize>,
        return_stack: Vec<isize>,
    ) -> Self {
        add_builtins(&mut data_space);
        ForthMachine {
            data_space: data_space,
            data_stack: data_stack,
            return_stack: return_stack,
            instruction_addr: 0,
            curr_def_addr: 0,
        }
    }
}

fn exec_fun_indirect(addr: usize, forth: &mut ForthMachine) {
    assert!(forth.data_space.is_valid_ptr(addr as *const usize));
    let fun_addr = unsafe { *(addr as *const usize) };
    assert!(forth.data_space.is_builtin_addr(fun_addr));
    let fun: fn(&mut ForthMachine) = unsafe { std::mem::transmute(fun_addr as *const u8) };
    return fun(forth);
}

fn next(forth: &mut ForthMachine) {
    if forth.instruction_addr == 0 {
        return;
    }
    assert!(forth
        .data_space
        .is_valid_ptr(forth.instruction_addr as *const usize));
    let def_addr = unsafe { *(forth.instruction_addr as *const usize) };
    forth.curr_def_addr = def_addr;
    forth.instruction_addr = forth.instruction_addr.checked_add(PTR_SIZE).unwrap();
    exec_fun_indirect(def_addr, forth);
}

fn interpret(forth: &mut ForthMachine, word: &str) {
    let maybe_definition_addr = forth
        .data_space
        .find_entry(word)
        .map(|entry| entry.definition_addr());
    if let Some(definition_addr) = maybe_definition_addr {
        exec_fun_indirect(definition_addr, forth);
    } else {
        forth.instruction_addr = 0;
    }
}

fn compile_word(data_space: &mut DataSpace, word: &str) -> usize {
    data_space
        .find_entry(word)
        .map(|entry| entry.definition_addr())
        .unwrap()
}

fn push_instruction(data_space: &mut DataSpace, def_addr: usize) -> usize {
    assert!(is_aligned(data_space.current as usize, PTR_SIZE));
    let instruction_buf = data_space.alloc(PTR_SIZE).unwrap();
    let mut cursor = std::io::Cursor::new(instruction_buf);
    cursor.write_all(&def_addr.to_ne_bytes()).unwrap();
    cursor.into_inner().as_ptr() as usize
}

fn push_word<'a, I>(data_space: &mut DataSpace, name: &str, words: I)
where
    I: Iterator<Item = &'a str>,
{
    data_space.push_dict_entry(name);
    data_space.align();
    push_instruction(data_space, docol as usize);
    for word in words {
        let def_addr = compile_word(data_space, word);
        push_instruction(data_space, def_addr);
    }
    let def_addr = compile_word(data_space, "EXIT");
    push_instruction(data_space, def_addr);
}

fn set_instructions<'a, I>(forth: &mut ForthMachine, words: I)
where
    I: Iterator<Item = &'a str>,
{
    forth.data_space.align();
    for word in words {
        let def_addr = compile_word(&mut forth.data_space, word);
        let instruction_addr = push_instruction(&mut forth.data_space, def_addr);
        if forth.instruction_addr == 0 {
            forth.instruction_addr = instruction_addr;
        }
    }
    let def_addr = compile_word(&mut forth.data_space, "BYE");
    push_instruction(&mut forth.data_space, def_addr);
}

fn main() {
    let mut forth = ForthMachine::new(
        DataSpace::with_size(1024),
        Vec::with_capacity(256),
        Vec::with_capacity(256),
    );
    forth.data_stack.push(1);
    forth.data_stack.push(2);
    push_word(
        &mut forth.data_space,
        "GO",
        ["SWAP", "DUP"].iter().map(|s| *s),
    );
    set_instructions(&mut forth, ["GO"].iter().map(|s| *s));
    while forth.instruction_addr != 0 {
        next(&mut forth);
    }
    println!("{:?}", forth);
}
