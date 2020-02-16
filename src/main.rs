use std::convert::TryFrom;
use std::io::Write;

const PTR_SIZE: usize = std::mem::size_of::<usize>();

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
struct DataSpace {
    layout: std::alloc::Layout,
    ptr: *mut u8,
    current: *mut u8,
    dict_head: *const u8,
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
        DataSpace {
            layout: layout,
            ptr: ptr,
            current: ptr,
            dict_head: std::ptr::null(),
        }
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn unused(&self) -> usize {
        let end = self.ptr as usize + self.size();
        end - self.current as usize
    }

    pub fn align(&mut self) {
        assert_eq!(std::mem::align_of::<usize>(), PTR_SIZE);
        let addr = self.current as usize;
        let aligned_addr = (addr + (PTR_SIZE - 1)) & (!PTR_SIZE + 1);
        assert_eq!(aligned_addr % PTR_SIZE, 0);
        assert!(aligned_addr >= self.current as usize);
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

struct ForthMachine {
    data_space: DataSpace,
    data_stack: Vec<isize>,
    return_stack: Vec<isize>,
}

fn main() {
    let mut forth_machine = ForthMachine {
        data_space: DataSpace::with_size(1024),
        data_stack: Vec::with_capacity(256),
        return_stack: Vec::with_capacity(256),
    };
    let data_space = &mut forth_machine.data_space;
    println!("Data space ptr: {:p}", data_space.ptr);
    println!("Unused data space: {} bytes", data_space.unused());
    data_space.push_dict_entry("DUP");
    let entry1 = data_space.latest_entry().unwrap();
    println!("Dict entry 1 {:?}", entry1);
    println!("Unused data space: {} bytes", data_space.unused());
    data_space.push_dict_entry("SWAP");
    let entry2 = data_space.latest_entry().unwrap();
    println!("Dict entry 2 {:?}", entry2);
}
