use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Opaque pointer to a Tokenizer instance
pub struct CTokenizer {
    inner: tokenizers::Tokenizer,
}

/// Creates a tokenizer from a JSON file
/// Returns null pointer on error
#[no_mangle]
pub extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut CTokenizer {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match tokenizers::Tokenizer::from_file(path_str) {
        Ok(tokenizer) => Box::into_raw(Box::new(CTokenizer { inner: tokenizer })),
        Err(_) => ptr::null_mut(),
    }
}

/// Creates a tokenizer from a JSON string
/// Returns null pointer on error
#[no_mangle]
pub extern "C" fn tokenizer_from_string(json: *const c_char) -> *mut CTokenizer {
    if json.is_null() {
        return ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(json) };
    let json_bytes = match c_str.to_bytes().to_vec() {
        bytes => bytes,
    };

    match tokenizers::Tokenizer::from_bytes(json_bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(CTokenizer { inner: tokenizer })),
        Err(_) => ptr::null_mut(),
    }
}

/// Encodes a text string to token IDs
/// Returns a pointer to an array of token IDs
/// The caller must free the returned array using tokenizer_free_ids
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    tokenizer: *const CTokenizer,
    text: *const c_char,
    add_special_tokens: bool,
    out_len: *mut usize,
) -> *mut u32 {
    if tokenizer.is_null() || text.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &(*tokenizer).inner };
    let c_str = unsafe { CStr::from_ptr(text) };
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match tokenizer.encode(text_str, add_special_tokens) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            let len = ids.len();
            unsafe {
                *out_len = len;
            }
            
            let mut result = Vec::with_capacity(len);
            result.extend_from_slice(ids);
            
            let ptr = result.as_mut_ptr();
            std::mem::forget(result);
            ptr
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Decodes token IDs to text
/// Returns a C string that must be freed with tokenizer_free_string
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    tokenizer: *const CTokenizer,
    ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
) -> *mut c_char {
    if tokenizer.is_null() || ids.is_null() {
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &(*tokenizer).inner };
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len) };

    match tokenizer.decode(ids_slice, skip_special_tokens) {
        Ok(text) => match CString::new(text) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Gets the vocabulary size
#[no_mangle]
pub extern "C" fn tokenizer_get_vocab_size(tokenizer: *const CTokenizer) -> usize {
    if tokenizer.is_null() {
        return 0;
    }
    let tokenizer = unsafe { &(*tokenizer).inner };
    tokenizer.get_vocab_size(true)
}

/// Frees a tokenizer instance
#[no_mangle]
pub extern "C" fn tokenizer_free(tokenizer: *mut CTokenizer) {
    if !tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(tokenizer));
        }
    }
}

/// Frees a token IDs array returned by tokenizer_encode
#[no_mangle]
pub extern "C" fn tokenizer_free_ids(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(ids, len, len));
        }
    }
}

/// Frees a string returned by tokenizer_decode
#[no_mangle]
pub extern "C" fn tokenizer_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

/// Gets the last error message (thread-local)
/// Returns null if no error
#[no_mangle]
pub extern "C" fn tokenizer_get_last_error() -> *const c_char {
    // TODO: Implement thread-local error storage
    ptr::null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_api() {
        // Basic smoke test
        assert!(ptr::null_mut::<CTokenizer>().is_null());
    }
}
