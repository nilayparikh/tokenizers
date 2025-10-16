use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;

use tokenizers::models::bpe::{Vocab, BPE};

use crate::{set_last_error, set_status};

/// Create a new BPE model from vocabulary and merges
///
/// # Parameters
/// - `vocab_json`: JSON string containing the vocabulary
/// - `merges_str`: String containing the merges (one per line)
/// - `cache_capacity`: Optional cache capacity (0 = use default)
/// - `dropout`: Optional dropout value (negative = no dropout)
/// - `unk_token`: Optional unknown token string
/// - `continuing_subword_prefix`: Optional prefix for continuing subwords
/// - `end_of_word_suffix`: Optional end-of-word suffix
/// - `fuse_unk`: Whether to fuse unknown tokens
/// - `byte_fallback`: Whether to use byte fallback
/// - `status`: Output status code (0 = success, non-zero = error)
///
/// # Returns
/// Opaque pointer to the BPE model, or null on error
#[no_mangle]
pub extern "C" fn tokenizers_bpe_create(
    vocab_json: *const c_char,
    merges_str: *const c_char,
    cache_capacity: usize,
    dropout: f32,
    unk_token: *const c_char,
    continuing_subword_prefix: *const c_char,
    end_of_word_suffix: *const c_char,
    fuse_unk: bool,
    byte_fallback: bool,
    status: *mut c_int,
) -> *mut BPE {
    // Parse vocab JSON
    let vocab_cstr = unsafe { CStr::from_ptr(vocab_json) };
    let vocab_str = match vocab_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid vocab JSON string: {}", e));
            set_status(status, -1);
            return ptr::null_mut();
        }
    };

    let vocab: Vocab = match serde_json::from_str(vocab_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("Failed to parse vocab JSON: {}", e));
            set_status(status, -2);
            return ptr::null_mut();
        }
    };

    // Parse merges
    let merges_cstr = unsafe { CStr::from_ptr(merges_str) };
    let merges_content = match merges_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid merges string: {}", e));
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    let mut merges = Vec::new();
    for (i, line) in merges_content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            set_last_error(&format!("Invalid merge at line {}: expected 2 tokens, found {}", i + 1, parts.len()));
            set_status(status, -4);
            return ptr::null_mut();
        }

        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    // Build BPE model
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .fuse_unk(fuse_unk)
        .byte_fallback(byte_fallback);

    if cache_capacity > 0 {
        builder = builder.cache_capacity(cache_capacity);
    }

    if dropout >= 0.0 && dropout <= 1.0 {
        builder = builder.dropout(dropout);
    }

    if !unk_token.is_null() {
        let unk_cstr = unsafe { CStr::from_ptr(unk_token) };
        if let Ok(unk_str) = unk_cstr.to_str() {
            builder = builder.unk_token(unk_str.to_string());
        }
    }

    if !continuing_subword_prefix.is_null() {
        let prefix_cstr = unsafe { CStr::from_ptr(continuing_subword_prefix) };
        if let Ok(prefix_str) = prefix_cstr.to_str() {
            builder = builder.continuing_subword_prefix(prefix_str.to_string());
        }
    }

    if !end_of_word_suffix.is_null() {
        let suffix_cstr = unsafe { CStr::from_ptr(end_of_word_suffix) };
        if let Ok(suffix_str) = suffix_cstr.to_str() {
            builder = builder.end_of_word_suffix(suffix_str.to_string());
        }
    }

    match builder.build() {
        Ok(bpe) => {
            set_status(status, 0);
            Box::into_raw(Box::new(bpe))
        }
        Err(e) => {
            set_last_error(&format!("Failed to build BPE model: {}", e));
            set_status(status, -5);
            ptr::null_mut()
        }
    }
}

/// Create a BPE model from vocabulary and merges files
///
/// # Parameters
/// - `vocab_path`: Path to vocabulary file (JSON)
/// - `merges_path`: Path to merges file
/// - `cache_capacity`: Optional cache capacity (0 = use default)
/// - `dropout`: Optional dropout value (negative = no dropout)
/// - `unk_token`: Optional unknown token string
/// - `continuing_subword_prefix`: Optional prefix for continuing subwords
/// - `end_of_word_suffix`: Optional end-of-word suffix
/// - `fuse_unk`: Whether to fuse unknown tokens
/// - `status`: Output status code (0 = success, non-zero = error)
///
/// # Returns
/// Opaque pointer to the BPE model, or null on error
#[no_mangle]
pub extern "C" fn tokenizers_bpe_from_file(
    vocab_path: *const c_char,
    merges_path: *const c_char,
    cache_capacity: usize,
    dropout: f32,
    unk_token: *const c_char,
    continuing_subword_prefix: *const c_char,
    end_of_word_suffix: *const c_char,
    fuse_unk: bool,
    status: *mut c_int,
) -> *mut BPE {
    // Get vocab path
    let vocab_path_cstr = unsafe { CStr::from_ptr(vocab_path) };
    let vocab_path_str = match vocab_path_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid vocab path: {}", e));
            set_status(status, -1);
            return ptr::null_mut();
        }
    };

    // Get merges path
    let merges_path_cstr = unsafe { CStr::from_ptr(merges_path) };
    let merges_path_str = match merges_path_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid merges path: {}", e));
            set_status(status, -2);
            return ptr::null_mut();
        }
    };

    // Read files using BPE::read_file() just like Python bindings do
    // This properly parses vocab and merges with correct validation
    let (vocab, merges) = match BPE::read_file(vocab_path_str, merges_path_str) {
        Ok((v, m)) => {
            let vocab_map: Vocab = v.into_iter().collect();
            (vocab_map, m)
        }
        Err(e) => {
            set_last_error(&format!("Failed to read BPE files: {}", e));
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    // Build BPE model with vocab and merges
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .fuse_unk(fuse_unk);

    if cache_capacity > 0 {
        builder = builder.cache_capacity(cache_capacity);
    }

    if dropout >= 0.0 && dropout <= 1.0 {
        builder = builder.dropout(dropout);
    }

    if !unk_token.is_null() {
        let unk_cstr = unsafe { CStr::from_ptr(unk_token) };
        if let Ok(unk_str) = unk_cstr.to_str() {
            builder = builder.unk_token(unk_str.to_string());
        }
    }

    if !continuing_subword_prefix.is_null() {
        let prefix_cstr = unsafe { CStr::from_ptr(continuing_subword_prefix) };
        if let Ok(prefix_str) = prefix_cstr.to_str() {
            builder = builder.continuing_subword_prefix(prefix_str.to_string());
        }
    }

    if !end_of_word_suffix.is_null() {
        let suffix_cstr = unsafe { CStr::from_ptr(end_of_word_suffix) };
        if let Ok(suffix_str) = suffix_cstr.to_str() {
            builder = builder.end_of_word_suffix(suffix_str.to_string());
        }
    }

    match builder.build() {
        Ok(bpe) => {
            set_status(status, 0);
            Box::into_raw(Box::new(bpe))
        }
        Err(e) => {
            set_last_error(&format!("Failed to build BPE model: {}", e));
            set_status(status, -4);
            ptr::null_mut()
        }
    }
}

/// Free a BPE model
#[no_mangle]
pub extern "C" fn tokenizers_bpe_free(model: *mut BPE) {
    if !model.is_null() {
        unsafe {
            let _ = Box::from_raw(model);
        }
    }
}
