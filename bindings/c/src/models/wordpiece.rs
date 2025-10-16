use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use tokenizers::models::wordpiece::WordPiece;

use crate::{set_last_error, set_status};

/// Creates a WordPiece model from a vocabulary file.
///
/// # Parameters
/// - `vocab_path`: Path to the vocab.txt file
/// - `unk_token`: The unknown token (optional, pass NULL for default "[UNK]")
/// - `max_input_chars_per_word`: Maximum characters per word (0 for default 100)
/// - `continuing_subword_prefix`: Prefix for continuing subwords (optional, pass NULL for default "##")
/// - `status`: Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created WordPiece model, or NULL on error
#[no_mangle]
pub extern "C" fn tokenizers_wordpiece_from_file(
    vocab_path: *const c_char,
    unk_token: *const c_char,
    max_input_chars_per_word: usize,
    continuing_subword_prefix: *const c_char,
    status: *mut i32,
) -> *mut WordPiece {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Validate and convert vocab_path
    if vocab_path.is_null() {
        set_last_error("Vocab path cannot be null");
        set_status(status, -2);
        return ptr::null_mut();
    }

    let vocab_path_str = match unsafe { CStr::from_ptr(vocab_path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in vocab path");
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    if vocab_path_str.is_empty() {
        set_last_error("Vocab path cannot be empty");
        set_status(status, -4);
        return ptr::null_mut();
    }

    // Read vocab file using WordPiece::read_file() (same pattern as Python bindings)
    let vocab = match WordPiece::read_file(vocab_path_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("Failed to read WordPiece vocab file: {}", e));
            set_status(status, -5);
            return ptr::null_mut();
        }
    };

    // Build WordPiece model with options
    let mut builder = WordPiece::builder().vocab(vocab);

    // Set unknown token (optional)
    if !unk_token.is_null() {
        match unsafe { CStr::from_ptr(unk_token) }.to_str() {
            Ok(token) if !token.is_empty() => {
                builder = builder.unk_token(token.to_string());
            }
            Err(_) => {
                set_last_error("Invalid UTF-8 in unknown token");
                set_status(status, -6);
                return ptr::null_mut();
            }
            _ => {} // Empty string, use default
        }
    }

    // Set max_input_chars_per_word (0 means use default)
    if max_input_chars_per_word > 0 {
        builder = builder.max_input_chars_per_word(max_input_chars_per_word);
    }

    // Set continuing_subword_prefix (optional)
    if !continuing_subword_prefix.is_null() {
        match unsafe { CStr::from_ptr(continuing_subword_prefix) }.to_str() {
            Ok(prefix) if !prefix.is_empty() => {
                builder = builder.continuing_subword_prefix(prefix.to_string());
            }
            Err(_) => {
                set_last_error("Invalid UTF-8 in continuing subword prefix");
                set_status(status, -7);
                return ptr::null_mut();
            }
            _ => {} // Empty string, use default
        }
    }

    // Build the model
    match builder.build() {
        Ok(wordpiece) => {
            set_status(status, 0);
            Box::into_raw(Box::new(wordpiece))
        }
        Err(e) => {
            set_last_error(&format!("Failed to build WordPiece model: {}", e));
            set_status(status, -8);
            ptr::null_mut()
        }
    }
}

/// Frees a WordPiece model.
///
/// # Parameters
/// - `model`: Pointer to the WordPiece model to free
///
/// # Safety
/// The model pointer must have been created by `tokenizers_wordpiece_from_file`.
/// After calling this function, the pointer is invalid and must not be used.
#[no_mangle]
pub extern "C" fn tokenizers_wordpiece_free(model: *mut WordPiece) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}

// Note: wordpiece_new (constructor) is not implemented because std::collections::HashMap 
// cannot be easily converted to ahash::AHashMap. Use wordpiece_from_file instead, which 
// uses WordPiece::read_file() that returns the correct AHashMap type.
