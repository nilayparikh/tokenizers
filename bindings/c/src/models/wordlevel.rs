use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use tokenizers::models::wordlevel::WordLevel;

use crate::{set_last_error, set_status};

/// Creates a WordLevel model from a vocabulary JSON file.
///
/// # Parameters
/// - `vocab_path`: Path to the vocab.json file
/// - `unk_token`: The unknown token (optional, pass NULL for no unknown token)
/// - `status`: Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created WordLevel model, or NULL on error
///
/// # Notes
/// WordLevel is the simplest tokenization model - it directly maps tokens to IDs
/// using a vocabulary dictionary. No subword splitting is performed.
#[no_mangle]
pub extern "C" fn tokenizers_wordlevel_from_file(
    vocab_path: *const c_char,
    unk_token: *const c_char,
    status: *mut i32,
) -> *mut WordLevel {
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

    // Read vocab file using WordLevel::read_file() (same pattern as Python bindings)
    let vocab = match WordLevel::read_file(vocab_path_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("Failed to read WordLevel vocab file: {}", e));
            set_status(status, -5);
            return ptr::null_mut();
        }
    };

    // Build WordLevel model
    let mut builder = WordLevel::builder().vocab(vocab);

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
            _ => {} // Empty string, use default (no unknown token)
        }
    }

    // Build the model
    match builder.build() {
        Ok(wordlevel) => {
            set_status(status, 0);
            Box::into_raw(Box::new(wordlevel))
        }
        Err(e) => {
            set_last_error(&format!("Failed to build WordLevel model: {}", e));
            set_status(status, -7);
            ptr::null_mut()
        }
    }
}

/// Frees a WordLevel model.
///
/// # Parameters
/// - `model`: Pointer to the WordLevel model to free
///
/// # Safety
/// The model pointer must have been created by `tokenizers_wordlevel_from_file`.
/// After calling this function, the pointer is invalid and must not be used.
#[no_mangle]
pub extern "C" fn tokenizers_wordlevel_free(model: *mut WordLevel) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}
