use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use tokenizers::models::unigram::Unigram;

use crate::{set_last_error, set_status};

/// Represents a vocabulary item with a token and its score.
#[repr(C)]
pub struct VocabItem {
    pub token: *const c_char,
    pub score: f64,
}

/// Creates a Unigram model from a vocabulary with scores.
///
/// # Parameters
/// - `vocab`: Array of VocabItem (token, score pairs)
/// - `vocab_len`: Number of items in the vocab array
/// - `unk_id`: ID of the unknown token (optional, pass null for None)
/// - `byte_fallback`: Whether to use byte fallback for unknown characters
/// - `status`: Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created Unigram model, or NULL on error
///
/// # Notes
/// Unigram is a probabilistic tokenization model commonly used by SentencePiece.
/// It uses scores to determine the most likely tokenization of a sequence.
#[no_mangle]
pub extern "C" fn tokenizers_unigram_new(
    vocab: *const VocabItem,
    vocab_len: usize,
    unk_id: *const usize,
    byte_fallback: bool,
    status: *mut i32,
) -> *mut Unigram {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Validate vocab pointer
    if vocab.is_null() {
        set_last_error("Vocab cannot be null");
        set_status(status, -2);
        return ptr::null_mut();
    }

    if vocab_len == 0 {
        set_last_error("Vocab cannot be empty");
        set_status(status, -3);
        return ptr::null_mut();
    }

    // Convert vocab items to Vec<(String, f64)>
    let vocab_slice = unsafe { slice::from_raw_parts(vocab, vocab_len) };
    let mut vocab_vec = Vec::with_capacity(vocab_len);

    for item in vocab_slice {
        if item.token.is_null() {
            set_last_error("Vocab item token cannot be null");
            set_status(status, -4);
            return ptr::null_mut();
        }

        let token_str = match unsafe { CStr::from_ptr(item.token) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("Invalid UTF-8 in vocab item token");
                set_status(status, -5);
                return ptr::null_mut();
            }
        };

        vocab_vec.push((token_str, item.score));
    }

    // Get optional unk_id
    let unk_id_value = if !unk_id.is_null() {
        Some(unsafe { *unk_id })
    } else {
        None
    };

    // Create Unigram model using the from() method (matches Python bindings)
    match Unigram::from(vocab_vec, unk_id_value, byte_fallback) {
        Ok(unigram) => {
            set_status(status, 0);
            Box::into_raw(Box::new(unigram))
        }
        Err(e) => {
            set_last_error(&format!("Failed to create Unigram model: {}", e));
            set_status(status, -6);
            ptr::null_mut()
        }
    }
}

/// Frees a Unigram model.
///
/// # Parameters
/// - `model`: Pointer to the Unigram model to free
///
/// # Safety
/// The model pointer must have been created by `tokenizers_unigram_new`.
/// After calling this function, the pointer is invalid and must not be used.
#[no_mangle]
pub extern "C" fn tokenizers_unigram_free(model: *mut Unigram) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}
