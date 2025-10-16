use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new BERT normalizer.
///
/// BertNormalizer takes care of normalizing raw text before giving it to a BERT model.
/// This includes cleaning the text, handling accents, Chinese chars, and lowercasing.
///
/// # Arguments
/// * `clean_text` - Whether to clean the text by removing control characters and normalizing whitespace
/// * `handle_chinese_chars` - Whether to handle Chinese chars by putting spaces around them
/// * `strip_accents` - Pointer to bool for accent stripping, or NULL to auto-determine from lowercase
/// * `lowercase` - Whether to convert text to lowercase
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created BertNormalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_bert_normalizer_free()
/// - If strip_accents is NULL, it will be automatically determined based on lowercase setting
///   (following the original BERT implementation)
/// - clean_text removes control characters and normalizes all whitespace to single spaces
/// - handle_chinese_chars adds spaces around CJK characters for better tokenization
///
/// # Example (C)
/// ```c
/// int status;
/// bool strip = true;
/// void* normalizer = tokenizers_bert_normalizer_new(
///     true,   // clean_text
///     true,   // handle_chinese_chars
///     &strip, // strip_accents
///     true,   // lowercase
///     &status
/// );
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_bert_normalizer_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_bert_normalizer_new(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: *const bool,
    lowercase: bool,
    status: *mut i32,
) -> *mut BertNormalizer {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Handle optional strip_accents parameter
    let strip_accents_opt = if strip_accents.is_null() {
        None
    } else {
        Some(unsafe { *strip_accents })
    };

    // Create the BertNormalizer
    let normalizer = BertNormalizer::new(
        clean_text,
        handle_chinese_chars,
        strip_accents_opt,
        lowercase,
    );

    // Success
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string using a BERT normalizer.
///
/// Applies BERT-specific normalization to the input text.
///
/// # Arguments
/// * `normalizer` - Pointer to the BertNormalizer
/// * `input` - Input string to normalize (UTF-8 encoded, null-terminated)
/// * `output` - Output buffer for the normalized string
/// * `output_len` - Size of the output buffer
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Length of the normalized string (excluding null terminator), or 0 on error
///
/// # Notes
/// - If output buffer is NULL, returns required size
/// - If buffer is too small, returns required size and sets status to -2
/// - Normalized string is null-terminated
#[no_mangle]
pub extern "C" fn tokenizers_bert_normalizer_normalize_str(
    normalizer: *const BertNormalizer,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    // Validate pointers
    if status.is_null() {
        return 0;
    }

    set_status(status, -1);

    if normalizer.is_null() || input.is_null() {
        set_last_error("Normalizer and input cannot be null");
        return 0;
    }

    // Convert input string
    let input_str = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid UTF-8 input: {}", e));
            return 0;
        }
    };

    // Create normalized string and apply normalization
    let mut normalized = NormalizedString::from(input_str);
    let norm_ref = unsafe { &*normalizer };
    
    if let Err(e) = norm_ref.normalize(&mut normalized) {
        set_last_error(&format!("Normalization failed: {}", e));
        set_status(status, -3);
        return 0;
    }

    let result = normalized.get();
    let result_bytes = result.as_bytes();
    let required_len = result_bytes.len() + 1; // +1 for null terminator

    // If output is null, just return required size
    if output.is_null() {
        set_status(status, 0);
        return required_len;
    }

    // Check if buffer is large enough
    if output_len < required_len {
        set_last_error(&format!(
            "Output buffer too small: need {} bytes, got {}",
            required_len, output_len
        ));
        set_status(status, -2);
        return required_len;
    }

    // Copy result to output buffer
    unsafe {
        ptr::copy_nonoverlapping(result_bytes.as_ptr(), output as *mut u8, result_bytes.len());
        *output.add(result_bytes.len()) = 0; // Null terminator
    }

    set_status(status, 0);
    result_bytes.len()
}

/// Frees a BertNormalizer instance.
///
/// # Arguments
/// * `normalizer` - Pointer to the BertNormalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_bert_normalizer_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
#[no_mangle]
pub extern "C" fn tokenizers_bert_normalizer_free(normalizer: *mut BertNormalizer) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
