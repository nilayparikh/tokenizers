use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::StripAccents;
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new StripAccents normalizer.
///
/// This normalizer removes all combining marks (accents, diacritics) from the input text.
/// It preserves the base characters while stripping accent marks, making it useful for
/// text normalization in languages with diacritics.
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created StripAccents normalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_strip_accents_free()
/// - This normalizer has no configurable parameters
/// - Works best when combined with Unicode normalization (NFD/NFKD)
/// - Preserves non-Latin scripts (e.g., Chinese, Arabic) without modification
///
/// # Example (C)
/// ```c
/// int status;
/// void* normalizer = tokenizers_strip_accents_new(&status);
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_strip_accents_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_strip_accents_new(status: *mut i32) -> *mut StripAccents {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Create the StripAccents normalizer (no parameters needed)
    let normalizer = StripAccents;

    // Success
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string by stripping accents and diacritics.
///
/// This function applies the StripAccents normalizer to the input string,
/// removing all combining marks while preserving base characters.
///
/// # Arguments
/// * `normalizer` - Pointer to the StripAccents normalizer
/// * `input` - Input string to normalize (UTF-8 encoded, null-terminated)
/// * `output` - Output buffer for the normalized string
/// * `output_len` - Size of the output buffer
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Length of the normalized string (excluding null terminator), or 0 on error
///
/// # Notes
/// - If output buffer is NULL or too small, returns required size
/// - Normalized string is null-terminated
/// - Status codes: 0 = success, -1 = invalid args, -2 = buffer too small, -3 = normalization failed
///
/// # Example (C)
/// ```c
/// // Input: "café" -> Output: "cafe"
/// char output[256];
/// int status;
/// size_t len = tokenizers_strip_accents_normalize_str(normalizer, "café", output, 256, &status);
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_strip_accents_normalize_str(
    normalizer: *const StripAccents,
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

/// Frees a StripAccents normalizer.
///
/// This function safely deallocates memory for the StripAccents normalizer.
///
/// # Arguments
/// * `normalizer` - Pointer to the StripAccents normalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_strip_accents_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
///
/// # Example (C)
/// ```c
/// tokenizers_strip_accents_free(normalizer);
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_strip_accents_free(normalizer: *mut StripAccents) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
