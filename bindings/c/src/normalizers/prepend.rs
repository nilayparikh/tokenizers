use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::Prepend;
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new Prepend normalizer.
///
/// Prepends a string to the beginning of the input text.
/// Commonly used to add special prefixes like "▁" (for SentencePiece) or other markers.
///
/// # Arguments
/// * `prepend` - String to prepend (UTF-8 encoded, null-terminated)
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created Prepend normalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_prepend_normalizer_free()
/// - The prepend string is added to every input string that is normalized
///
/// # Example (C)
/// ```c
/// int status;
/// // Prepend "▁" (SentencePiece style)
/// void* normalizer = tokenizers_prepend_normalizer_new("▁", &status);
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_prepend_normalizer_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_prepend_normalizer_new(
    prepend: *const c_char,
    status: *mut i32,
) -> *mut Prepend {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    if prepend.is_null() {
        set_last_error("Prepend string cannot be null");
        return ptr::null_mut();
    }

    // Convert prepend string
    let prepend_str = match unsafe { CStr::from_ptr(prepend) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid UTF-8 prepend string: {}", e));
            return ptr::null_mut();
        }
    };

    // Create the Prepend normalizer
    let normalizer = Prepend::new(prepend_str.to_string());

    // Success
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string using a Prepend normalizer.
///
/// Adds the prepend string to the beginning of the input.
///
/// # Arguments
/// * `normalizer` - Pointer to the Prepend normalizer
/// * `input` - Input string to normalize (UTF-8 encoded, null-terminated)
/// * `output` - Output buffer for the normalized string
/// * `output_len` - Size of the output buffer
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Length of the normalized string (excluding null terminator), or 0 on error
#[no_mangle]
pub extern "C" fn tokenizers_prepend_normalizer_normalize_str(
    normalizer: *const Prepend,
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

/// Frees a Prepend normalizer instance.
///
/// # Arguments
/// * `normalizer` - Pointer to the Prepend normalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_prepend_normalizer_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
#[no_mangle]
pub extern "C" fn tokenizers_prepend_normalizer_free(normalizer: *mut Prepend) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
