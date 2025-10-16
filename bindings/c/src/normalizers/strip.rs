use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::Strip;
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new Strip normalizer.
///
/// Strips whitespace from the left and/or right side of the input text.
///
/// # Arguments
/// * `left` - Whether to strip whitespace from the left side
/// * `right` - Whether to strip whitespace from the right side
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created Strip normalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_strip_normalizer_free()
/// - At least one of `left` or `right` should be true for the normalizer to have effect
///
/// # Example (C)
/// ```c
/// int status;
/// // Strip whitespace from both sides (trim)
/// void* normalizer = tokenizers_strip_normalizer_new(true, true, &status);
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_strip_normalizer_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_strip_normalizer_new(
    left: bool,
    right: bool,
    status: *mut i32,
) -> *mut Strip {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Create the Strip normalizer
    let normalizer = Strip::new(left, right);

    // Success
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string using a Strip normalizer.
///
/// Removes whitespace from the left and/or right side of the input.
///
/// # Arguments
/// * `normalizer` - Pointer to the Strip normalizer
/// * `input` - Input string to normalize (UTF-8 encoded, null-terminated)
/// * `output` - Output buffer for the normalized string
/// * `output_len` - Size of the output buffer
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Length of the normalized string (excluding null terminator), or 0 on error
#[no_mangle]
pub extern "C" fn tokenizers_strip_normalizer_normalize_str(
    normalizer: *const Strip,
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

/// Frees a Strip normalizer instance.
///
/// # Arguments
/// * `normalizer` - Pointer to the Strip normalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_strip_normalizer_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
#[no_mangle]
pub extern "C" fn tokenizers_strip_normalizer_free(normalizer: *mut Strip) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
