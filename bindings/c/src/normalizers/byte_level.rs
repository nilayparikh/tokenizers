use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::ByteLevel;
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new ByteLevel normalizer.
///
/// This normalizer converts text to a byte-level representation, which is commonly used
/// in models like GPT-2. It maps each byte to a unique character, allowing the model
/// to handle any input regardless of the Unicode characters used.
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created ByteLevel normalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_byte_level_normalizer_free()
/// - This normalizer has no configurable parameters
/// - Commonly used with GPT-2 and similar models
/// - Maps UTF-8 bytes to a set of visible Unicode characters
///
/// # Example (C)
/// ```c
/// int status;
/// void* normalizer = tokenizers_byte_level_normalizer_new(&status);
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_byte_level_normalizer_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_byte_level_normalizer_new(status: *mut i32) -> *mut ByteLevel {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    // Create the ByteLevel normalizer (no parameters needed)
    let normalizer = ByteLevel::new();

    // Success
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string using byte-level encoding.
///
/// This function applies the ByteLevel normalizer to the input string,
/// converting each byte to a unique visible Unicode character.
///
/// # Arguments
/// * `normalizer` - Pointer to the ByteLevel normalizer
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
/// - Output may be longer than input due to byte-level encoding
///
/// # Example (C)
/// ```c
/// // Input: "Hello" -> Output: "Hello" (with special byte-level encoding for non-ASCII)
/// char output[1024];
/// int status;
/// size_t len = tokenizers_byte_level_normalizer_normalize_str(normalizer, "Hello", output, 1024, &status);
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_byte_level_normalizer_normalize_str(
    normalizer: *const ByteLevel,
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

/// Frees a ByteLevel normalizer.
///
/// This function safely deallocates memory for the ByteLevel normalizer.
///
/// # Arguments
/// * `normalizer` - Pointer to the ByteLevel normalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_byte_level_normalizer_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
///
/// # Example (C)
/// ```c
/// tokenizers_byte_level_normalizer_free(normalizer);
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_byte_level_normalizer_free(normalizer: *mut ByteLevel) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
