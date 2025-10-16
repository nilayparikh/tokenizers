use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::{Replace, replace::ReplacePattern};
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new Replace normalizer.
///
/// Replaces all occurrences of a pattern with the specified content.
/// The pattern is interpreted as a regular expression.
///
/// # Arguments
/// * `pattern` - Regular expression pattern to search for (UTF-8 encoded, null-terminated)
/// * `content` - Replacement string (UTF-8 encoded, null-terminated)
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created Replace normalizer, or NULL on error
///
/// # Notes
/// - The returned pointer must be freed using tokenizers_replace_normalizer_free()
/// - The pattern is a regular expression (use "\\s+" to replace multiple spaces, etc.)
/// - The content is a literal string (not a regex replacement pattern)
///
/// # Example (C)
/// ```c
/// int status;
/// // Replace multiple spaces with single space
/// void* normalizer = tokenizers_replace_normalizer_new("\\s+", " ", &status);
/// if (status == 0 && normalizer != NULL) {
///     // Use normalizer...
///     tokenizers_replace_normalizer_free(normalizer);
/// }
/// ```
#[no_mangle]
pub extern "C" fn tokenizers_replace_normalizer_new(
    pattern: *const c_char,
    content: *const c_char,
    status: *mut i32,
) -> *mut Replace {
    // Validate status pointer
    if status.is_null() {
        return ptr::null_mut();
    }

    // Initialize status to error
    set_status(status, -1);

    if pattern.is_null() || content.is_null() {
        set_last_error("Pattern and content cannot be null");
        return ptr::null_mut();
    }

    // Convert pattern string
    let pattern_str = match unsafe { CStr::from_ptr(pattern) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid UTF-8 pattern: {}", e));
            return ptr::null_mut();
        }
    };

    // Convert content string
    let content_str = match unsafe { CStr::from_ptr(content) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid UTF-8 content: {}", e));
            return ptr::null_mut();
        }
    };

    // Create regex pattern using ReplacePattern::Regex
    let regex_pattern = ReplacePattern::Regex(pattern_str.to_string());

    // Create the Replace normalizer
    let normalizer = Replace::new(regex_pattern, content_str).map_err(|e| {
        set_last_error(&format!("Failed to create Replace normalizer: {}", e));
    });

    match normalizer {
        Ok(n) => {
            set_status(status, 0);
            Box::into_raw(Box::new(n))
        }
        Err(_) => {
            set_status(status, -2);
            ptr::null_mut()
        }
    }
}

/// Normalizes a string using a Replace normalizer.
///
/// Replaces all pattern matches with the specified content.
///
/// # Arguments
/// * `normalizer` - Pointer to the Replace normalizer
/// * `input` - Input string to normalize (UTF-8 encoded, null-terminated)
/// * `output` - Output buffer for the normalized string
/// * `output_len` - Size of the output buffer
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Length of the normalized string (excluding null terminator), or 0 on error
#[no_mangle]
pub extern "C" fn tokenizers_replace_normalizer_normalize_str(
    normalizer: *const Replace,
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

/// Frees a Replace normalizer instance.
///
/// # Arguments
/// * `normalizer` - Pointer to the Replace normalizer to free
///
/// # Safety
/// - The pointer must have been created by tokenizers_replace_normalizer_new()
/// - After calling this function, the pointer must not be used again
/// - Calling with NULL pointer is safe (no-op)
#[no_mangle]
pub extern "C" fn tokenizers_replace_normalizer_free(normalizer: *mut Replace) {
    if !normalizer.is_null() {
        unsafe {
            drop(Box::from_raw(normalizer));
        }
    }
}
