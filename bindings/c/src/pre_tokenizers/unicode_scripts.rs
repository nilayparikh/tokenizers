use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tokenizers::PreTokenizer;

/// Creates a new UnicodeScripts pre-tokenizer.
///
/// This pre-tokenizer splits text based on Unicode script boundaries.
/// It groups characters from the same script together (e.g., Latin, Cyrillic, Han).
///
/// # Parameters
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// * Pointer to the created UnicodeScripts pre-tokenizer, or null on failure
///
/// # Safety
/// Caller must eventually free the returned pointer using `tokenizers_unicode_scripts_free`.
#[no_mangle]
pub extern "C" fn tokenizers_unicode_scripts_new(status: *mut i32) -> *mut UnicodeScripts {
    // Set status to success
    if !status.is_null() {
        unsafe {
            *status = 0;
        }
    }

    let pre_tokenizer = UnicodeScripts;
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using UnicodeScripts.
///
/// On the first call with `output` as null, this function returns the required buffer size.
/// On the second call with a properly sized buffer, it fills the buffer with the JSON result.
///
/// # Parameters
/// * `pre_tokenizer` - Pointer to the UnicodeScripts pre-tokenizer
/// * `input` - Null-terminated UTF-8 string to pre-tokenize
/// * `output` - Buffer to receive the JSON result (null to query size)
/// * `output_len` - Size of the output buffer in bytes
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// * Required buffer size (including null terminator) if `output` is null
/// * Number of bytes written (including null terminator) if successful
/// * 0 on error
///
/// # Output Format
/// JSON array: `[{"token":"str","offsets":[start,end]},...]`
///
/// # Safety
/// * `pre_tokenizer` must be a valid pointer from `tokenizers_unicode_scripts_new`
/// * `input` must be a valid null-terminated UTF-8 string
/// * `output` (if not null) must point to a buffer of at least `output_len` bytes
#[no_mangle]
pub extern "C" fn tokenizers_unicode_scripts_pre_tokenize_str(
    pre_tokenizer: *const UnicodeScripts,
    input: *const c_char,
    output: *mut u8,
    output_len: usize,
    status: *mut i32,
) -> usize {
    // Helper to set status
    let set_status = |s: *mut i32, code: i32| {
        if !s.is_null() {
            unsafe {
                *s = code;
            }
        }
    };

    // Validate pre_tokenizer pointer
    if pre_tokenizer.is_null() {
        set_status(status, -1);
        return 0;
    }

    // Validate and convert input string
    if input.is_null() {
        set_status(status, -2);
        return 0;
    }

    let input_str = unsafe {
        match CStr::from_ptr(input).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return 0;
            }
        }
    };

    // Perform pre-tokenization
    let pre_tokenizer_ref = unsafe { &*pre_tokenizer };
    let mut pretokenized = tokenizers::tokenizer::PreTokenizedString::from(input_str);

    if let Err(_) = pre_tokenizer_ref.pre_tokenize(&mut pretokenized) {
        set_status(status, -4);
        return 0;
    }

    // Extract tokens and offsets
    let splits = pretokenized.get_splits(
        tokenizers::OffsetReferential::Original,
        tokenizers::OffsetType::Byte,
    );

    // Build JSON manually to avoid serde dependency
    let mut json = String::from("[");
    for (i, (token, (start, end), _)) in splits.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        // Escape special characters in token
        let escaped_token = token
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        json.push_str(&format!(
            r#"{{"token":"{}","offsets":[{},{}]}}"#,
            escaped_token, start, end
        ));
    }
    json.push(']');

    // Add null terminator
    let json_bytes = json.as_bytes();
    let required_size = json_bytes.len() + 1; // +1 for null terminator

    // If output is null, return required size
    if output.is_null() {
        set_status(status, 0);
        return required_size;
    }

    // Check if buffer is large enough
    if output_len < required_size {
        set_status(status, -5);
        return 0;
    }

    // Copy JSON to output buffer
    unsafe {
        ptr::copy_nonoverlapping(json_bytes.as_ptr(), output, json_bytes.len());
        *output.add(json_bytes.len()) = 0; // Null terminator
    }

    set_status(status, 0);
    required_size
}

/// Frees a UnicodeScripts pre-tokenizer.
///
/// # Parameters
/// * `pre_tokenizer` - Pointer to the UnicodeScripts pre-tokenizer to free
///
/// # Safety
/// * `pre_tokenizer` must be a valid pointer from `tokenizers_unicode_scripts_new`
/// * Must not be called more than once for the same pointer
/// * Pointer must not be used after calling this function
#[no_mangle]
pub extern "C" fn tokenizers_unicode_scripts_free(pre_tokenizer: *mut UnicodeScripts) {
    if !pre_tokenizer.is_null() {
        unsafe {
            let _ = Box::from_raw(pre_tokenizer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_scripts_new() {
        let mut status = 0;
        let ptr = tokenizers_unicode_scripts_new(&mut status);
        assert!(!ptr.is_null());
        assert_eq!(status, 0);
        tokenizers_unicode_scripts_free(ptr);
    }

    #[test]
    fn test_unicode_scripts_pre_tokenize_latin_cyrillic() {
        let mut status = 0;
        let ptr = tokenizers_unicode_scripts_new(&mut status);
        assert!(!ptr.is_null());

        let input = std::ffi::CString::new("Hello Привет").unwrap();

        // First call to get size
        let size = tokenizers_unicode_scripts_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert!(size > 0);
        assert_eq!(status, 0);

        // Second call to get result
        let mut buffer = vec![0u8; size];
        let written = tokenizers_unicode_scripts_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            buffer.as_mut_ptr(),
            size,
            &mut status,
        );
        assert_eq!(written, size);
        assert_eq!(status, 0);

        // Verify JSON contains both scripts
        let json = std::str::from_utf8(&buffer[..size - 1]).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("Привет"));

        tokenizers_unicode_scripts_free(ptr);
    }

    #[test]
    fn test_unicode_scripts_null_pre_tokenizer() {
        let mut status = 0;
        let input = std::ffi::CString::new("test").unwrap();
        let size = tokenizers_unicode_scripts_pre_tokenize_str(
            ptr::null(),
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert_eq!(size, 0);
        assert_eq!(status, -1);
    }

    #[test]
    fn test_unicode_scripts_null_input() {
        let mut status = 0;
        let ptr = tokenizers_unicode_scripts_new(&mut status);
        let size = tokenizers_unicode_scripts_pre_tokenize_str(
            ptr,
            ptr::null(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert_eq!(size, 0);
        assert_eq!(status, -2);
        tokenizers_unicode_scripts_free(ptr);
    }
}
