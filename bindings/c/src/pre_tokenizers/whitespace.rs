use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::PreTokenizer;

/// Sets the status code in the provided pointer.
///
/// # Arguments
/// * `status` - Pointer to status code (0 = success, non-zero = error)
/// * `code` - Status code to set
fn set_status(status: *mut i32, code: i32) {
    if !status.is_null() {
        unsafe {
            *status = code;
        }
    }
}

/// Creates a new Whitespace pre-tokenizer instance.
///
/// This pre-tokenizer splits on word boundaries using the regex pattern `\w+|[^\w\s]+`.
/// It splits on word characters or characters that aren't words or whitespaces.
///
/// # Arguments
/// * `status` - Pointer to status code (0 = success, -1 = null pointer, other = error)
///
/// # Returns
/// Pointer to the created Whitespace pre-tokenizer, or null pointer on failure.
///
/// # Safety
/// The returned pointer must be freed with `tokenizers_whitespace_free`.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_new(status: *mut i32) -> *mut Whitespace {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);

    let pre_tokenizer = Whitespace;
    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using the Whitespace pre-tokenizer.
///
/// Returns the result as a JSON string containing an array of objects with "token" and "offsets" fields.
/// Format: [{"token": "word", "offsets": [start, end]}, ...]
///
/// Uses a two-call pattern:
/// 1. Call with output=null to get required buffer size
/// 2. Call with allocated buffer to get the JSON string
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the Whitespace pre-tokenizer
/// * `input` - Input string to pre-tokenize (UTF-8)
/// * `output` - Output buffer for JSON result (UTF-8), or null to get required size
/// * `output_len` - Length of output buffer in bytes
/// * `status` - Pointer to status code (0 = success, -1 = null pointer, other = error)
///
/// # Returns
/// Required buffer size (including null terminator) if output is null,
/// or number of bytes written (excluding null terminator) if output is provided.
///
/// # Safety
/// The caller must ensure valid pointers and buffer sizes.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_pre_tokenize_str(
    pre_tokenizer: *const Whitespace,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    if status.is_null() {
        return 0;
    }
    set_status(status, -1);

    // Validate pointers
    if pre_tokenizer.is_null() || input.is_null() {
        return 0;
    }

    // Convert input string
    let input_str = unsafe {
        match CStr::from_ptr(input).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return 0;
            }
        }
    };

    // Pre-tokenize
    let pre_tokenizer_ref = unsafe { &*pre_tokenizer };
    let mut pretokenized = tokenizers::tokenizer::PreTokenizedString::from(input_str);

    if let Err(_) = pre_tokenizer_ref.pre_tokenize(&mut pretokenized) {
        set_status(status, -3);
        return 0;
    }

    // Get splits as (token, offsets) pairs
    let splits = pretokenized.get_splits(
        tokenizers::OffsetReferential::Original,
        tokenizers::OffsetType::Char,
    );

    // Build JSON result manually for better control
    let mut json = String::from("[");
    for (idx, (token, offsets, _)) in splits.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        // Escape token string for JSON
        let escaped_token = token.replace('\\', "\\\\").replace('"', "\\\"");
        json.push_str(&format!(
            r#"{{"token":"{}","offsets":[{},{}]}}"#,
            escaped_token, offsets.0, offsets.1
        ));
    }
    json.push(']');

    let json_bytes = json.as_bytes();
    let required_size = json_bytes.len() + 1; // +1 for null terminator

    // First call: return required size
    if output.is_null() {
        set_status(status, 0);
        return required_size;
    }

    // Second call: write to buffer
    if output_len < required_size {
        set_status(status, -4); // Buffer too small
        return 0;
    }

    unsafe {
        ptr::copy_nonoverlapping(json_bytes.as_ptr(), output as *mut u8, json_bytes.len());
        *output.add(json_bytes.len()) = 0; // Null terminator
    }

    set_status(status, 0);
    json_bytes.len()
}

/// Frees a Whitespace pre-tokenizer instance.
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the Whitespace pre-tokenizer to free
///
/// # Safety
/// The pointer must have been created with `tokenizers_whitespace_new` and must not be used after calling this function.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_free(pre_tokenizer: *mut Whitespace) {
    if !pre_tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(pre_tokenizer));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_whitespace_new() {
        let mut status = -1;
        let ptr = tokenizers_whitespace_new(&mut status);
        assert_eq!(status, 0);
        assert!(!ptr.is_null());
        tokenizers_whitespace_free(ptr);
    }

    #[test]
    fn test_whitespace_pre_tokenize_str() {
        let mut status = -1;
        let ptr = tokenizers_whitespace_new(&mut status);
        assert_eq!(status, 0);

        let input = CString::new("Hello, world!").unwrap();

        // First call to get size
        let size = tokenizers_whitespace_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(size > 0);

        // Second call to get result
        let mut buffer = vec![0u8; size];
        let written = tokenizers_whitespace_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            buffer.as_mut_ptr() as *mut c_char,
            size,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(written > 0);

        let result = CStr::from_bytes_with_nul(&buffer)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));

        tokenizers_whitespace_free(ptr);
    }
}
