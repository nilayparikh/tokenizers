use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{PreTokenizer, SplitDelimiterBehavior};

/// Helper to set status code
fn set_status(status: *mut i32, code: i32) {
    if !status.is_null() {
        unsafe {
            *status = code;
        }
    }
}

/// Helper to parse SplitDelimiterBehavior from string
fn parse_behavior(behavior_str: &str) -> Result<SplitDelimiterBehavior, ()> {
    match behavior_str {
        "removed" => Ok(SplitDelimiterBehavior::Removed),
        "isolated" => Ok(SplitDelimiterBehavior::Isolated),
        "merged_with_previous" => Ok(SplitDelimiterBehavior::MergedWithPrevious),
        "merged_with_next" => Ok(SplitDelimiterBehavior::MergedWithNext),
        "contiguous" => Ok(SplitDelimiterBehavior::Contiguous),
        _ => Err(()),
    }
}

/// Creates a new Split pre-tokenizer with a string pattern.
///
/// # Parameters
/// * `pattern` - String pattern to split on (literal string, not regex)
/// * `behavior` - Split delimiter behavior ("removed", "isolated", etc.)
/// * `invert` - Whether to invert the pattern (split on non-matching instead)
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// * Pointer to the created Split pre-tokenizer, or null on failure
///
/// # Safety
/// Caller must eventually free the returned pointer using `tokenizers_split_free`.
#[no_mangle]
pub extern "C" fn tokenizers_split_new(
    pattern: *const c_char,
    behavior: *const c_char,
    invert: bool,
    status: *mut i32,
) -> *mut Split {
    set_status(status, -1);

    // Validate and convert pattern
    if pattern.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let pattern_str = unsafe {
        match CStr::from_ptr(pattern).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    // Validate and convert behavior
    if behavior.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let behavior_str = unsafe {
        match CStr::from_ptr(behavior).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    let behavior_enum = match parse_behavior(behavior_str) {
        Ok(b) => b,
        Err(_) => {
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    // Create Split pre-tokenizer with string pattern
    let split_pattern = SplitPattern::String(pattern_str.to_string());
    let pre_tokenizer = match Split::new(split_pattern, behavior_enum, invert) {
        Ok(p) => p,
        Err(_) => {
            set_status(status, -4);
            return ptr::null_mut();
        }
    };

    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Creates a new Split pre-tokenizer with a regex pattern.
///
/// # Parameters
/// * `pattern` - Regex pattern string
/// * `behavior` - Split delimiter behavior ("removed", "isolated", etc.)
/// * `invert` - Whether to invert the pattern
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// * Pointer to the created Split pre-tokenizer, or null on failure
///
/// # Safety
/// Caller must eventually free the returned pointer using `tokenizers_split_free`.
#[no_mangle]
pub extern "C" fn tokenizers_split_new_regex(
    pattern: *const c_char,
    behavior: *const c_char,
    invert: bool,
    status: *mut i32,
) -> *mut Split {
    set_status(status, -1);

    // Validate and convert pattern
    if pattern.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let pattern_str = unsafe {
        match CStr::from_ptr(pattern).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    // Validate and convert behavior
    if behavior.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let behavior_str = unsafe {
        match CStr::from_ptr(behavior).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    let behavior_enum = match parse_behavior(behavior_str) {
        Ok(b) => b,
        Err(_) => {
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    // Create regex pattern
    let split_pattern = SplitPattern::Regex(pattern_str.to_string());
    let pre_tokenizer = match Split::new(split_pattern, behavior_enum, invert) {
        Ok(p) => p,
        Err(_) => {
            set_status(status, -4);
            return ptr::null_mut();
        }
    };

    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using Split.
///
/// On the first call with `output` as null, this function returns the required buffer size.
/// On the second call with a properly sized buffer, it fills the buffer with the JSON result.
///
/// # Parameters
/// * `pre_tokenizer` - Pointer to the Split pre-tokenizer
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
/// * `pre_tokenizer` must be a valid pointer from `tokenizers_split_new`
/// * `input` must be a valid null-terminated UTF-8 string
/// * `output` (if not null) must point to a buffer of at least `output_len` bytes
#[no_mangle]
pub extern "C" fn tokenizers_split_pre_tokenize_str(
    pre_tokenizer: *const Split,
    input: *const c_char,
    output: *mut u8,
    output_len: usize,
    status: *mut i32,
) -> usize {
    set_status(status, -1);

    // Validate pre_tokenizer pointer
    if pre_tokenizer.is_null() {
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

    // Build JSON manually
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
    let required_size = json_bytes.len() + 1;

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

/// Frees a Split pre-tokenizer.
///
/// # Parameters
/// * `pre_tokenizer` - Pointer to the Split pre-tokenizer to free
///
/// # Safety
/// * `pre_tokenizer` must be a valid pointer from `tokenizers_split_new`
/// * Must not be called more than once for the same pointer
/// * Pointer must not be used after calling this function
#[no_mangle]
pub extern "C" fn tokenizers_split_free(pre_tokenizer: *mut Split) {
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
    fn test_split_new_string_pattern() {
        let pattern = std::ffi::CString::new("|").unwrap();
        let behavior = std::ffi::CString::new("isolated").unwrap();
        let mut status = 0;

        let ptr = tokenizers_split_new(pattern.as_ptr(), behavior.as_ptr(), false, &mut status);
        assert!(!ptr.is_null());
        assert_eq!(status, 0);

        tokenizers_split_free(ptr);
    }

    #[test]
    fn test_split_pre_tokenize_csv() {
        let pattern = std::ffi::CString::new("|").unwrap();
        let behavior = std::ffi::CString::new("removed").unwrap();
        let mut status = 0;

        let ptr = tokenizers_split_new(pattern.as_ptr(), behavior.as_ptr(), false, &mut status);
        assert!(!ptr.is_null());

        let input = std::ffi::CString::new("hello|world|test").unwrap();

        // Get size
        let size =
            tokenizers_split_pre_tokenize_str(ptr, input.as_ptr(), ptr::null_mut(), 0, &mut status);
        assert!(size > 0);
        assert_eq!(status, 0);

        // Get result
        let mut buffer = vec![0u8; size];
        let written = tokenizers_split_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            buffer.as_mut_ptr(),
            size,
            &mut status,
        );
        assert_eq!(written, size);
        assert_eq!(status, 0);

        let json = std::str::from_utf8(&buffer[..size - 1]).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("world"));
        assert!(json.contains("test"));

        tokenizers_split_free(ptr);
    }

    #[test]
    fn test_split_new_regex_pattern() {
        let pattern = std::ffi::CString::new(r"\s+").unwrap();
        let behavior = std::ffi::CString::new("removed").unwrap();
        let mut status = 0;

        let ptr =
            tokenizers_split_new_regex(pattern.as_ptr(), behavior.as_ptr(), false, &mut status);
        assert!(!ptr.is_null());
        assert_eq!(status, 0);

        tokenizers_split_free(ptr);
    }

    #[test]
    fn test_split_invalid_behavior() {
        let pattern = std::ffi::CString::new("|").unwrap();
        let behavior = std::ffi::CString::new("invalid").unwrap();
        let mut status = 0;

        let ptr = tokenizers_split_new(pattern.as_ptr(), behavior.as_ptr(), false, &mut status);
        assert!(ptr.is_null());
        assert_eq!(status, -3);
    }
}
