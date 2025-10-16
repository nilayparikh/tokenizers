use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::punctuation::Punctuation;
use tokenizers::{PreTokenizer, SplitDelimiterBehavior};

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

/// Converts a string to SplitDelimiterBehavior enum.
///
/// # Arguments
/// * `behavior_str` - String representation of behavior
///
/// # Returns
/// * Ok(SplitDelimiterBehavior) on success
/// * Err(()) if invalid behavior string
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

/// Creates a new Punctuation pre-tokenizer.
///
/// # Arguments
/// * `behavior` - String specifying the behavior: "removed", "isolated" (default),
///                "merged_with_previous", "merged_with_next", "contiguous"
/// * `status` - Output parameter for error status (0 = success, negative = error)
///
/// # Returns
/// * Pointer to the newly created Punctuation pre-tokenizer, or null on error
///
/// # Safety
/// * The returned pointer must be freed using `tokenizers_punctuation_free`
#[no_mangle]
pub extern "C" fn tokenizers_punctuation_new(
    behavior: *const c_char,
    status: *mut i32,
) -> *mut Punctuation {
    if status.is_null() {
        return ptr::null_mut();
    }

    if behavior.is_null() {
        set_status(status, -1);
        return ptr::null_mut();
    }

    let behavior_str = match unsafe { CStr::from_ptr(behavior) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_status(status, -2);
            return ptr::null_mut();
        }
    };

    let behavior_enum = match parse_behavior(behavior_str) {
        Ok(b) => b,
        Err(_) => {
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    let pre_tokenizer = Punctuation::new(behavior_enum);
    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using Punctuation.
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the Punctuation pre-tokenizer
/// * `input` - The input string to pre-tokenize
/// * `output` - Output buffer for JSON array of tokens with offsets
/// * `output_len` - Length of the output buffer
/// * `status` - Output parameter for error status (0 = success, negative = error)
///
/// # Returns
/// * Number of bytes written to output (including null terminator)
/// * If output is null, returns required buffer size
///
/// # Output Format
/// JSON array: [{"token":"str","offsets":[start,end]}, ...]
///
/// # Safety
/// * `pre_tokenizer` must be a valid pointer from `tokenizers_punctuation_new`
/// * `input` must be a valid UTF-8 null-terminated string
/// * `output` must be null or point to a buffer of at least `output_len` bytes
#[no_mangle]
pub extern "C" fn tokenizers_punctuation_pre_tokenize_str(
    pre_tokenizer: *mut Punctuation,
    input: *const c_char,
    output: *mut u8,
    output_len: usize,
    status: *mut i32,
) -> usize {
    if status.is_null() {
        return 0;
    }

    if pre_tokenizer.is_null() {
        set_status(status, -1);
        return 0;
    }

    if input.is_null() {
        set_status(status, -2);
        return 0;
    }

    let input_str = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_status(status, -3);
            return 0;
        }
    };

    let pre_tokenizer_ref = unsafe { &*pre_tokenizer };

    let mut pretokenized = tokenizers::PreTokenizedString::from(input_str);

    if let Err(_) = pre_tokenizer_ref.pre_tokenize(&mut pretokenized) {
        set_status(status, -4);
        return 0;
    }

    // Get splits
    let splits = pretokenized.get_splits(
        tokenizers::OffsetReferential::Original,
        tokenizers::OffsetType::Byte,
    );

    // Build JSON manually
    let mut json = String::from("[");
    for (i, (token, offsets, _)) in splits.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str("{\"token\":\"");

        // Escape special characters in token
        for ch in token.chars() {
            match ch {
                '\\' => json.push_str("\\\\"),
                '"' => json.push_str("\\\""),
                '\n' => json.push_str("\\n"),
                '\r' => json.push_str("\\r"),
                '\t' => json.push_str("\\t"),
                _ => json.push(ch),
            }
        }

        json.push_str("\",\"offsets\":[");
        json.push_str(&offsets.0.to_string());
        json.push(',');
        json.push_str(&offsets.1.to_string());
        json.push_str("]}");
    }
    json.push(']');

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
        return required_size;
    }

    // Copy JSON to output buffer
    unsafe {
        std::ptr::copy_nonoverlapping(json_bytes.as_ptr(), output, json_bytes.len());
        *output.add(json_bytes.len()) = 0; // null terminator
    }
    set_status(status, 0);

    required_size
}

/// Frees a Punctuation pre-tokenizer.
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the pre-tokenizer to free (can be null)
///
/// # Safety
/// * `pre_tokenizer` must be null or a valid pointer from `tokenizers_punctuation_new`
/// * After calling this function, the pointer must not be used again
#[no_mangle]
pub extern "C" fn tokenizers_punctuation_free(pre_tokenizer: *mut Punctuation) {
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
    fn test_punctuation_new_isolated() {
        let behavior = CString::new("isolated").unwrap();
        let mut status = 0;
        let pre_tokenizer = tokenizers_punctuation_new(behavior.as_ptr(), &mut status);
        assert_eq!(status, 0);
        assert!(!pre_tokenizer.is_null());
        tokenizers_punctuation_free(pre_tokenizer);
    }

    #[test]
    fn test_punctuation_new_removed() {
        let behavior = CString::new("removed").unwrap();
        let mut status = 0;
        let pre_tokenizer = tokenizers_punctuation_new(behavior.as_ptr(), &mut status);
        assert_eq!(status, 0);
        assert!(!pre_tokenizer.is_null());
        tokenizers_punctuation_free(pre_tokenizer);
    }

    #[test]
    fn test_punctuation_pre_tokenize_isolated() {
        let behavior = CString::new("isolated").unwrap();
        let mut status = 0;
        let pre_tokenizer = tokenizers_punctuation_new(behavior.as_ptr(), &mut status);
        assert_eq!(status, 0);

        let input = CString::new("Hello, world!").unwrap();
        let mut buffer = vec![0u8; 1024];
        let written = tokenizers_punctuation_pre_tokenize_str(
            pre_tokenizer,
            input.as_ptr(),
            buffer.as_mut_ptr(),
            buffer.len(),
            &mut status,
        );

        assert_eq!(status, 0);
        assert!(written > 0);

        let json = String::from_utf8_lossy(&buffer[..written - 1]);
        assert!(json.contains("\"Hello\""));
        assert!(json.contains("\",\""));
        assert!(json.contains("\"world\""));
        assert!(json.contains("\"!\""));

        tokenizers_punctuation_free(pre_tokenizer);
    }

    #[test]
    fn test_punctuation_invalid_behavior() {
        let behavior = CString::new("invalid").unwrap();
        let mut status = 0;
        let pre_tokenizer = tokenizers_punctuation_new(behavior.as_ptr(), &mut status);
        assert_eq!(status, -3);
        assert!(pre_tokenizer.is_null());
    }
}
