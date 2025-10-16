use libc::c_char;
use std::ffi::CStr;
use std::ptr;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::tokenizer::PreTokenizedString;
use tokenizers::PreTokenizer;

/// Helper to set status code
#[inline]
fn set_status(status: *mut i32, value: i32) {
    if !status.is_null() {
        unsafe { *status = value };
    }
}

/// Creates a new Digits pre-tokenizer
///
/// # Parameters
/// - `individual_digits`: If true, each digit is separated individually. If false, consecutive digits are grouped.
/// - `status`: Output status code (0 = success, non-zero = error)
///
/// # Returns
/// Pointer to the pre-tokenizer instance, or null on error
///
/// # Safety
/// The returned pointer must be freed with `tokenizers_digits_free`
#[no_mangle]
pub extern "C" fn tokenizers_digits_new(individual_digits: bool, status: *mut i32) -> *mut Digits {
    if status.is_null() {
        return ptr::null_mut();
    }

    set_status(status, -1);

    let pre_tokenizer = Digits::new(individual_digits);

    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using the Digits pre-tokenizer
///
/// This function splits the input on digits according to the `individual_digits` setting.
/// Returns JSON array: [{"token":"string","offsets":[start,end]},...]
///
/// # Two-call pattern
/// 1. Call with `output = null` to get required buffer size
/// 2. Allocate buffer of returned size
/// 3. Call again with allocated buffer to get JSON result
///
/// # Parameters
/// - `pre_tokenizer`: The Digits pre-tokenizer instance
/// - `input`: UTF-8 input string to pre-tokenize
/// - `output`: Output buffer for JSON result (or null to query size)
/// - `output_len`: Size of output buffer
/// - `status`: Output status code (0 = success, non-zero = error)
///
/// # Returns
/// Required buffer size (including null terminator)
#[no_mangle]
pub extern "C" fn tokenizers_digits_pre_tokenize_str(
    pre_tokenizer: *mut Digits,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    if status.is_null() || pre_tokenizer.is_null() || input.is_null() {
        if !status.is_null() {
            set_status(status, -1);
        }
        return 0;
    }

    set_status(status, -1);

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

    // Get pre-tokenizer reference
    let pre_tokenizer_ref = unsafe { &*pre_tokenizer };

    // Create PreTokenizedString and apply pre-tokenizer
    let mut pretokenized = PreTokenizedString::from(input_str);
    if let Err(_) = pre_tokenizer_ref.pre_tokenize(&mut pretokenized) {
        set_status(status, -3);
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

    // Check buffer size
    if output_len < required_size {
        set_status(status, -4);
        return required_size;
    }

    // Copy to output buffer
    unsafe {
        ptr::copy_nonoverlapping(json_bytes.as_ptr(), output as *mut u8, json_bytes.len());
        *output.add(json_bytes.len()) = 0; // null terminator
    }

    set_status(status, 0);
    required_size
}

/// Frees a Digits pre-tokenizer instance
///
/// # Parameters
/// - `pre_tokenizer`: The instance to free
///
/// # Safety
/// The pointer must have been created by `tokenizers_digits_new` and must not be used after this call
#[no_mangle]
pub extern "C" fn tokenizers_digits_free(pre_tokenizer: *mut Digits) {
    if !pre_tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(pre_tokenizer));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digits_new() {
        let mut status = 0;
        let pre_tokenizer = tokenizers_digits_new(false, &mut status);
        assert_eq!(status, 0);
        assert!(!pre_tokenizer.is_null());
        tokenizers_digits_free(pre_tokenizer);
    }

    #[test]
    fn test_digits_pre_tokenize_grouped() {
        let mut status = 0;
        let pre_tokenizer = tokenizers_digits_new(false, &mut status);
        assert!(!pre_tokenizer.is_null());

        let input = std::ffi::CString::new("Call 123 please").unwrap();

        // First call to get size
        let size = tokenizers_digits_pre_tokenize_str(
            pre_tokenizer,
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(size > 0);

        // Second call to get result
        let mut buffer = vec![0u8; size];
        let written = tokenizers_digits_pre_tokenize_str(
            pre_tokenizer,
            input.as_ptr(),
            buffer.as_mut_ptr() as *mut c_char,
            size,
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(written, size);

        let json = unsafe { CStr::from_ptr(buffer.as_ptr() as *const c_char) }
            .to_str()
            .unwrap();

        // Should have "Call ", "123", " please" as grouped
        assert!(json.contains("\"token\":\"123\""));

        tokenizers_digits_free(pre_tokenizer);
    }

    #[test]
    fn test_digits_pre_tokenize_individual() {
        let mut status = 0;
        let pre_tokenizer = tokenizers_digits_new(true, &mut status);
        assert!(!pre_tokenizer.is_null());

        let input = std::ffi::CString::new("Call 123 please").unwrap();

        // Get result
        let size = tokenizers_digits_pre_tokenize_str(
            pre_tokenizer,
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        let mut buffer = vec![0u8; size];
        tokenizers_digits_pre_tokenize_str(
            pre_tokenizer,
            input.as_ptr(),
            buffer.as_mut_ptr() as *mut c_char,
            size,
            &mut status,
        );

        let json = unsafe { CStr::from_ptr(buffer.as_ptr() as *const c_char) }
            .to_str()
            .unwrap();

        // Should have individual digits "1", "2", "3"
        assert!(json.contains("\"token\":\"1\""));
        assert!(json.contains("\"token\":\"2\""));
        assert!(json.contains("\"token\":\"3\""));

        tokenizers_digits_free(pre_tokenizer);
    }
}
