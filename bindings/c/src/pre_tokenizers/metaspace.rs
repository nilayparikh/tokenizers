use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::PreTokenizer;

/// Helper to set status code
fn set_status(status: *mut i32, code: i32) {
    if !status.is_null() {
        unsafe {
            *status = code;
        }
    }
}

/// Helper to parse PrependScheme from string
fn parse_prepend_scheme(scheme_str: &str) -> Result<PrependScheme, ()> {
    match scheme_str {
        "always" => Ok(PrependScheme::Always),
        "never" => Ok(PrependScheme::Never),
        "first" => Ok(PrependScheme::First),
        _ => Err(()),
    }
}

/// Creates a new Metaspace pre-tokenizer.
///
/// # Parameters
/// * `replacement` - Single character to replace whitespace (UTF-8 string, should be 1 char)
/// * `prepend_scheme` - Prepend scheme ("always", "never", "first")
/// * `split` - Whether to split on the replacement character
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// * Pointer to the created Metaspace pre-tokenizer, or null on failure
///
/// # Safety
/// Caller must eventually free the returned pointer using `tokenizers_metaspace_free`.
#[no_mangle]
pub extern "C" fn tokenizers_metaspace_new(
    replacement: *const c_char,
    prepend_scheme: *const c_char,
    split: bool,
    status: *mut i32,
) -> *mut Metaspace {
    set_status(status, -1);

    // Validate and convert replacement character
    if replacement.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let replacement_str = unsafe {
        match CStr::from_ptr(replacement).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    // Replacement must be exactly 1 character
    let replacement_char = match replacement_str.chars().next() {
        Some(c) if replacement_str.chars().count() == 1 => c,
        _ => {
            set_status(status, -2);
            return ptr::null_mut();
        }
    };

    // Validate and convert prepend_scheme
    if prepend_scheme.is_null() {
        set_status(status, -2);
        return ptr::null_mut();
    }

    let scheme_str = unsafe {
        match CStr::from_ptr(prepend_scheme).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_status(status, -2);
                return ptr::null_mut();
            }
        }
    };

    let scheme_enum = match parse_prepend_scheme(scheme_str) {
        Ok(s) => s,
        Err(_) => {
            set_status(status, -3);
            return ptr::null_mut();
        }
    };

    // Create Metaspace pre-tokenizer
    let pre_tokenizer = Metaspace::new(replacement_char, scheme_enum, split);

    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using the Metaspace pre-tokenizer.
///
/// Returns JSON array: [{"token":"str","offsets":[start,end]},...]
///
/// # Parameters
/// * `pre_tokenizer` - Pointer to the Metaspace pre-tokenizer
/// * `input` - Input string to pre-tokenize
/// * `output` - Buffer to write JSON result (or null to get required size)
/// * `output_len` - Size of output buffer
/// * `status` - Output status code
///
/// # Returns
/// Required buffer size (including null terminator) if output is null,
/// or number of bytes written (including null terminator) if successful
#[no_mangle]
pub extern "C" fn tokenizers_metaspace_pre_tokenize_str(
    pre_tokenizer: *const Metaspace,
    input: *const c_char,
    output: *mut u8,
    output_len: usize,
    status: *mut i32,
) -> usize {
    set_status(status, -1);

    if pre_tokenizer.is_null() || input.is_null() {
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

    // Get splits as JSON
    let splits = pretokenized.get_splits(
        tokenizers::OffsetReferential::Original,
        tokenizers::OffsetType::Byte,
    );

    // Build JSON array
    let mut json = String::from("[");
    for (i, (token, offsets, _)) in splits.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            r#"{{"token":"{}","offsets":[{},{}]}}"#,
            token.replace('\\', "\\\\").replace('"', "\\\""),
            offsets.0,
            offsets.1
        ));
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
        set_status(status, -5);
        return 0;
    }

    // Copy to output buffer
    unsafe {
        ptr::copy_nonoverlapping(json_bytes.as_ptr(), output, json_bytes.len());
        *output.add(json_bytes.len()) = 0; // null terminator
    }

    set_status(status, 0);
    required_size
}

/// Frees a Metaspace pre-tokenizer.
///
/// # Safety
/// The pointer must have been created by `tokenizers_metaspace_new` and
/// must not be used after this call.
#[no_mangle]
pub extern "C" fn tokenizers_metaspace_free(pre_tokenizer: *mut Metaspace) {
    if !pre_tokenizer.is_null() {
        unsafe {
            let _ = Box::from_raw(pre_tokenizer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_metaspace_new_with_default_replacement() {
        let replacement = CString::new("▁").unwrap();
        let scheme = CString::new("always").unwrap();
        let mut status = 0;

        let ptr = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme.as_ptr(),
            true,
            &mut status,
        );

        assert_eq!(status, 0);
        assert!(!ptr.is_null());

        tokenizers_metaspace_free(ptr);
    }

    #[test]
    fn test_metaspace_new_with_underscore() {
        let replacement = CString::new("_").unwrap();
        let scheme = CString::new("always").unwrap();
        let mut status = 0;

        let ptr = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme.as_ptr(),
            true,
            &mut status,
        );

        assert_eq!(status, 0);
        assert!(!ptr.is_null());

        tokenizers_metaspace_free(ptr);
    }

    #[test]
    fn test_metaspace_pre_tokenize_with_always_scheme() {
        let replacement = CString::new("▁").unwrap();
        let scheme = CString::new("always").unwrap();
        let input = CString::new("Hello world").unwrap();
        let mut status = 0;

        let ptr = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme.as_ptr(),
            true,
            &mut status,
        );
        assert_eq!(status, 0);

        // Get required buffer size
        let size = tokenizers_metaspace_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            ptr::null_mut(),
            0,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(size > 0);

        // Allocate buffer and get result
        let mut buffer = vec![0u8; size];
        let written = tokenizers_metaspace_pre_tokenize_str(
            ptr,
            input.as_ptr(),
            buffer.as_mut_ptr(),
            size,
            &mut status,
        );
        assert_eq!(status, 0);
        assert_eq!(written, size);

        let result = String::from_utf8_lossy(&buffer[..size - 1]);
        assert!(result.contains("token"));
        assert!(result.contains("offsets"));

        tokenizers_metaspace_free(ptr);
    }

    #[test]
    fn test_metaspace_new_with_invalid_scheme() {
        let replacement = CString::new("_").unwrap();
        let scheme = CString::new("invalid").unwrap();
        let mut status = 0;

        let ptr = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme.as_ptr(),
            true,
            &mut status,
        );

        assert_eq!(status, -3);
        assert!(ptr.is_null());
    }

    #[test]
    fn test_metaspace_new_with_multi_char_replacement() {
        let replacement = CString::new("__").unwrap();
        let scheme = CString::new("always").unwrap();
        let mut status = 0;

        let ptr = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme.as_ptr(),
            true,
            &mut status,
        );

        assert_eq!(status, -2);
        assert!(ptr.is_null());
    }

    #[test]
    fn test_metaspace_prepend_scheme_variants() {
        let replacement = CString::new("▁").unwrap();
        let mut status = 0;

        // Test "always"
        let scheme_always = CString::new("always").unwrap();
        let ptr_always = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme_always.as_ptr(),
            true,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(!ptr_always.is_null());
        tokenizers_metaspace_free(ptr_always);

        // Test "never"
        let scheme_never = CString::new("never").unwrap();
        let ptr_never = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme_never.as_ptr(),
            true,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(!ptr_never.is_null());
        tokenizers_metaspace_free(ptr_never);

        // Test "first"
        let scheme_first = CString::new("first").unwrap();
        let ptr_first = tokenizers_metaspace_new(
            replacement.as_ptr(),
            scheme_first.as_ptr(),
            true,
            &mut status,
        );
        assert_eq!(status, 0);
        assert!(!ptr_first.is_null());
        tokenizers_metaspace_free(ptr_first);
    }
}
