use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::PreTokenizer;

/// Sets the status code in the provided pointer.
fn set_status(status: *mut i32, code: i32) {
    if !status.is_null() {
        unsafe {
            *status = code;
        }
    }
}

/// Creates a new WhitespaceSplit pre-tokenizer instance.
///
/// This pre-tokenizer simply splits on whitespace. Works like `.split()`.
///
/// # Arguments
/// * `status` - Pointer to status code (0 = success, -1 = null pointer, other = error)
///
/// # Returns
/// Pointer to the created WhitespaceSplit pre-tokenizer, or null pointer on failure.
///
/// # Safety
/// The returned pointer must be freed with `tokenizers_whitespace_split_free`.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_split_new(status: *mut i32) -> *mut WhitespaceSplit {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);

    let pre_tokenizer = WhitespaceSplit;
    set_status(status, 0);
    Box::into_raw(Box::new(pre_tokenizer))
}

/// Pre-tokenizes a string using the WhitespaceSplit pre-tokenizer.
///
/// Returns the result as a JSON string containing an array of objects with "token" and "offsets" fields.
/// Format: [{"token": "word", "offsets": [start, end]}, ...]
///
/// Uses a two-call pattern:
/// 1. Call with output=null to get required buffer size
/// 2. Call with allocated buffer to get the JSON string
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the WhitespaceSplit pre-tokenizer
/// * `input` - Input string to pre-tokenize (UTF-8)
/// * `output` - Output buffer for JSON result (UTF-8), or null to get required size
/// * `output_len` - Length of output buffer in bytes
/// * `status` - Pointer to status code (0 = success, -1 = null pointer, other = error)
///
/// # Returns
/// Required buffer size (including null terminator) if output is null,
/// or number of bytes written (excluding null terminator) if output is provided.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_split_pre_tokenize_str(
    pre_tokenizer: *const WhitespaceSplit,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    if status.is_null() {
        return 0;
    }
    set_status(status, -1);

    if pre_tokenizer.is_null() || input.is_null() {
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

    let pre_tokenizer_ref = unsafe { &*pre_tokenizer };
    let mut pretokenized = tokenizers::tokenizer::PreTokenizedString::from(input_str);

    if let Err(_) = pre_tokenizer_ref.pre_tokenize(&mut pretokenized) {
        set_status(status, -3);
        return 0;
    }

    let splits = pretokenized.get_splits(
        tokenizers::OffsetReferential::Original,
        tokenizers::OffsetType::Char,
    );

    let mut json = String::from("[");
    for (idx, (token, offsets, _)) in splits.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        let escaped_token = token.replace('\\', "\\\\").replace('"', "\\\"");
        json.push_str(&format!(
            r#"{{"token":"{}","offsets":[{},{}]}}"#,
            escaped_token, offsets.0, offsets.1
        ));
    }
    json.push(']');

    let json_bytes = json.as_bytes();
    let required_size = json_bytes.len() + 1;

    if output.is_null() {
        set_status(status, 0);
        return required_size;
    }

    if output_len < required_size {
        set_status(status, -4);
        return 0;
    }

    unsafe {
        ptr::copy_nonoverlapping(json_bytes.as_ptr(), output as *mut u8, json_bytes.len());
        *output.add(json_bytes.len()) = 0;
    }

    set_status(status, 0);
    json_bytes.len()
}

/// Frees a WhitespaceSplit pre-tokenizer instance.
///
/// # Arguments
/// * `pre_tokenizer` - Pointer to the WhitespaceSplit pre-tokenizer to free
///
/// # Safety
/// The pointer must have been created with `tokenizers_whitespace_split_new` and must not be used after calling this function.
#[no_mangle]
pub extern "C" fn tokenizers_whitespace_split_free(pre_tokenizer: *mut WhitespaceSplit) {
    if !pre_tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(pre_tokenizer));
        }
    }
}
