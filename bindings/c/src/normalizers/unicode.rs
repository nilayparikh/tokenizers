use std::ffi::{c_char, CStr};
use std::ptr;
use tokenizers::normalizers::{NFC, NFD, NFKC, NFKD};
use tokenizers::NormalizedString;
use tokenizers::Normalizer;

use crate::{set_last_error, set_status};

/// Creates a new NFD (Canonical Decomposition) Unicode normalizer.
///
/// NFD normalizes text using Unicode Normalization Form D (Canonical Decomposition).
/// Characters are decomposed by canonical equivalence (e.g., "é" → "e" + combining acute accent).
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created NFD normalizer, or NULL on error
#[no_mangle]
pub extern "C" fn tokenizers_nfd_new(status: *mut i32) -> *mut NFD {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);
    
    let normalizer = NFD;
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Creates a new NFC (Canonical Decomposition, followed by Canonical Composition) Unicode normalizer.
///
/// NFC normalizes text using Unicode Normalization Form C (Canonical Decomposition + Composition).
/// This is the most commonly used Unicode normalization form.
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created NFC normalizer, or NULL on error
#[no_mangle]
pub extern "C" fn tokenizers_nfc_new(status: *mut i32) -> *mut NFC {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);
    
    let normalizer = NFC;
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Creates a new NFKD (Compatibility Decomposition) Unicode normalizer.
///
/// NFKD normalizes text using Unicode Normalization Form KD (Compatibility Decomposition).
/// More aggressive than NFD - decomposes compatibility characters (e.g., "ﬁ" → "f" + "i").
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created NFKD normalizer, or NULL on error
#[no_mangle]
pub extern "C" fn tokenizers_nfkd_new(status: *mut i32) -> *mut NFKD {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);
    
    let normalizer = NFKD;
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Creates a new NFKC (Compatibility Decomposition, followed by Canonical Composition) Unicode normalizer.
///
/// NFKC normalizes text using Unicode Normalization Form KC (Compatibility Decomposition + Composition).
/// Most aggressive normalization - useful for search/comparison but may lose information.
///
/// # Arguments
/// * `status` - Output status code (0 = success, negative = error)
///
/// # Returns
/// Pointer to the created NFKC normalizer, or NULL on error
#[no_mangle]
pub extern "C" fn tokenizers_nfkc_new(status: *mut i32) -> *mut NFKC {
    if status.is_null() {
        return ptr::null_mut();
    }
    set_status(status, -1);
    
    let normalizer = NFKC;
    set_status(status, 0);
    Box::into_raw(Box::new(normalizer))
}

/// Normalizes a string using an NFD normalizer.
#[no_mangle]
pub extern "C" fn tokenizers_nfd_normalize_str(
    normalizer: *const NFD,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    normalize_with(normalizer, input, output, output_len, status)
}

/// Normalizes a string using an NFC normalizer.
#[no_mangle]
pub extern "C" fn tokenizers_nfc_normalize_str(
    normalizer: *const NFC,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    normalize_with(normalizer, input, output, output_len, status)
}

/// Normalizes a string using an NFKD normalizer.
#[no_mangle]
pub extern "C" fn tokenizers_nfkd_normalize_str(
    normalizer: *const NFKD,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    normalize_with(normalizer, input, output, output_len, status)
}

/// Normalizes a string using an NFKC normalizer.
#[no_mangle]
pub extern "C" fn tokenizers_nfkc_normalize_str(
    normalizer: *const NFKC,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    normalize_with(normalizer, input, output, output_len, status)
}

/// Generic normalization helper function.
///
/// This function works with any normalizer that implements the Normalizer trait.
fn normalize_with<N: Normalizer>(
    normalizer: *const N,
    input: *const c_char,
    output: *mut c_char,
    output_len: usize,
    status: *mut i32,
) -> usize {
    if status.is_null() {
        return 0;
    }
    set_status(status, -1);

    if normalizer.is_null() || input.is_null() {
        set_last_error("Normalizer and input cannot be null");
        return 0;
    }

    let input_str = match unsafe { CStr::from_ptr(input) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("Invalid UTF-8 input: {}", e));
            return 0;
        }
    };

    let mut normalized = NormalizedString::from(input_str);
    let norm_ref = unsafe { &*normalizer };
    
    if let Err(e) = norm_ref.normalize(&mut normalized) {
        set_last_error(&format!("Normalization failed: {}", e));
        set_status(status, -3);
        return 0;
    }

    let result = normalized.get();
    let result_bytes = result.as_bytes();
    let required_len = result_bytes.len() + 1;

    if output.is_null() {
        set_status(status, 0);
        return required_len;
    }

    if output_len < required_len {
        set_last_error(&format!(
            "Output buffer too small: need {} bytes, got {}",
            required_len, output_len
        ));
        set_status(status, -2);
        return required_len;
    }

    unsafe {
        ptr::copy_nonoverlapping(result_bytes.as_ptr(), output as *mut u8, result_bytes.len());
        *output.add(result_bytes.len()) = 0;
    }

    set_status(status, 0);
    result_bytes.len()
}

/// Frees an NFD normalizer instance.
#[no_mangle]
pub extern "C" fn tokenizers_nfd_free(normalizer: *mut NFD) {
    if !normalizer.is_null() {
        unsafe { drop(Box::from_raw(normalizer)); }
    }
}

/// Frees an NFC normalizer instance.
#[no_mangle]
pub extern "C" fn tokenizers_nfc_free(normalizer: *mut NFC) {
    if !normalizer.is_null() {
        unsafe { drop(Box::from_raw(normalizer)); }
    }
}

/// Frees an NFKD normalizer instance.
#[no_mangle]
pub extern "C" fn tokenizers_nfkd_free(normalizer: *mut NFKD) {
    if !normalizer.is_null() {
        unsafe { drop(Box::from_raw(normalizer)); }
    }
}

/// Frees an NFKC normalizer instance.
#[no_mangle]
pub extern "C" fn tokenizers_nfkc_free(normalizer: *mut NFKC) {
    if !normalizer.is_null() {
        unsafe { drop(Box::from_raw(normalizer)); }
    }
}
