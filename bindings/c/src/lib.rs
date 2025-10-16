use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::AddedToken;
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokenizers::PostProcessor;

mod encoding;
mod models;
mod normalizers;
mod pre_tokenizers;

pub struct CTokenizer {
    inner: tokenizers::Tokenizer,
}

#[derive(Clone)]
pub struct CEncoding {
    ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(u32, u32)>,
    type_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    special_tokens_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
    sequence_ids: Vec<Option<usize>>,
    overflowing: Vec<CEncoding>,
}

#[derive(Deserialize)]
struct AddedTokenPayload {
    content: String,
    single_word: Option<bool>,
    lstrip: Option<bool>,
    rstrip: Option<bool>,
    normalized: Option<bool>,
    special: Option<bool>,
}

impl AddedTokenPayload {
    fn into_added_token(self, default_special: bool) -> AddedToken {
        let mut token = AddedToken::from(self.content, self.special.unwrap_or(default_special));

        if let Some(single_word) = self.single_word {
            token = token.single_word(single_word);
        }

        if let Some(lstrip) = self.lstrip {
            token = token.lstrip(lstrip);
        }

        if let Some(rstrip) = self.rstrip {
            token = token.rstrip(rstrip);
        }

        if let Some(normalized) = self.normalized {
            token = token.normalized(normalized);
        }

        token
    }
}

#[derive(Serialize)]
struct AddedTokenDecoderEntry<'a> {
    id: u32,
    content: &'a str,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

pub(crate) fn set_status(status: *mut c_int, value: c_int) {
    if !status.is_null() {
        unsafe {
            *status = value;
        }
    }
}

fn set_length(len_ptr: *mut usize, value: usize) {
    if !len_ptr.is_null() {
        unsafe {
            *len_ptr = value;
        }
    }
}

pub(crate) fn set_last_error(message: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(message).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

fn read_utf8(ptr: *const c_char) -> Result<String, ()> {
    if ptr.is_null() {
        return Err(());
    }

    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_str().map(|s| s.to_owned()).map_err(|_| ())
}

fn read_optional_utf8(ptr: *const c_char) -> Result<Option<String>, ()> {
    if ptr.is_null() {
        Ok(None)
    } else {
        read_utf8(ptr).map(Some)
    }
}

fn parse_added_tokens(
    json_ptr: *const c_char,
    default_special: bool,
) -> Result<Vec<AddedToken>, String> {
    if json_ptr.is_null() {
        return Ok(vec![]);
    }

    let json = read_utf8(json_ptr)
        .map_err(|_| "tokenizers_add_tokens could not decode JSON payload".to_string())?;
    if json.trim().is_empty() {
        return Ok(vec![]);
    }

    let payloads: Result<Vec<AddedTokenPayload>, _> = serde_json::from_str(&json);
    match payloads {
        Ok(items) => Ok(items
            .into_iter()
            .map(|payload| payload.into_added_token(default_special))
            .collect()),
        Err(err) => Err(format!(
            "tokenizers_add_tokens failed to parse payload: {err}"
        )),
    }
}

impl CEncoding {
    fn from_encoding(encoding: tokenizers::Encoding) -> Self {
        let ids = encoding.get_ids().to_vec();
        let tokens = encoding.get_tokens().to_vec();
        let offsets = encoding
            .get_offsets()
            .iter()
            .map(|(start, end)| (*start as u32, *end as u32))
            .collect();
        let type_ids = encoding.get_type_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let special_tokens_mask = encoding.get_special_tokens_mask().to_vec();
        let word_ids = encoding.get_word_ids().to_vec();
        let sequence_ids = encoding.get_sequence_ids();
        let overflowing = encoding
            .get_overflowing()
            .iter()
            .map(|enc| Self::from_encoding(enc.clone()))
            .collect();

        Self {
            ids,
            tokens,
            offsets,
            type_ids,
            attention_mask,
            special_tokens_mask,
            word_ids,
            sequence_ids,
            overflowing,
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_get_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match &*cell.borrow() {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

#[no_mangle]
pub extern "C" fn tokenizers_create(json: *const c_char, status: *mut c_int) -> *mut CTokenizer {
    if json.is_null() {
        set_last_error("tokenizers_create received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    match read_utf8(json) {
        Ok(content) => match tokenizers::Tokenizer::from_bytes(content.into_bytes()) {
            Ok(tokenizer) => {
                clear_last_error();
                set_status(status, 0);
                Box::into_raw(Box::new(CTokenizer { inner: tokenizer }))
            }
            Err(err) => {
                set_last_error(&format!("tokenizers_create failed: {err}"));
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        Err(_) => {
            set_last_error("tokenizers_create could not decode JSON string");
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free(tokenizer: *mut CTokenizer) {
    if !tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(tokenizer));
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encode(
    tokenizer: *mut CTokenizer,
    sequence: *const c_char,
    pair: *const c_char,
    add_special_tokens: bool,
    len_ptr: *mut usize,
    status: *mut c_int,
) -> *mut CEncoding {
    if tokenizer.is_null() || sequence.is_null() {
        set_last_error("tokenizers_encode received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &mut *tokenizer };

    let primary = match read_utf8(sequence) {
        Ok(value) => value,
        Err(_) => {
            set_last_error("tokenizers_encode could not decode primary sequence");
            set_status(status, 2);
            return ptr::null_mut();
        }
    };

    let encoding_result = if pair.is_null() {
        tokenizer.inner.encode(primary.as_str(), add_special_tokens)
    } else {
        match read_utf8(pair) {
            Ok(second) => tokenizer
                .inner
                .encode((primary.as_str(), second.as_str()), add_special_tokens),
            Err(_) => {
                set_last_error("tokenizers_encode could not decode pair sequence");
                set_status(status, 3);
                return ptr::null_mut();
            }
        }
    };

    match encoding_result {
        Ok(encoding) => {
            let encoding = CEncoding::from_encoding(encoding);
            let len = encoding.ids.len();
            set_length(len_ptr, len);
            clear_last_error();
            set_status(status, 0);
            Box::into_raw(Box::new(encoding))
        }
        Err(err) => {
            set_last_error(&format!("tokenizers_encode failed: {err}"));
            set_status(status, 4);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_free(encoding: *mut CEncoding) {
    if !encoding.is_null() {
        unsafe {
            drop(Box::from_raw(encoding));
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_ids(
    encoding: *const CEncoding,
    buffer: *mut u32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.ids.len());
    unsafe {
        ptr::copy_nonoverlapping(encoding.ids.as_ptr(), buffer, copy_len);
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_tokens(
    encoding: *const CEncoding,
    buffer: *mut *mut c_char,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let count = len.min(encoding.tokens.len());

    for (index, token) in encoding.tokens.iter().take(count).enumerate() {
        if let Ok(c_string) = CString::new(token.as_str()) {
            unsafe {
                *buffer.add(index) = c_string.into_raw();
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_offsets(
    encoding: *const CEncoding,
    buffer: *mut u32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let required = encoding.offsets.len() * 2;
    let copy_len = len.min(required);

    for (index, (start, end)) in encoding.offsets.iter().enumerate() {
        let base = index * 2;
        if base + 1 >= copy_len {
            break;
        }

        unsafe {
            *buffer.add(base) = *start;
            *buffer.add(base + 1) = *end;
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_token_to_id(
    tokenizer: *const CTokenizer,
    token: *const c_char,
    status: *mut c_int,
) -> i32 {
    if tokenizer.is_null() || token.is_null() {
        set_last_error("tokenizers_token_to_id received null pointer");
        set_status(status, 1);
        return -1;
    }

    let tokenizer = unsafe { &*tokenizer };
    match read_utf8(token) {
        Ok(value) => match tokenizer.inner.token_to_id(&value) {
            Some(id) => {
                clear_last_error();
                set_status(status, 0);
                id as i32
            }
            None => {
                set_last_error("tokenizers_token_to_id: token not found");
                set_status(status, 2);
                -1
            }
        },
        Err(_) => {
            set_last_error("tokenizers_token_to_id could not decode token");
            set_status(status, 3);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_id_to_token(
    tokenizer: *const CTokenizer,
    id: u32,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_id_to_token received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    match tokenizer.inner.id_to_token(id) {
        Some(token) => match CString::new(token) {
            Ok(c_string) => {
                clear_last_error();
                set_status(status, 0);
                c_string.into_raw()
            }
            Err(_) => {
                set_last_error("tokenizers_id_to_token failed to build CString");
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        None => {
            set_last_error("tokenizers_id_to_token: id not found");
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_get_config(
    tokenizer: *const CTokenizer,
    pretty: bool,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_config received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    match tokenizer.inner.to_string(pretty) {
        Ok(json) => match CString::new(json) {
            Ok(c_string) => {
                clear_last_error();
                set_status(status, 0);
                c_string.into_raw()
            }
            Err(_) => {
                set_last_error("tokenizers_get_config failed to allocate CString");
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("tokenizers_get_config failed: {err}"));
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_decode(
    tokenizer: *const CTokenizer,
    ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() || ids.is_null() {
        set_last_error("tokenizers_decode received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    let ids = unsafe { std::slice::from_raw_parts(ids, len) };

    match tokenizer.inner.decode(ids, skip_special_tokens) {
        Ok(text) => match CString::new(text) {
            Ok(c_string) => {
                clear_last_error();
                set_status(status, 0);
                c_string.into_raw()
            }
            Err(_) => {
                set_last_error("tokenizers_decode failed to allocate CString");
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("tokenizers_decode failed: {err}"));
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_get_vocab(
    tokenizer: *const CTokenizer,
    with_added: bool,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_vocab received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    match serde_json::to_string(&tokenizer.inner.get_vocab(with_added)) {
        Ok(json) => match CString::new(json) {
            Ok(c_string) => {
                clear_last_error();
                set_status(status, 0);
                c_string.into_raw()
            }
            Err(_) => {
                set_last_error("tokenizers_get_vocab failed to allocate CString");
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("tokenizers_get_vocab failed: {err}"));
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_add_tokens(
    tokenizer: *mut CTokenizer,
    tokens_json: *const c_char,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_add_tokens received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &mut *tokenizer };
    let tokens = match parse_added_tokens(tokens_json, false) {
        Ok(tokens) => tokens,
        Err(message) => {
            set_last_error(&message);
            set_status(status, 2);
            return 0;
        }
    };

    if tokens.is_empty() {
        clear_last_error();
        set_status(status, 0);
        return 0;
    }

    let added = tokenizer.inner.add_tokens(&tokens);
    if added > i32::MAX as usize {
        set_last_error("tokenizers_add_tokens overflowed");
        set_status(status, 3);
        return 0;
    }

    clear_last_error();
    set_status(status, 0);
    added as c_int
}

#[no_mangle]
pub extern "C" fn tokenizers_add_special_tokens(
    tokenizer: *mut CTokenizer,
    tokens_json: *const c_char,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_add_special_tokens received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &mut *tokenizer };
    let tokens = match parse_added_tokens(tokens_json, true) {
        Ok(tokens) => tokens,
        Err(message) => {
            set_last_error(&message);
            set_status(status, 2);
            return 0;
        }
    };

    if tokens.is_empty() {
        clear_last_error();
        set_status(status, 0);
        return 0;
    }

    let added = tokenizer.inner.add_special_tokens(&tokens);
    if added > i32::MAX as usize {
        set_last_error("tokenizers_add_special_tokens overflowed");
        set_status(status, 3);
        return 0;
    }

    clear_last_error();
    set_status(status, 0);
    added as c_int
}

#[no_mangle]
pub extern "C" fn tokenizers_get_added_tokens_decoder(
    tokenizer: *const CTokenizer,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_added_tokens_decoder received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    let decoder = tokenizer.inner.get_added_tokens_decoder();
    let mut entries = Vec::with_capacity(decoder.len());

    for (id, token) in decoder.iter() {
        entries.push(AddedTokenDecoderEntry {
            id: *id,
            content: &token.content,
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        });
    }

    match serde_json::to_string(&entries) {
        Ok(json) => match CString::new(json) {
            Ok(c_string) => {
                clear_last_error();
                set_status(status, 0);
                c_string.into_raw()
            }
            Err(_) => {
                set_last_error("tokenizers_get_added_tokens_decoder failed to allocate CString");
                set_status(status, 2);
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!(
                "tokenizers_get_added_tokens_decoder failed: {err}"
            ));
            set_status(status, 3);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_set_encode_special_tokens(
    tokenizer: *mut CTokenizer,
    value: bool,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_set_encode_special_tokens received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &mut *tokenizer };
    tokenizer.inner.set_encode_special_tokens(value);
    clear_last_error();
    set_status(status, 0);
    1
}

#[no_mangle]
pub extern "C" fn tokenizers_get_encode_special_tokens(
    tokenizer: *const CTokenizer,
    status: *mut c_int,
) -> bool {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_encode_special_tokens received null pointer");
        set_status(status, 1);
        return false;
    }

    let tokenizer = unsafe { &*tokenizer };
    let value = tokenizer.inner.get_encode_special_tokens();
    clear_last_error();
    set_status(status, 0);
    value
}

#[no_mangle]
pub extern "C" fn tokenizers_num_special_tokens_to_add(
    tokenizer: *const CTokenizer,
    is_pair: bool,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_num_special_tokens_to_add received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &*tokenizer };
    let count = tokenizer
        .inner
        .get_post_processor()
        .map(|processor| processor.added_tokens(is_pair))
        .unwrap_or(0);

    if count > i32::MAX as usize {
        set_last_error("tokenizers_num_special_tokens_to_add overflowed");
        set_status(status, 2);
        return 0;
    }

    clear_last_error();
    set_status(status, 0);
    count as c_int
}

#[no_mangle]
pub extern "C" fn tokenizers_enable_padding(
    tokenizer: *mut CTokenizer,
    direction: c_int,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: *const c_char,
    length: c_int,
    pad_to_multiple_of: c_int,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_enable_padding received null pointer");
        set_status(status, 1);
        return 0;
    }

    let direction = match direction {
        0 => PaddingDirection::Left,
        1 => PaddingDirection::Right,
        other => {
            set_last_error(&format!(
                "tokenizers_enable_padding unknown direction: {other}"
            ));
            set_status(status, 2);
            return 0;
        }
    };

    let pad_token = match read_optional_utf8(pad_token) {
        Ok(Some(token)) => token,
        Ok(None) => String::from("[PAD]"),
        Err(_) => {
            set_last_error("tokenizers_enable_padding could not decode pad token");
            set_status(status, 3);
            return 0;
        }
    };

    let strategy = if length >= 0 {
        PaddingStrategy::Fixed(length as usize)
    } else {
        PaddingStrategy::BatchLongest
    };

    let pad_to_multiple = if pad_to_multiple_of > 0 {
        Some(pad_to_multiple_of as usize)
    } else {
        None
    };

    let tokenizer = unsafe { &mut *tokenizer };
    let params = PaddingParams {
        strategy,
        direction,
        pad_to_multiple_of: pad_to_multiple,
        pad_id,
        pad_type_id,
        pad_token,
    };

    tokenizer.inner.with_padding(Some(params));
    clear_last_error();
    set_status(status, 0);
    1
}

#[no_mangle]
pub extern "C" fn tokenizers_disable_padding(
    tokenizer: *mut CTokenizer,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_disable_padding received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &mut *tokenizer };
    tokenizer.inner.with_padding(None);
    clear_last_error();
    set_status(status, 0);
    1
}

#[no_mangle]
pub extern "C" fn tokenizers_get_padding(
    tokenizer: *const CTokenizer,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_padding received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    match tokenizer.inner.get_padding() {
        Some(params) => match serde_json::to_string(params) {
            Ok(json) => match CString::new(json) {
                Ok(c_string) => {
                    clear_last_error();
                    set_status(status, 0);
                    c_string.into_raw()
                }
                Err(_) => {
                    set_last_error("tokenizers_get_padding failed to allocate CString");
                    set_status(status, 2);
                    ptr::null_mut()
                }
            },
            Err(err) => {
                set_last_error(&format!("tokenizers_get_padding failed: {err}"));
                set_status(status, 3);
                ptr::null_mut()
            }
        },
        None => {
            clear_last_error();
            set_status(status, 0);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_enable_truncation(
    tokenizer: *mut CTokenizer,
    max_length: usize,
    stride: usize,
    strategy: c_int,
    direction: c_int,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_enable_truncation received null pointer");
        set_status(status, 1);
        return 0;
    }

    let strategy = match strategy {
        0 => TruncationStrategy::LongestFirst,
        1 => TruncationStrategy::OnlyFirst,
        2 => TruncationStrategy::OnlySecond,
        other => {
            set_last_error(&format!(
                "tokenizers_enable_truncation unknown strategy: {other}"
            ));
            set_status(status, 2);
            return 0;
        }
    };

    let direction = match direction {
        0 => TruncationDirection::Left,
        1 => TruncationDirection::Right,
        other => {
            set_last_error(&format!(
                "tokenizers_enable_truncation unknown direction: {other}"
            ));
            set_status(status, 3);
            return 0;
        }
    };

    let tokenizer = unsafe { &mut *tokenizer };
    let params = TruncationParams {
        max_length,
        stride,
        strategy,
        direction,
    };

    match tokenizer.inner.with_truncation(Some(params)) {
        Ok(_) => {
            clear_last_error();
            set_status(status, 0);
            1
        }
        Err(err) => {
            set_last_error(&format!("tokenizers_enable_truncation failed: {err}"));
            set_status(status, 4);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_disable_truncation(
    tokenizer: *mut CTokenizer,
    status: *mut c_int,
) -> c_int {
    if tokenizer.is_null() {
        set_last_error("tokenizers_disable_truncation received null pointer");
        set_status(status, 1);
        return 0;
    }

    let tokenizer = unsafe { &mut *tokenizer };
    match tokenizer.inner.with_truncation(None) {
        Ok(_) => {
            clear_last_error();
            set_status(status, 0);
            1
        }
        Err(err) => {
            set_last_error(&format!("tokenizers_disable_truncation failed: {err}"));
            set_status(status, 2);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_get_truncation(
    tokenizer: *const CTokenizer,
    status: *mut c_int,
) -> *mut c_char {
    if tokenizer.is_null() {
        set_last_error("tokenizers_get_truncation received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer };
    match tokenizer.inner.get_truncation() {
        Some(params) => match serde_json::to_string(params) {
            Ok(json) => match CString::new(json) {
                Ok(c_string) => {
                    clear_last_error();
                    set_status(status, 0);
                    c_string.into_raw()
                }
                Err(_) => {
                    set_last_error("tokenizers_get_truncation failed to allocate CString");
                    set_status(status, 2);
                    ptr::null_mut()
                }
            },
            Err(err) => {
                set_last_error(&format!("tokenizers_get_truncation failed: {err}"));
                set_status(status, 3);
                ptr::null_mut()
            }
        },
        None => {
            clear_last_error();
            set_status(status, 0);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_type_ids(
    encoding: *const CEncoding,
    buffer: *mut u32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.type_ids.len());
    unsafe {
        ptr::copy_nonoverlapping(encoding.type_ids.as_ptr(), buffer, copy_len);
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_attention_mask(
    encoding: *const CEncoding,
    buffer: *mut u32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.attention_mask.len());
    unsafe {
        ptr::copy_nonoverlapping(encoding.attention_mask.as_ptr(), buffer, copy_len);
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_special_tokens_mask(
    encoding: *const CEncoding,
    buffer: *mut u32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.special_tokens_mask.len());
    unsafe {
        ptr::copy_nonoverlapping(encoding.special_tokens_mask.as_ptr(), buffer, copy_len);
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_word_ids(
    encoding: *const CEncoding,
    buffer: *mut i32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.word_ids.len());

    for (index, word_id) in encoding.word_ids.iter().take(copy_len).enumerate() {
        unsafe {
            *buffer.add(index) = word_id.map(|id| id as i32).unwrap_or(-1);
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_sequence_ids(
    encoding: *const CEncoding,
    buffer: *mut i32,
    len: usize,
) {
    if encoding.is_null() || buffer.is_null() {
        return;
    }

    let encoding = unsafe { &*encoding };
    let copy_len = len.min(encoding.sequence_ids.len());

    for (index, seq_id) in encoding.sequence_ids.iter().take(copy_len).enumerate() {
        unsafe {
            *buffer.add(index) = seq_id.map(|id| id as i32).unwrap_or(-1);
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_overflowing_count(encoding: *const CEncoding) -> usize {
    if encoding.is_null() {
        return 0;
    }

    let encoding = unsafe { &*encoding };
    encoding.overflowing.len()
}

#[no_mangle]
pub extern "C" fn tokenizers_encoding_get_overflowing(
    encoding: *const CEncoding,
    index: usize,
    len_ptr: *mut usize,
    status: *mut c_int,
) -> *mut CEncoding {
    if encoding.is_null() {
        set_last_error("tokenizers_encoding_get_overflowing received null pointer");
        set_status(status, 1);
        return ptr::null_mut();
    }

    let encoding = unsafe { &*encoding };

    if index >= encoding.overflowing.len() {
        set_last_error("tokenizers_encoding_get_overflowing index out of bounds");
        set_status(status, 2);
        return ptr::null_mut();
    }

    let overflowing_encoding = &encoding.overflowing[index];
    let len = overflowing_encoding.ids.len();
    set_length(len_ptr, len);
    clear_last_error();
    set_status(status, 0);

    // Clone and box the overflowing encoding
    Box::into_raw(Box::new(overflowing_encoding.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_pointer_behaves() {
        assert!(ptr::null_mut::<CTokenizer>().is_null());
    }
}
