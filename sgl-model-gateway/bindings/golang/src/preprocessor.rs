//! Preprocessing FFI functions.

use std::ffi::{CStr, CString};
use std::os::raw::c_uint;
use std::os::raw::{c_char, c_int};
use std::ptr;

use super::error::{set_error_message, SglErrorCode};
use super::memory::{sgl_free_string, sgl_free_token_ids};
use super::router_utils::{generate_tool_constraints, process_chat_messages};
use super::tokenizer::TokenizerHandle;
use serde_json::Value;
use smg::protocols::chat::ChatCompletionRequest;
use smg::tokenizer::create_tokenizer_from_file;

/// Tool constraint in the shape the Go caller unmarshals: an object keyed by
/// constraint type.
fn constraint_json(constraint_type: &str, constraint_value: &str) -> Value {
    serde_json::json!({ constraint_type: constraint_value })
}

/// Preprocess a chat completion request This function:. Applies chat_template to messages.
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocess_chat_request(
    request_json: *const c_char,
    tokenizer_path: *const c_char,
    prompt_text_out: *mut *mut c_char,
    token_ids_out: *mut *mut c_uint,
    token_ids_len_out: *mut usize,
    tool_constraints_json_out: *mut *mut c_char,
    prompt_tokens_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if request_json.is_null()
        || tokenizer_path.is_null()
        || prompt_text_out.is_null()
        || token_ids_out.is_null()
        || token_ids_len_out.is_null()
        || prompt_tokens_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    // Parse input strings
    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse ChatCompletionRequest
    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Create tokenizer
    let tokenizer = match create_tokenizer_from_file(tokenizer_path_str) {
        Ok(t) => t,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create tokenizer: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };

    // Process chat messages (apply chat_template)
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref()) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to process chat messages: {}", e),
            );
            return SglErrorCode::ParsingError;
        }
    };

    // Tokenize the processed text
    let encoding = match tokenizer.encode(&processed_messages.text, false) {
        Ok(enc) => enc,
        Err(e) => {
            set_error_message(error_out, &format!("Tokenization failed: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };

    let token_ids_vec: Vec<i32> = encoding.token_ids().iter().map(|&id| id as i32).collect();

    let prompt_tokens = token_ids_vec.len() as i32;

    // Generate tool constraints if tools are present
    let tool_constraints_json = if let Some(tools) = chat_request.tools.as_ref() {
        match generate_tool_constraints(tools, &chat_request.tool_choice, &chat_request.model) {
            Ok(Some((constraint_type, constraint_value))) => {
                match serde_json::to_string(&constraint_json(&constraint_type, &constraint_value)) {
                    Ok(json_str) => Some(CString::new(json_str).unwrap()),
                    Err(e) => {
                        set_error_message(
                            error_out,
                            &format!("Failed to serialize tool constraints: {}", e),
                        );
                        return SglErrorCode::ParsingError;
                    }
                }
            }
            Ok(None) => None,
            Err(e) => {
                set_error_message(
                    error_out,
                    &format!("Failed to generate tool constraints: {}", e),
                );
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Allocate memory for outputs
    let prompt_text_cstr = match CString::new(processed_messages.text) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {}", e));
            return SglErrorCode::MemoryError;
        }
    };

    let token_ids_len = token_ids_vec.len();
    // Convert i32 to u32 for token IDs (as expected by the memory management functions)
    let token_ids_u32: Vec<u32> = token_ids_vec.iter().map(|&id| id as u32).collect();
    let token_ids_ptr = if token_ids_u32.is_empty() {
        ptr::null_mut()
    } else {
        let boxed = token_ids_u32.into_boxed_slice();
        Box::into_raw(boxed) as *mut c_uint
    };

    // Set output values
    *prompt_text_out = prompt_text_cstr.into_raw();
    *token_ids_out = token_ids_ptr;
    *token_ids_len_out = token_ids_len;
    *prompt_tokens_out = prompt_tokens;

    if !tool_constraints_json_out.is_null() {
        if let Some(constraints) = tool_constraints_json {
            *tool_constraints_json_out = constraints.into_raw();
        } else {
            *tool_constraints_json_out = ptr::null_mut();
        }
    }

    SglErrorCode::Success
}

/// Preprocess a chat completion request using an existing tokenizer handle This function is similar to sgl_preprocess_chat_request.
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocess_chat_request_with_tokenizer(
    request_json: *const c_char,
    tokenizer_handle: *mut TokenizerHandle,
    prompt_text_out: *mut *mut c_char,
    token_ids_out: *mut *mut c_uint,
    token_ids_len_out: *mut usize,
    tool_constraints_json_out: *mut *mut c_char,
    prompt_tokens_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if request_json.is_null()
        || tokenizer_handle.is_null()
        || prompt_text_out.is_null()
        || token_ids_out.is_null()
        || token_ids_len_out.is_null()
        || prompt_tokens_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    // Parse input string
    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse ChatCompletionRequest
    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Use existing tokenizer from handle (no need to create new one!)
    let handle_ref = &*tokenizer_handle;
    let tokenizer = &handle_ref.tokenizer;

    // Process chat messages (apply chat_template)
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref()) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to process chat messages: {}", e),
            );
            return SglErrorCode::ParsingError;
        }
    };

    // Tokenize the processed text
    let encoding = match tokenizer.encode(&processed_messages.text, false) {
        Ok(enc) => enc,
        Err(e) => {
            set_error_message(error_out, &format!("Tokenization failed: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };

    let token_ids_vec: Vec<i32> = encoding.token_ids().iter().map(|&id| id as i32).collect();

    let prompt_tokens = token_ids_vec.len() as i32;

    // Generate tool constraints if tools are present
    let tool_constraints_json = if let Some(tools) = chat_request.tools.as_ref() {
        match generate_tool_constraints(tools, &chat_request.tool_choice, &chat_request.model) {
            Ok(Some((constraint_type, constraint_value))) => {
                match serde_json::to_string(&constraint_json(&constraint_type, &constraint_value)) {
                    Ok(json_str) => Some(CString::new(json_str).unwrap()),
                    Err(e) => {
                        set_error_message(
                            error_out,
                            &format!("Failed to serialize tool constraints: {}", e),
                        );
                        return SglErrorCode::ParsingError;
                    }
                }
            }
            Ok(None) => None,
            Err(e) => {
                set_error_message(
                    error_out,
                    &format!("Failed to generate tool constraints: {}", e),
                );
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Allocate memory for outputs
    let prompt_text_cstr = match CString::new(processed_messages.text) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {}", e));
            return SglErrorCode::MemoryError;
        }
    };

    let token_ids_len = token_ids_vec.len();
    // Convert i32 to u32 for token IDs (as expected by the memory management functions)
    let token_ids_u32: Vec<u32> = token_ids_vec.iter().map(|&id| id as u32).collect();
    let token_ids_ptr = if token_ids_u32.is_empty() {
        ptr::null_mut()
    } else {
        let boxed = token_ids_u32.into_boxed_slice();
        Box::into_raw(boxed) as *mut c_uint
    };

    // Set output values
    *prompt_text_out = prompt_text_cstr.into_raw();
    *token_ids_out = token_ids_ptr;
    *token_ids_len_out = token_ids_len;
    *prompt_tokens_out = prompt_tokens;

    if !tool_constraints_json_out.is_null() {
        if let Some(constraints) = tool_constraints_json {
            *tool_constraints_json_out = constraints.into_raw();
        } else {
            *tool_constraints_json_out = ptr::null_mut();
        }
    }

    SglErrorCode::Success
}

/// Free a preprocessed request handle (cleanup function) This function frees the memory allocated.
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocessed_request_free(
    prompt_text: *mut c_char,
    token_ids: *mut c_uint,
    token_ids_len: usize,
    tool_constraints_json: *mut c_char,
) {
    if !prompt_text.is_null() {
        sgl_free_string(prompt_text);
    }

    if !token_ids.is_null() && token_ids_len > 0 {
        sgl_free_token_ids(token_ids, token_ids_len);
    }

    if !tool_constraints_json.is_null() {
        sgl_free_string(tool_constraints_json);
    }
}

#[cfg(test)]
mod tests {
    use super::constraint_json;
    use std::collections::HashMap;

    /// `internal/grpc/client_grpc.go` decodes this as `map[string]string` and reads the constraint type back out as the map key.
    #[test]
    fn constraint_decodes_as_a_map_keyed_by_constraint_type() {
        let schema = r#"{"type":"object","properties":{"city":{"type":"string"}}}"#;
        let encoded = serde_json::to_string(&constraint_json("json_schema", schema)).unwrap();

        let decoded: HashMap<String, String> = serde_json::from_str(&encoded)
            .expect("Go reads the constraint as map[string]string; it must decode as one");

        assert_eq!(decoded.get("json_schema").map(String::as_str), Some(schema));
    }
}
