//! Preprocessing FFI functions for chat requests
//!
//! This module provides C-compatible functions for preprocessing chat completion requests:
//! - Apply chat_template to messages
//! - Tokenize the processed text
//! - Generate tool constraints
//!
//! These functions are designed to be called once per request, reducing FFI overhead.

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_uint},
    ptr,
};

use smg::{
    protocols::chat::ChatCompletionRequest,
    routers::grpc::utils::{
        filter_chat_request_by_tool_choice, generate_tool_constraints, process_chat_messages,
    },
    tokenizer::create_tokenizer_from_file,
};

use super::{
    error::{set_error_message, SglErrorCode},
    memory::{sgl_free_string, sgl_free_token_ids},
    tokenizer::TokenizerHandle,
};

/// Preprocess a chat completion request
///
/// This function:
/// 1. Applies chat_template to messages
/// 2. Tokenizes the processed text
/// 3. Generates tool constraints (if tools are present)
///
/// # Arguments
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `tokenizer_path` - Path to tokenizer directory
/// * `prompt_text_out` - Pointer to receive prompt text (C string, must be freed with sgl_free_string)
/// * `token_ids_out` - Pointer to receive token IDs array (must be freed with sgl_free_token_ids)
/// * `token_ids_len_out` - Pointer to receive token IDs array length
/// * `tool_constraints_json_out` - Optional pointer to receive tool constraints JSON (must be freed with sgl_free_string)
/// * `prompt_tokens_out` - Pointer to receive prompt token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
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
    let parsed_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };
    let chat_request = filter_chat_request_by_tool_choice(&parsed_request);

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

    let tool_constraints_json = match tool_constraints_json(&chat_request) {
        Ok(json) => json,
        Err(e) => {
            set_error_message(error_out, &e);
            return SglErrorCode::ParsingError;
        }
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

/// Tool constraint of an already tool-choice-filtered request, as the JSON object
/// `{"<constraint type>": "<constraint value>"}` that the Go client reads.
fn tool_constraints_json(request: &ChatCompletionRequest) -> Result<Option<CString>, String> {
    let Some(tools) = request.tools.as_ref() else {
        return Ok(None);
    };
    let constraint = generate_tool_constraints(tools, &request.tool_choice, &request.model)
        .map_err(|e| format!("Failed to generate tool constraints: {}", e))?;
    let Some((constraint_type, constraint_value)) = constraint else {
        return Ok(None);
    };
    let mut object = serde_json::Map::new();
    object.insert(constraint_type, serde_json::Value::String(constraint_value));
    let json = serde_json::Value::Object(object).to_string();
    CString::new(json)
        .map(Some)
        .map_err(|e| format!("Failed to create C string: {}", e))
}

/// Preprocess a chat completion request using an existing tokenizer handle
///
/// This function is similar to sgl_preprocess_chat_request, but accepts a TokenizerHandle
/// instead of creating a new tokenizer. This allows reusing a cached tokenizer instance,
/// significantly reducing initialization overhead in concurrent scenarios.
///
/// # Arguments
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `tokenizer_handle` - Existing tokenizer handle (must be valid)
/// * `prompt_text_out` - Pointer to receive prompt text (C string, must be freed with sgl_free_string)
/// * `token_ids_out` - Pointer to receive token IDs array (must be freed with sgl_free_token_ids)
/// * `token_ids_len_out` - Pointer to receive token IDs array length
/// * `tool_constraints_json_out` - Optional pointer to receive tool constraints JSON (must be freed with sgl_free_string)
/// * `prompt_tokens_out` - Pointer to receive prompt token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
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
    let parsed_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };
    let chat_request = filter_chat_request_by_tool_choice(&parsed_request);

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

    let tool_constraints_json = match tool_constraints_json(&chat_request) {
        Ok(json) => json,
        Err(e) => {
            set_error_message(error_out, &e);
            return SglErrorCode::ParsingError;
        }
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

/// Free a preprocessed request handle (cleanup function)
///
/// This function frees the memory allocated by sgl_preprocess_chat_request.
/// It should be called after the preprocessed data is no longer needed.
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
    use serde_json::{json, Value};

    use super::*;

    fn request(tool_choice: Value) -> ChatCompletionRequest {
        let tool = |name: &str, arg: &str| {
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": {arg: {"type": "string"}}}
                }
            })
        };
        serde_json::from_value(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool("get_weather", "city"), tool("get_time", "zone")],
            "tool_choice": tool_choice,
        }))
        .unwrap()
    }

    fn constraints(tool_choice: Value) -> Option<Value> {
        let parsed = request(tool_choice);
        let filtered = filter_chat_request_by_tool_choice(&parsed);
        tool_constraints_json(&filtered)
            .unwrap()
            .map(|c| serde_json::from_str(c.to_str().unwrap()).unwrap())
    }

    #[test]
    fn named_function_constrains_to_that_function() {
        let json = constraints(json!({"type": "function", "function": {"name": "get_time"}}))
            .expect("a named function yields a constraint");
        // The Go client reads a JSON object keyed by constraint type.
        let schema: Value = serde_json::from_str(json["json_schema"].as_str().unwrap()).unwrap();
        assert!(schema["properties"].get("zone").is_some(), "{schema}");
    }

    #[test]
    fn required_constrains_to_any_tool() {
        let json = constraints(json!("required")).expect("required yields a constraint");
        let schema: Value = serde_json::from_str(json["json_schema"].as_str().unwrap()).unwrap();
        assert_eq!(schema["items"]["anyOf"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn auto_has_no_constraint() {
        assert_eq!(constraints(json!("auto")), None);
    }
}
