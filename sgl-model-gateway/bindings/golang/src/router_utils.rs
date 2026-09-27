//! Chat-request preparation, built on the gateway's public crate APIs.
//!
//! Mirrors the gRPC router's preparation steps so an out-of-tree FFI consumer
//! produces the same prompt text, tool constraint and stop decoder as the
//! in-process router.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use smg::protocols::chat::{ChatCompletionRequest, ChatMessage};
use smg::protocols::common::{StringOrArray, Tool, ToolChoice, ToolChoiceValue};
use smg::tokenizer::cache::CachedTokenizer;
use smg::tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use smg::tokenizer::stop::StopSequenceDecoderBuilder;
use smg::tokenizer::traits::Tokenizer;
use smg::tokenizer::{HuggingFaceTokenizer, StopSequenceDecoder};
use smg_grpc_client::sglang_proto::MultimodalInputs;

/// Chat messages rendered through the tokenizer's chat template
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<MultimodalInputs>,
}

/// Resolve the HuggingFace tokenizer behind `tokenizer`, unwrapping a
/// `CachedTokenizer` wrapper if one is present.
fn as_huggingface(tokenizer: &dyn Tokenizer) -> Option<&HuggingFaceTokenizer> {
    tokenizer
        .as_any()
        .downcast_ref::<HuggingFaceTokenizer>()
        .or_else(|| {
            tokenizer
                .as_any()
                .downcast_ref::<CachedTokenizer>()
                .and_then(|cached| cached.inner().as_any().downcast_ref::<HuggingFaceTokenizer>())
        })
}

/// Serialize messages and reshape each `content` field for the template's
/// expected content format.
fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json =
                serde_json::to_value(message).map_err(|e| format!("Failed to serialize message: {}", e))?;

            if let Some(obj) = message_json.as_object_mut() {
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format);
                }
            }

            Ok(message_json)
        })
        .collect()
}

fn transform_content_field(content_value: &mut Value, content_format: ChatTemplateContentFormat) {
    let Some(content_array) = content_value.as_array() else {
        return;
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            let text_parts: Vec<String> = content_array
                .iter()
                .filter_map(|part| {
                    part.as_object()?
                        .get("type")?
                        .as_str()
                        .filter(|&t| t == "text")
                        .and_then(|_| part.as_object()?.get("text")?.as_str())
                        .map(String::from)
                })
                .collect();

            if !text_parts.is_empty() {
                *content_value = Value::String(text_parts.join(" "));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

/// Rewrite assistant `tool_calls[].function.arguments` from JSON strings to objects;
/// templates index into them rather than re-parsing.
fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{}'. Error: {}",
                        args_str, e
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Apply the tokenizer's chat template to the request messages.
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<ProcessedMessages, String> {
    let hf_tokenizer = as_huggingface(tokenizer).ok_or_else(|| {
        "gRPC router requires HuggingFace tokenizer with chat template support".to_string()
    })?;

    let content_format = hf_tokenizer.chat_template_content_format();
    let mut transformed_messages = process_content_format(&request.messages, content_format)?;
    process_tool_call_arguments(&mut transformed_messages)?;

    let tools_json: Option<Vec<Value>> = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| format!("Failed to serialize tools: {}", e))?;

    let kwargs_capacity = 1 + request.chat_template_kwargs.as_ref().map_or(0, |k| k.len());
    let mut combined_template_kwargs = HashMap::with_capacity(kwargs_capacity);

    if let Some(reasoning_effort) = &request.reasoning_effort {
        combined_template_kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.clone()),
        );
    }

    if let Some(template_kwargs) = &request.chat_template_kwargs {
        for (key, value) in template_kwargs {
            combined_template_kwargs.insert(key.clone(), value.clone());
        }
    }

    let final_template_kwargs = if combined_template_kwargs.is_empty() {
        None
    } else {
        Some(&combined_template_kwargs)
    };

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: tools_json.as_deref(),
        template_kwargs: final_template_kwargs,
        ..Default::default()
    };

    // A trailing assistant message is continued rather than re-emitted, so it is
    // removed from the template input and appended to the rendered text instead.
    let assistant_prefix = if request.continue_final_message
        && !transformed_messages.is_empty()
        && transformed_messages
            .last()
            .and_then(|msg| msg.get("role"))
            .and_then(|v| v.as_str())
            == Some("assistant")
    {
        let last_msg = transformed_messages.pop().unwrap();
        last_msg
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    let rendered = hf_tokenizer
        .apply_chat_template(&transformed_messages, params)
        .map_err(|e| format!("Failed to apply chat template: {}", e))?;

    let text = match assistant_prefix {
        Some(prefix) => format!("{}{}", rendered, prefix),
        None => rendered,
    };

    Ok(ProcessedMessages {
        text,
        multimodal_inputs: None,
    })
}

/// Build the structured-output constraint implied by `tool_choice`, as
/// `(constraint_type, constraint_value)` for `SamplingParams`.
///
/// `tools` must already be narrowed to the tools the choice permits.
pub fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    _model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice.as_ref() else {
        return Ok(None);
    };

    match choice {
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];

            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {}", e))?;
            Ok(Some((String::from("json_schema"), params_schema)))
        }

        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_array_schema(tools)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                Ok(None)
            }
        }

        _ => Ok(None),
    }
}

/// Array schema forcing at least one tool call, with every tool's `$defs` hoisted
/// to the top level so shared `$ref`s still resolve.
fn build_required_array_schema(tools: &[Tool]) -> Result<String, String> {
    let mut any_of_schemas = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    let mut all_defs: Map<String, Value> = Map::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        if existing != def_schema {
                            let error_msg = format!(
                                "Tool definition '{}' has multiple conflicting schemas, which is not supported",
                                def_name
                            );
                            tracing::error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("$defs".to_string(), Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {}", e))
}

/// Stop decoder for the request's stop parameters; `no_stop_trim` keeps the
/// matched stop in the emitted text instead of swallowing it.
pub fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
) -> StopSequenceDecoder {
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

    if let Some(token_ids) = stop_token_ids {
        for &token_id in token_ids {
            builder = if no_stop_trim {
                builder.visible_stop_token(token_id)
            } else {
                builder.stop_token(token_id)
            };
        }
    }

    builder.build()
}
