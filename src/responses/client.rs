//! Shared client logic for providers that use the OpenAI-style responses API.
//!
//! This module contains reusable functionality for:
//! - Building requests from core types
//! - Handling tool calling loops (parallel and sequential)
//! - Converting between core and API message formats
//! - Serializing tools and generating JSON schemas
//! - Parsing API responses back to core types

use crate::{
    Provider,
    core::{
        ChatRole, ConversationMessage, LanguageModelUsage, LlmError, ResponseMetadata,
        StructuredRequest, StructuredResponse, Tool, ToolCall, ToolRegistry,
    },
    responses::{
        Format, FormatType, FunctionToolCall, FunctionToolCallOutput, JsonSchema, JsonSchemaType,
        request::{InputItem, InputMessage, InputMessageRole, Request},
        response::{MessageContent, OutputContent, Response},
    },
};
use schemars::schema_for;
use serde::Deserialize;

/// Configuration trait for providers that use the OpenAI-style responses API
pub trait ResponsesProviderConfig {
    /// Model Provider
    fn provider(&self) -> Provider;

    /// Base URL for the API (e.g., `https://api.openai.com`)
    fn base_url(&self) -> &str;

    /// API endpoint for responses (e.g., `/v1/responses`)
    fn endpoint(&self) -> &str;

    /// Authentication header as (header_name, header_value) tuple
    fn auth_header(&self) -> (String, String);

    /// Additional headers to include with each request
    fn extra_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// Shared client for providers using the OpenAI-style responses API
pub struct ResponsesClient<P: ResponsesProviderConfig> {
    pub config: P,
    client: reqwest::Client,
}

impl<P: ResponsesProviderConfig> ResponsesClient<P> {
    /// Create a new responses client with the given configuration
    pub fn new(config: P) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Make an API request to the responses endpoint
    pub async fn make_api_request(&self, request: Request) -> Result<Response, LlmError> {
        let url = format!("{}{}", self.config.base_url(), self.config.endpoint());

        let mut req_builder = self.client.post(&url).json(&request);

        // Add authentication header
        let (auth_name, auth_value) = self.config.auth_header();
        req_builder = req_builder.header(&auth_name, auth_value);

        // Add extra headers
        for (name, value) in self.config.extra_headers() {
            req_builder = req_builder.header(&name, value);
        }

        let res = req_builder.send().await.map_err(|e| LlmError::Network {
            message: "Failed to complete request".to_string(),
            source: Box::new(e),
        })?;

        if !res.status().is_success() {
            let status = res.status();
            let error_text = res
                .text()
                .await
                .map_err(|e| LlmError::Api {
                    message: "Failed to get the response text".to_string(),
                    status_code: Some(status.as_u16()),
                    source: Some(Box::new(e)),
                })?
                .clone();

            return Err(LlmError::Api {
                message: format!("API returned error: {error_text}"),
                status_code: Some(status.as_u16()),
                source: None,
            });
        }

        res.json().await.map_err(|e| LlmError::Parse {
            message: "Failed to parse API response".to_string(),
            source: Box::new(e),
        })
    }

    /// Handle the complete tool calling loop until a final response is received
    pub async fn handle_tool_calling_loop<T>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        let mut responses_input = convert_messages_to_responses_format(request.messages.clone())?;
        let is_parallel = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.parallel_tool_calls)
            .unwrap_or(true);

        loop {
            let responses_request = self.build_request::<T>(&request, &responses_input)?;
            let api_response = self.make_api_request(responses_request).await?;

            let function_calls = self.extract_function_calls(&api_response);

            if function_calls.is_empty() {
                return create_core_structured_response(api_response, self.config.provider());
            }

            self.process_function_calls(
                &function_calls,
                &mut responses_input,
                tool_registry,
                is_parallel,
            )
            .await?;
        }
    }

    /// Build a responses API request from core request and input
    pub fn build_request<T>(
        &self,
        request: &StructuredRequest,
        responses_input: &[InputItem],
    ) -> Result<Request, LlmError>
    where
        T: schemars::JsonSchema,
    {
        let mut req = Request {
            model: request.model.clone(),
            input: responses_input.to_vec(),
            text: create_format_for_type::<T>()?,
            // Default fields
            parallel_tool_calls: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            store: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            user: None,
        };

        // Apply tool configuration if present
        if let Some(tool_config) = &request.tool_config {
            req.tools = tool_config.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .map(crate::responses::create_function_tool)
                    .collect::<Box<[crate::responses::Tool]>>()
            });
            req.tool_choice = tool_config.tool_choice.as_ref().map(|tc| tc.clone().into());
            req.parallel_tool_calls = tool_config.parallel_tool_calls;
        }

        // Apply generation configuration if present
        if let Some(gen_config) = &request.generation_config {
            req.temperature = gen_config.temperature;
            req.max_output_tokens = gen_config.max_tokens;
            req.top_p = gen_config.top_p;
        }

        Ok(req)
    }

    /// Extract function calls from API response
    pub fn extract_function_calls<'a>(
        &self,
        api_response: &'a Response,
    ) -> Vec<&'a super::types::FunctionToolCall> {
        api_response
            .output
            .iter()
            .filter_map(|output| match output {
                OutputContent::FunctionCall(fc) => Some(fc),
                OutputContent::OutputMessage(_) => None,
            })
            .collect()
    }

    /// Process function calls either in parallel or sequentially
    pub async fn process_function_calls(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry,
        is_parallel: bool,
    ) -> Result<(), LlmError> {
        if is_parallel && function_calls.len() > 1 {
            self.process_parallel_function_calls(function_calls, responses_input, tool_registry)
                .await
        } else {
            self.process_sequential_function_calls(function_calls, responses_input, tool_registry)
                .await
        }
    }

    /// Process function calls in parallel (all calls first, then all results)
    pub async fn process_parallel_function_calls(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry,
    ) -> Result<(), LlmError> {
        let mut pending_executions = Vec::new();

        // Add all function calls to input and prepare for execution
        for function_call in function_calls {
            responses_input.push(InputItem::FunctionCall((*function_call).clone()));

            let arguments = self.parse_function_arguments(&function_call.arguments)?;
            pending_executions.push((
                function_call.id.clone(),
                function_call.call_id.clone(),
                function_call.name.clone(),
                arguments,
            ));
        }

        // Execute all tools and add their results
        for (id, call_id, name, arguments) in pending_executions {
            let tool_call = ToolCall {
                id,
                call_id: call_id.clone(),
                name,
                arguments,
            };
            let result = tool_registry.execute(&tool_call).await?;

            responses_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                call_id,
                output: result,
                r#type: "function_call_output".to_string(),
            }));
        }

        Ok(())
    }

    /// Process function calls sequentially (call and result pairs)
    pub async fn process_sequential_function_calls(
        &self,
        function_calls: &[&FunctionToolCall],
        responses_input: &mut Vec<InputItem>,
        tool_registry: &ToolRegistry,
    ) -> Result<(), LlmError> {
        for function_call in function_calls {
            responses_input.push(InputItem::FunctionCall((*function_call).clone()));

            let arguments = self.parse_function_arguments(&function_call.arguments)?;
            let tool_call = ToolCall {
                id: function_call.id.clone(),
                call_id: function_call.call_id.clone(),
                name: function_call.name.clone(),
                arguments,
            };

            let result = tool_registry.execute(&tool_call).await?;

            responses_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                call_id: function_call.call_id.clone(),
                output: result,
                r#type: "function_call_output".to_string(),
            }));
        }

        Ok(())
    }

    /// Parse function arguments from JSON value
    pub fn parse_function_arguments(
        &self,
        arguments: &serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        match arguments {
            serde_json::Value::String(s) => serde_json::from_str(s).map_err(|e| LlmError::Parse {
                message: format!("Failed to parse tool arguments: {s}"),
                source: Box::new(e),
            }),
            other => Ok(other.clone()),
        }
    }
}

/// Wrapper for non-object JSON Schema types (enums, strings, numbers, etc.)
/// OpenAI's structured output API requires the root schema to be an object,
/// so we wrap non-object types in an object with a "value" property.
#[derive(Deserialize)]
pub(crate) struct ValueWrapper<T> {
    value: T,
}

/// Convert core Tool to responses API Tool
pub(crate) fn create_function_tool(tool: &Tool) -> crate::responses::Tool {
    let strict = tool.strict.unwrap_or(true);
    let mut parameters = tool.parameters.clone();

    // OpenAI strict mode requires ALL properties to be in the required array
    if strict && let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) {
        let all_property_names: Vec<serde_json::Value> = properties
            .keys()
            .map(|k| serde_json::Value::String(k.clone()))
            .collect();

        if let Some(params_obj) = parameters.as_object_mut() {
            params_obj.insert(
                "required".to_string(),
                serde_json::Value::Array(all_property_names),
            );
        }
    }

    crate::responses::Tool {
        name: tool.name.clone(),
        parameters,
        strict: Some(strict),
        description: tool.description.clone(),
    }
}

/// Convert core messages to responses API format
pub(crate) fn convert_messages_to_responses_format(
    messages: Vec<ConversationMessage>,
) -> Result<Vec<InputItem>, LlmError> {
    messages
        .into_iter()
        .map(|msg| match msg {
            ConversationMessage::Chat(m) => Ok(InputItem::Message(InputMessage {
                role: match m.role {
                    ChatRole::System => InputMessageRole::System,
                    ChatRole::User => InputMessageRole::User,
                    ChatRole::Assistant => InputMessageRole::Assistant,
                },
                content: m.content,
            })),
            ConversationMessage::ToolCall(tc) => Ok(InputItem::FunctionCall(FunctionToolCall {
                r#type: "function_call".to_string(),
                id: tc.id,
                call_id: tc.call_id,
                name: tc.name,
                arguments: serde_json::Value::String(
                    serde_json::to_string(&tc.arguments).map_err(|e| LlmError::Parse {
                        message: "Failed to serialize tool call arguments".to_string(),
                        source: Box::new(e),
                    })?,
                ),
            })),
            ConversationMessage::ToolCallResult(tr) => {
                Ok(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                    call_id: tr.tool_call_id,
                    output: tr.content,
                    r#type: "function_call_output".to_string(),
                }))
            }
        })
        .collect()
}

/// Create JSON schema format for a given type
pub(crate) fn create_format_for_type<T>() -> Result<Format, LlmError>
where
    T: schemars::JsonSchema,
{
    let s = schema_for!(T);

    let obj = s.as_object().ok_or_else(|| LlmError::Provider {
        message: "Failed to build JSON Schema: root is not an object".to_string(),
        source: None,
    })?;

    let schema_name = obj
        .get("title")
        .ok_or_else(|| LlmError::Provider {
            message: "Failed to build JSON Schema: Missing schema name".to_string(),
            source: None,
        })?
        .as_str()
        .ok_or_else(|| LlmError::Provider {
            message: "Failed to build JSON Schema: title is not a string".to_string(),
            source: None,
        })?
        .to_owned();

    let mut schema_value = serde_json::to_value(&s).map_err(|e| LlmError::Parse {
        message: "Failed to build JSON Schema".to_string(),
        source: Box::new(e),
    })?;

    let needs_wrapping = schema_value
        .get("type")
        .and_then(|t| t.as_str())
        .map(|t| t != "object")
        .unwrap_or(false);

    if needs_wrapping {
        schema_value = serde_json::json!({
            "type": "object",
            "properties": {
                "value": schema_value
            },
            "required": ["value"],
            "additionalProperties": false
        })
    }

    Ok(Format {
        format: FormatType::JsonSchema(JsonSchema {
            name: schema_name,
            schema: schema_value,
            r#type: JsonSchemaType::JsonSchema,
        }),
    })
}

/// Convert API response to core structured response with specified provider
pub(crate) fn create_core_structured_response<T>(
    res: Response,
    provider: crate::provider::Provider,
) -> Result<StructuredResponse<T>, LlmError>
where
    T: serde::de::DeserializeOwned,
{
    let output_content = res.output.first().ok_or_else(|| LlmError::Provider {
        message: "No output in response".to_string(),
        source: None,
    })?;

    match output_content {
        OutputContent::OutputMessage(message) => {
            let content = message.content.first().ok_or_else(|| LlmError::Provider {
                message: "No content in message".to_string(),
                source: None,
            })?;

            let text = match content {
                MessageContent::OutputText(output) => &output.text,
                MessageContent::Refusal(refusal) => {
                    return Err(LlmError::Api {
                        message: format!("Model refused: {}", refusal.refusal),
                        status_code: None,
                        source: None,
                    });
                }
            };

            // Try to parse as wrapped value first, then fall back to direct parsing
            let parsed_content: T =
                if let Ok(wrapped) = serde_json::from_str::<ValueWrapper<T>>(text) {
                    wrapped.value
                } else {
                    serde_json::from_str(text).map_err(|e| LlmError::Parse {
                        message: "Failed to parse structured output".to_string(),
                        source: Box::new(e),
                    })?
                };

            Ok(StructuredResponse {
                content: parsed_content,
                usage: LanguageModelUsage {
                    prompt_tokens: res.usage.input_tokens,
                    completion_tokens: res.usage.output_tokens,
                    total_tokens: res.usage.total_tokens,
                },
                metadata: ResponseMetadata {
                    provider,
                    model: res.model,
                    id: res.id,
                },
            })
        }
        OutputContent::FunctionCall(_) => Err(LlmError::Provider {
            message: "Function call response received when expecting structured output".to_string(),
            source: None,
        }),
    }
}
