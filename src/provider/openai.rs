//! OpenAI provider implementation.
//!
//! # API Compatibility
//!
//! This module preserves all fields from the OpenAI API responses, even those not currently used.
//! Fields marked with `#[allow(dead_code)]` are retained for:
//! - API contract completeness
//! - Future compatibility without breaking changes
//! - Debugging and logging purposes
//!
//! When adding new API structs, include all fields from the OpenAI documentation and mark
//! unused ones with `#[allow(dead_code)]` rather than omitting them.

use crate::core::{
    self,
    types::{ConversationMessage, StructuredRequest, StructuredResponse, ToolCall, ToolRegistry},
};
use crate::provider::constants::openai;
use async_trait::async_trait;
use schemars::schema_for;
use serde::{Deserialize, Serialize};

/// Wrapper for non-object JSON Schema types (enums, strings, numbers, etc.)
/// OpenAI's structured output API requires the root schema to be an object,
/// so we wrap non-object types in an object with a "value" property.
#[derive(Deserialize)]
struct ValueWrapper<T> {
    value: T,
}

use crate::core::{builder::LlmBuilder, error::LlmError, traits::LlmProvider};

use super::Provider;

pub struct OpenAiClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    default_model: String,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openai::API_BASE.to_string(),
            client: reqwest::Client::new(),
            default_model: openai::DEFAULT_MODEL.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_default_model(mut self, model: String) -> Self {
        self.default_model = model;
        self
    }

    async fn make_api_request(
        &self,
        request: OpenAiStructuredRequest,
    ) -> Result<OpenAiStructuredResponse, LlmError> {
        let url = format!("{}{}", self.base_url, openai::RESPONSES_ENDPOINT);

        let res = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Network {
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
                message: format!("OpenAI API returned error: {error_text}"),
                status_code: Some(status.as_u16()),
                source: None,
            });
        }

        res.json().await.map_err(|e| LlmError::Parse {
            message: "Failed to parse OpenAI response".to_string(),
            source: Box::new(e),
        })
    }

    async fn handle_tool_calling_loop<T>(
        &self,
        request: StructuredRequest,
        tool_registry: &ToolRegistry,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        // Convert initial request to OpenAI format once
        let mut openai_input = convert_messages_to_openai_format(request.messages)?;

        loop {
            let openai_request = OpenAiStructuredRequest {
                model: request.model.clone(),
                input: openai_input.clone(),
                text: create_format_for_type::<T>()?,
                tools: request.tools.as_ref().map(|tools| {
                    tools
                        .iter()
                        .map(create_function_tool)
                        .collect::<Box<[Tool]>>()
                }),
                tool_choice: request
                    .tool_choice
                    .as_ref()
                    .map(|tc| create_function_tool_choice(tc.clone())),
                parallel_tool_calls: request.parallel_tool_calls,
            };

            let api_response = self.make_api_request(openai_request).await?;

            // Check if response contains function calls
            let function_calls: Vec<&FunctionToolCall> = api_response
                .output
                .iter()
                .filter_map(|output| match output {
                    OutputContent::FunctionCall(fc) => Some(fc),
                    OutputContent::OutputMessage(_) => None,
                })
                .collect();

            if function_calls.is_empty() {
                // No more tool calls, return the structured response
                return create_core_structured_response(api_response);
            }

            // Handle tool calls based on whether parallel execution is enabled
            let is_parallel = request.parallel_tool_calls.unwrap_or(true);

            if is_parallel && function_calls.len() > 1 {
                // For parallel tool calls: add all tool calls first, then all results
                let mut pending_executions = Vec::new();

                // Add all function calls to the input
                for function_call in &function_calls {
                    openai_input.push(InputItem::FunctionCall((*function_call).clone()));

                    // Parse arguments for execution
                    let arguments = match &function_call.arguments {
                        serde_json::Value::String(s) => {
                            serde_json::from_str(s).map_err(|e| LlmError::Parse {
                                message: format!("Failed to parse tool arguments: {s}"),
                                source: Box::new(e),
                            })?
                        }
                        other => other.clone(),
                    };

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
                        id: id.clone(),
                        call_id: call_id.clone(),
                        name,
                        arguments,
                    };

                    let result = tool_registry.execute(&tool_call).await?;

                    openai_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                        call_id,
                        output: serde_json::Value::String(result),
                        r#type: "function_call_output".to_string(),
                    }));
                }
            } else {
                // For sequential tool calls: add tool call and result pairs one by one
                for function_call in function_calls {
                    // Add the function call
                    openai_input.push(InputItem::FunctionCall((*function_call).clone()));

                    // Parse arguments for execution
                    let arguments = match &function_call.arguments {
                        serde_json::Value::String(s) => {
                            serde_json::from_str(s).map_err(|e| LlmError::Parse {
                                message: format!("Failed to parse tool arguments: {s}"),
                                source: Box::new(e),
                            })?
                        }
                        other => other.clone(),
                    };

                    let tool_call = ToolCall {
                        id: function_call.id.clone(),
                        call_id: function_call.call_id.clone(),
                        name: function_call.name.clone(),
                        arguments,
                    };

                    // Execute tool and add result immediately
                    let result = tool_registry.execute(&tool_call).await?;

                    openai_input.push(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                        call_id: function_call.call_id.clone(),
                        output: serde_json::Value::String(result),
                        r#type: "function_call_output".to_string(),
                    }));
                }
            }
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn generate_structured<T>(
        &self,
        request: StructuredRequest,
        tool_registry: Option<&ToolRegistry>,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        // If tools are present and we have a registry, handle automatic tool calling
        if request.tools.is_some() && tool_registry.is_some() {
            return self
                .handle_tool_calling_loop(request, tool_registry.unwrap())
                .await;
        }

        // Otherwise, make a single request expecting structured content
        let openai_request = create_openai_structured_request::<T>(request)?;
        let api_response = self.make_api_request(openai_request).await?;
        create_core_structured_response(api_response)
    }
}

#[derive(Debug, Serialize)]
struct OpenAiStructuredRequest {
    model: String,
    input: Vec<InputItem>,
    text: Format,

    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Box<[Tool]>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(ToolMode),
    Definite(ToolChoiceDefinite),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoiceDefinite {
    // TODO
    // Hosted(HostedToolChoice),
    Function(FunctionToolChoice),
}

// #[derive(Debug, Serialize)]
// #[serde(rename_all = "snake_case")]
// enum HostedToolChoice {
//     // FileSearch,
//     // WebSearchPreview,
//     // ComputerUsePreview,
//     // CodeInterpreter,
//     // Mcp,
//     // ImageGeneration,
// }

#[derive(Debug, Serialize)]
/// Use this option to force the model to call a specific function.
struct FunctionToolChoice {
    /// The name of the function to call.
    name: String,

    #[serde(rename = "type")]
    r#type: FunctionType,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Tool {
    Function(FunctionTool),
    // TODO: Add file search tool
    // TODO: Add web search tool
    // TODO: Add computer use tool
    // TODO: Add MCP Tool
    // TODO: Add code interpreter tool
    // TODO: Add image generation tool
    // TODO: Add local shell tool
}

#[derive(Debug, Serialize)]
struct FunctionTool {
    name: String,
    parameters: serde_json::Value,
    strict: bool,
    #[serde(rename = "type")]
    r#type: FunctionType,
    description: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FunctionType {
    Function,
}

#[derive(Debug, Serialize)]
struct Format {
    format: FormatType,
}

#[derive(Debug, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
enum FormatType {
    // TODO: remove this, once text input is supported
    #[allow(dead_code)]
    Text {
        #[serde(rename = "type")]
        r#type: TextType,
    },
    JsonSchema(JsonSchema),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TextType {
    // TODO: Remove this, once text input is supported
    #[allow(dead_code)]
    Text,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,

    schema: serde_json::Value,

    #[serde(rename = "type")]
    r#type: JsonSchemaType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum JsonSchemaType {
    JsonSchema,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum InputItem {
    Message(InputMessage),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolCallOutput),
}

#[derive(Debug, Serialize, Clone)]
struct InputMessage {
    role: InputMessageRole,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiStructuredResponse {
    id: String,
    model: String,
    output: Vec<OutputContent>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OutputContent {
    OutputMessage(OutputMessage),
    FunctionCall(FunctionToolCall),
}

// TODO: Remove this, once text input is supported
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OutputMessage {
    id: String,

    #[allow(dead_code)]
    /// This is always `message`
    #[serde(rename = "type")]
    r#type: String,

    status: Status,

    content: Vec<MessageContent>,

    /// This is always `assistant`
    role: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FunctionToolCall {
    #[serde(rename = "type")]
    r#type: String,
    id: String,
    call_id: String,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FunctionToolCallOutput {
    call_id: String,
    output: serde_json::Value,
    #[serde(rename = "type")]
    r#type: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FunctionToolCallOutputType {
    FunctionCallOutput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
enum MessageContent {
    OutputText(OutputText),
    Refusal(Refusal),
}

#[derive(Debug, Deserialize)]
struct OutputText {
    #[allow(dead_code)]
    /// Always `output_text`
    #[serde(rename = "type")]
    r#type: String,

    text: String,
    // TODO
    // annotations
}

#[derive(Debug, Deserialize)]
struct Refusal {
    /// The refusal explanationfrom the model.
    refusal: String,

    #[allow(dead_code)]
    /// Always `refusal`
    #[serde(rename = "type")]
    r#type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
enum Status {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: i32,
    output_tokens: i32,
    total_tokens: i32,
}

pub fn create_openai_client_from_builder<State>(
    builder: &LlmBuilder<State>,
) -> Result<OpenAiClient, LlmError> {
    // Setting the model should be optional
    let model = builder
        .get_model()
        .ok_or_else(|| LlmError::ProviderConfiguration("Model not set".to_string()))?
        .to_string();

    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENAI_API_KEY not set.".to_string()))?
        .to_string();

    let client = OpenAiClient::new(api_key).with_default_model(model);
    Ok(client)
}

fn create_openai_structured_request<T>(
    req: StructuredRequest,
) -> Result<OpenAiStructuredRequest, LlmError>
where
    T: schemars::JsonSchema,
{
    let input = convert_messages_to_openai_format(req.messages)?;
    let text = create_format_for_type::<T>()?;

    let tools = if let Some(req_tools) = req.tools {
        let tools = req_tools
            .iter()
            .map(create_function_tool)
            .collect::<Box<[Tool]>>();
        Some(tools)
    } else {
        None
    };

    let tool_choice = req.tool_choice.map(create_function_tool_choice);

    Ok(OpenAiStructuredRequest {
        model: req.model,
        input,
        text,
        tools,
        tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
    })
}

fn create_core_structured_response<T>(
    res: OpenAiStructuredResponse,
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
                usage: core::types::LanguageModelUsage {
                    prompt_tokens: res.usage.input_tokens,
                    completion_tokens: res.usage.output_tokens,
                    total_tokens: res.usage.total_tokens,
                },
                metadata: core::types::ResponseMetadata {
                    provider: Provider::OpenAI,
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

fn create_function_tool(tool: &core::types::Tool) -> Tool {
    let strict = tool.strict.unwrap_or(true);
    let mut parameters = tool.parameters.clone();

    // OpenAI strict mode requires ALL properties to be in the required array
    if strict {
        if let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) {
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
    }

    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        parameters,
        strict,
        r#type: FunctionType::Function,
        description: tool.description.clone(),
    })
}

fn create_function_tool_choice(tool_choice: core::types::ToolChoice) -> ToolChoice {
    match tool_choice {
        core::types::ToolChoice::None => ToolChoice::Mode(ToolMode::None),
        core::types::ToolChoice::Auto => ToolChoice::Mode(ToolMode::Auto),
        core::types::ToolChoice::Required => ToolChoice::Mode(ToolMode::Required),
        core::types::ToolChoice::Function { name } => {
            ToolChoice::Definite(ToolChoiceDefinite::Function(FunctionToolChoice {
                name,
                r#type: FunctionType::Function,
            }))
        }
    }
}

fn convert_messages_to_openai_format(
    messages: Vec<ConversationMessage>,
) -> Result<Vec<InputItem>, LlmError> {
    messages
        .into_iter()
        .map(|msg| match msg {
            core::types::ConversationMessage::Chat(m) => Ok(InputItem::Message(InputMessage {
                role: match m.role {
                    core::types::ChatRole::System => InputMessageRole::System,
                    core::types::ChatRole::User => InputMessageRole::User,
                    core::types::ChatRole::Assistant => InputMessageRole::Assistant,
                },
                content: m.content,
            })),
            core::types::ConversationMessage::ToolCall(tc) => {
                Ok(InputItem::FunctionCall(FunctionToolCall {
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
                }))
            }
            core::types::ConversationMessage::ToolCallResult(tr) => {
                Ok(InputItem::FunctionCallOutput(FunctionToolCallOutput {
                    call_id: tr.tool_call_id,
                    output: serde_json::Value::String(tr.content),
                    r#type: "function_call_output".to_string(),
                }))
            }
        })
        .collect()
}

fn create_format_for_type<T>() -> Result<Format, LlmError>
where
    T: schemars::JsonSchema,
{
    let s = schema_for!(T);

    let schema_name = s
        .schema
        .metadata
        .as_ref()
        .and_then(|meta| meta.title.as_ref())
        .ok_or_else(|| LlmError::Provider {
            message: "Failed to build JSON Schema: Missing schema name".to_string(),
            source: None,
        })?
        .clone();

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
