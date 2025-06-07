use crate::core::{
    self,
    types::{LlmResponse, StructuredRequest, StructuredResponse, ToolCall},
};
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
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_model: "gpt-4.1".to_string(),
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
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn generate_structured<T>(
        &self,
        request: StructuredRequest,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        // TODO: Fix error handling.

        let request = create_openai_structured_request::<T>(request)?;

        let url = format!("{}/responses", self.base_url);
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
                message: format!("OpenAI API returned error: {}", error_text),
                status_code: Some(status.as_u16()),
                source: None,
            });
        }

        let api_res: OpenAiStructuredResponse = res.json().await.map_err(|e| LlmError::Parse {
            message: "Failed to parse OpenAI response".to_string(),
            source: Box::new(e),
        })?;

        create_core_structured_response(api_res)
    }

    async fn generate<T>(
        &self,
        request: StructuredRequest,
    ) -> Result<LlmResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        let request = create_openai_structured_request::<T>(request)?;

        let url = format!("{}/responses", self.base_url);
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
                message: format!("OpenAI API returned error: {}", error_text),
                status_code: Some(status.as_u16()),
                source: None,
            });
        }

        let api_res: OpenAiStructuredResponse = res.json().await.map_err(|e| LlmError::Parse {
            message: "Failed to parse OpenAI response".to_string(),
            source: Box::new(e),
        })?;

        create_core_llm_response(api_res)
    }
}

#[derive(Debug, Serialize)]
struct OpenAiStructuredRequest {
    model: String,
    input: Vec<InputMessage>,
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
    Hosted(HostedToolChoice),
    Function(FunctionToolChoice),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum HostedToolChoice {
    // FileSearch,
    // WebSearchPreview,
    // ComputerUsePreview,
    // CodeInterpreter,
    // Mcp,
    // ImageGeneration,
}

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
#[serde(untagged)]
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
    Text {
        #[serde(rename = "type")]
        r#type: TextType,
    },
    JsonSchema(JsonSchema),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TextType {
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
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
    OutputMessage(Message),
    FunctionCall(FunctionCall),
}

#[derive(Debug, Deserialize)]
struct Message {
    id: String,

    /// This is always `message`
    #[serde(rename = "type")]
    r#type: String,

    status: MessageStatus,

    content: Vec<MessageContent>,

    /// This is always `assistant`
    role: String,
}

#[derive(Debug, Deserialize)]
struct FunctionCall {
    #[serde(rename = "type")]
    r#type: String,
    id: String,
    call_id: String,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
enum MessageContent {
    OutputText(OutputText),
    Refusal(Refusal),
}

#[derive(Debug, Deserialize)]
struct OutputText {
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

    /// Always `refusal`
    #[serde(rename = "type")]
    r#type: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MessageStatus {
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
    let input = req
        .messages
        .into_iter()
        .map(|m| InputMessage {
            role: match m.role {
                core::types::ChatRole::System => InputMessageRole::System,
                core::types::ChatRole::User => InputMessageRole::User,
                core::types::ChatRole::Assistant => InputMessageRole::Assistant,
            },
            content: m.content,
        })
        .collect();

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

    let schema = JsonSchema {
        name: schema_name,
        schema: schema_value,
        r#type: JsonSchemaType::JsonSchema,
    };

    let tools = if let Some(req_tools) = req.tools {
        let tools = req_tools
            .iter()
            .map(create_function_tool)
            .collect::<Box<[Tool]>>();
        Some(tools)
    } else {
        None
    };

    let tool_choice = if let Some(req_tool_choice) = req.tool_choice {
        Some(create_function_tool_choice(req_tool_choice))
    } else {
        None
    };

    Ok(OpenAiStructuredRequest {
        model: req.model,
        input,
        text: Format {
            format: FormatType::JsonSchema(schema),
        },
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
            let parsed_content: T = if let Ok(wrapped) = serde_json::from_str::<ValueWrapper<T>>(&text) {
                wrapped.value
            } else {
                serde_json::from_str(&text).map_err(|e| LlmError::Parse {
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
        OutputContent::FunctionCall(_) => {
            Err(LlmError::Provider {
                message: "Function call response received when expecting structured output".to_string(),
                source: None,
            })
        }
    }
}

fn create_core_llm_response<T>(
    res: OpenAiStructuredResponse,
) -> Result<LlmResponse<T>, LlmError>
where
    T: serde::de::DeserializeOwned,
{
    // Check if response contains function calls
    let function_calls: Vec<&FunctionCall> = res.output.iter()
        .filter_map(|output| match output {
            OutputContent::FunctionCall(fc) => Some(fc),
            OutputContent::OutputMessage(_) => None,
        })
        .collect();

    if !function_calls.is_empty() {
        let tool_calls = function_calls.into_iter()
            .map(|fc| ToolCall {
                id: fc.id.clone(),
                name: fc.name.clone(),
                arguments: fc.arguments.clone(),
            })
            .collect();
        
        return Ok(LlmResponse::ToolCalls(tool_calls));
    }

    // Otherwise, handle as structured content
    let structured_response = create_core_structured_response(res)?;
    Ok(LlmResponse::Content(structured_response))
}

fn create_function_tool(tool: &core::types::Tool) -> Tool {
    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        parameters: tool.parameters.clone(),
        strict: tool.strict.unwrap_or(true),
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
