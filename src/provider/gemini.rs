//! Google Gemini provider implementation.
//!
//! This module implements the Gemini API using the completions abstraction layer.
//! It supports text generation, structured output, and function calling.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completions::{
    CompletionClient, CompletionProviderConfig, CompletionRequestBuilder, ConversationItem,
};
use crate::core::{
    FunctionCallData, HttpClientConfig, LanguageModelUsage, LlmBuilder, LlmError, LlmProvider,
    ProviderResponse, ResponseContent, StructuredRequest, ToolCallingConfig, ToolCallingGuard,
    ToolRegistry,
};
use crate::provider::constants::gemini;
use crate::responses::{Format, request::FormatType};

// ============================================================================
// Gemini API Request Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GeminiToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    Text(TextPart),
    FunctionCall(FunctionCallPart),
    FunctionResponse(FunctionResponsePart),
}

impl Part {
    pub fn text(s: String) -> Self {
        Self::Text(TextPart { text: s })
    }

    pub fn function_call(name: String, args: Value) -> Self {
        Self::FunctionCall(FunctionCallPart {
            function_call: FunctionCall { name, args },
        })
    }

    pub fn function_response(name: String, response: Value) -> Self {
        Self::FunctionResponse(FunctionResponsePart {
            function_response: FunctionResponse { name, response },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallPart {
    pub function_call: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponsePart {
    pub function_response: FunctionResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

// ============================================================================
// Gemini API Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Option<Vec<Candidate>>,
    pub usage_metadata: Option<UsageMetadata>,
    #[allow(dead_code)]
    pub model_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<Content>,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
    #[allow(dead_code)]
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: Option<i32>,
    pub candidates_token_count: Option<i32>,
    pub total_token_count: Option<i32>,
}

// ============================================================================
// Gemini Configuration
// ============================================================================

pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub tool_calling_config: Option<ToolCallingConfig>,
    pub http_config: HttpClientConfig,
}

impl GeminiConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: gemini::API_BASE.to_string(),
            tool_calling_config: Some(ToolCallingConfig::default()),
            http_config: HttpClientConfig::default(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        self.tool_calling_config = Some(config);
        self
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Self {
        self.http_config = config;
        self
    }

    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::with_limits(config.max_iterations, config.timeout)
        } else {
            ToolCallingGuard::new()
        }
    }
}

impl CompletionProviderConfig for GeminiConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_header(&self) -> (String, String) {
        ("x-goog-api-key".to_string(), self.api_key.clone())
    }

    fn http_config(&self) -> HttpClientConfig {
        self.http_config.clone()
    }
}

// ============================================================================
// Request Builder Implementation
// ============================================================================

pub struct GeminiRequestBuilder;

impl CompletionRequestBuilder for GeminiRequestBuilder {
    type Request = GeminiRequest;
    type Response = GeminiResponse;

    fn build_request(
        &self,
        request: &StructuredRequest,
        format: &Format,
        conversation: &[ConversationItem],
    ) -> Result<Self::Request, LlmError> {
        let (system_instruction, contents) = build_contents_from_conversation(conversation)?;

        let generation_config = build_generation_config(request, format);
        let (tools, tool_config) = build_tools_config(request);

        // Gemini doesn't support combining function calling with structured JSON output
        if tools.is_some() && matches!(format.format, FormatType::JsonSchema(_)) {
            return Err(LlmError::ProviderConfiguration(
                "Gemini does not support combining function calling with structured JSON output. \
                 Use TextResponse with tools, or structured output without tools."
                    .to_string(),
            ));
        }

        Ok(GeminiRequest {
            contents,
            generation_config,
            system_instruction,
            tools,
            tool_config,
        })
    }

    fn parse_response(&self, response: Self::Response) -> Result<ProviderResponse, LlmError> {
        let candidate = response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .ok_or_else(|| LlmError::Provider {
                message: "No candidates in Gemini response".to_string(),
                source: None,
            })?;

        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| LlmError::Provider {
                message: "No content in Gemini candidate".to_string(),
                source: None,
            })?;

        let response_content = parse_parts_to_content(&content.parts)?;

        let usage = response
            .usage_metadata
            .map(|u| LanguageModelUsage {
                prompt_tokens: u.prompt_token_count.unwrap_or(0),
                completion_tokens: u.candidates_token_count.unwrap_or(0),
                total_tokens: u.total_token_count.unwrap_or(0),
            })
            .unwrap_or(LanguageModelUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });

        Ok(ProviderResponse {
            id: String::new(), // Gemini doesn't return an ID
            model: response.model_version.unwrap_or_default(),
            provider: super::Provider::Gemini,
            content: response_content,
            usage,
        })
    }

    fn endpoint(&self, model: &str) -> String {
        format!("/models/{}:generateContent", model)
    }

    fn extract_function_calls(&self, response: &Self::Response) -> Option<Vec<FunctionCallData>> {
        let candidate = response.candidates.as_ref()?.first()?;
        let content = candidate.content.as_ref()?;

        let mut calls = Vec::new();
        for (idx, part) in content.parts.iter().enumerate() {
            if let Part::FunctionCall(FunctionCallPart { function_call }) = part {
                calls.push(FunctionCallData {
                    id: format!("call_{}", idx),
                    name: function_call.name.clone(),
                    arguments: function_call.args.clone(),
                });
            }
        }

        if calls.is_empty() { None } else { Some(calls) }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_contents_from_conversation(
    conversation: &[ConversationItem],
) -> Result<(Option<Content>, Vec<Content>), LlmError> {
    let mut system_instruction: Option<Content> = None;
    let mut contents: Vec<Content> = Vec::new();

    for item in conversation {
        match item {
            ConversationItem::Message { role, content } => {
                if role == "system" {
                    system_instruction = Some(Content {
                        role: None,
                        parts: vec![Part::text(content.clone())],
                    });
                } else {
                    let gemini_role = match role.as_str() {
                        "user" => "user",
                        "assistant" => "model",
                        _ => "user",
                    };
                    contents.push(Content {
                        role: Some(gemini_role.to_string()),
                        parts: vec![Part::text(content.clone())],
                    });
                }
            }
            ConversationItem::FunctionCall {
                name, arguments, ..
            } => {
                contents.push(Content {
                    role: Some("model".to_string()),
                    parts: vec![Part::function_call(name.clone(), arguments.clone())],
                });
            }
            ConversationItem::FunctionResult { call_id, result } => {
                // For Gemini, we need to find the function name from previous calls
                let name = find_function_name_by_call_id(conversation, call_id)
                    .unwrap_or_else(|| call_id.clone());

                // Gemini requires function_response.response to be a Struct (object).
                // Wrap non-object values in a result wrapper.
                let response_value = match result {
                    Value::Object(_) => result.clone(),
                    other => serde_json::json!({ "result": other }),
                };

                contents.push(Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::function_response(name, response_value)],
                });
            }
        }
    }

    Ok((system_instruction, contents))
}

fn find_function_name_by_call_id(
    conversation: &[ConversationItem],
    call_id: &str,
) -> Option<String> {
    for item in conversation {
        if let ConversationItem::FunctionCall { id, name, .. } = item
            && id == call_id
        {
            return Some(name.clone());
        }
    }
    None
}

/// Convert standard JSON Schema to Gemini's schema format.
///
/// Gemini uses a simplified schema with uppercase type names and fewer fields.
/// If multiple providers adopt this format in the future, this should become
/// the default schema format, with OpenAI doing its own conversion.
fn convert_to_gemini_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(obj) => {
            let mut result = serde_json::Map::new();

            for (key, value) in obj {
                match key.as_str() {
                    // Skip unsupported fields
                    "$schema" | "additionalProperties" | "title" => continue,
                    // Convert type to uppercase
                    "type" => {
                        if let Value::String(t) = value {
                            result.insert("type".to_string(), Value::String(t.to_uppercase()));
                        }
                    }
                    // Recurse into nested schemas
                    "properties" => {
                        if let Value::Object(props) = value {
                            let converted: serde_json::Map<String, Value> = props
                                .iter()
                                .map(|(k, v)| (k.clone(), convert_to_gemini_schema(v)))
                                .collect();
                            result.insert("properties".to_string(), Value::Object(converted));
                        }
                    }
                    "items" => {
                        result.insert("items".to_string(), convert_to_gemini_schema(value));
                    }
                    // Pass through supported fields
                    "required" | "enum" | "description" => {
                        result.insert(key.clone(), value.clone());
                    }
                    _ => {}
                }
            }

            Value::Object(result)
        }
        other => other.clone(),
    }
}

fn build_generation_config(
    request: &StructuredRequest,
    format: &Format,
) -> Option<GeminiGenerationConfig> {
    let gen_config = request.generation_config.as_ref();

    let (response_mime_type, response_schema) = match &format.format {
        FormatType::JsonSchema(json_schema) => (
            Some("application/json".to_string()),
            Some(convert_to_gemini_schema(&json_schema.schema)),
        ),
        FormatType::Text { .. } => (None, None),
    };

    // Only create config if there's something to configure
    if gen_config.is_none() && response_mime_type.is_none() {
        return None;
    }

    Some(GeminiGenerationConfig {
        temperature: gen_config.and_then(|c| c.temperature),
        top_p: gen_config.and_then(|c| c.top_p),
        top_k: None,
        max_output_tokens: gen_config.and_then(|c| c.max_tokens),
        response_mime_type,
        response_schema,
    })
}

fn build_tools_config(
    request: &StructuredRequest,
) -> (Option<Vec<GeminiTool>>, Option<GeminiToolConfig>) {
    let Some(tool_config) = request.tool_config.as_ref() else {
        return (None, None);
    };
    let Some(tools) = tool_config.tools.as_ref() else {
        return (None, None);
    };

    if tools.is_empty() {
        return (None, None);
    }

    let function_declarations: Vec<GeminiFunctionDeclaration> = tools
        .iter()
        .map(|t| GeminiFunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: convert_to_gemini_schema(&t.parameters),
        })
        .collect();

    let gemini_tools = vec![GeminiTool {
        function_declarations,
    }];

    let mode = match &tool_config.tool_choice {
        Some(crate::core::ToolChoice::None) => "NONE",
        Some(crate::core::ToolChoice::Auto) => "AUTO",
        Some(crate::core::ToolChoice::Required) => "ANY",
        Some(crate::core::ToolChoice::Function { name }) => {
            // For specific function, we use ANY with allowed_function_names
            return (
                Some(gemini_tools),
                Some(GeminiToolConfig {
                    function_calling_config: FunctionCallingConfig {
                        mode: "ANY".to_string(),
                        allowed_function_names: Some(vec![name.clone()]),
                    },
                }),
            );
        }
        None => "AUTO",
    };

    (
        Some(gemini_tools),
        Some(GeminiToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: mode.to_string(),
                allowed_function_names: None,
            },
        }),
    )
}

fn parse_parts_to_content(parts: &[Part]) -> Result<ResponseContent, LlmError> {
    let mut text_parts = Vec::new();
    let mut function_calls = Vec::new();

    for (idx, part) in parts.iter().enumerate() {
        match part {
            Part::Text(TextPart { text }) => text_parts.push(text.clone()),
            Part::FunctionCall(FunctionCallPart { function_call }) => {
                function_calls.push(FunctionCallData {
                    id: format!("call_{}", idx),
                    name: function_call.name.clone(),
                    arguments: function_call.args.clone(),
                });
            }
            Part::FunctionResponse(_) => {}
        }
    }

    if !function_calls.is_empty() {
        Ok(ResponseContent::FunctionCalls(function_calls))
    } else if !text_parts.is_empty() {
        Ok(ResponseContent::Text(text_parts.join("")))
    } else {
        Err(LlmError::Provider {
            message: "Empty response from Gemini".to_string(),
            source: None,
        })
    }
}

// ============================================================================
// Gemini Client
// ============================================================================

pub struct GeminiClient {
    completion_client: CompletionClient<GeminiConfig>,
    config: GeminiConfig,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = GeminiConfig::new(api_key.clone());
        let completion_client = CompletionClient::new(GeminiConfig::new(api_key))?;

        Ok(Self {
            completion_client,
            config,
        })
    }

    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        let new_config = GeminiConfig {
            api_key: self.config.api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: self.config.tool_calling_config.clone(),
            http_config: self.config.http_config.clone(),
        };
        self.config = GeminiConfig {
            api_key: self.config.api_key.clone(),
            base_url,
            tool_calling_config: self.config.tool_calling_config.clone(),
            http_config: self.config.http_config.clone(),
        };
        self.completion_client = CompletionClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_tool_calling_config(
        mut self,
        tool_config: ToolCallingConfig,
    ) -> Result<Self, LlmError> {
        let new_config = GeminiConfig {
            api_key: self.config.api_key.clone(),
            base_url: self.config.base_url.clone(),
            tool_calling_config: Some(tool_config.clone()),
            http_config: self.config.http_config.clone(),
        };
        self.config.tool_calling_config = Some(tool_config);
        self.completion_client = CompletionClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_http_config(mut self, http_config: HttpClientConfig) -> Result<Self, LlmError> {
        let new_config = GeminiConfig {
            api_key: self.config.api_key.clone(),
            base_url: self.config.base_url.clone(),
            tool_calling_config: self.config.tool_calling_config.clone(),
            http_config: http_config.clone(),
        };
        self.config.http_config = http_config;
        self.completion_client = CompletionClient::new(new_config)?;
        Ok(self)
    }
}

#[async_trait]
impl LlmProvider for GeminiClient {
    async fn generate_completion<T, Ctx>(
        &self,
        request: StructuredRequest,
        format: Format,
        tool_registry: Option<&ToolRegistry<Ctx>>,
    ) -> Result<T::Output, LlmError>
    where
        T: crate::CompletionTarget + Send,
        Ctx: Send + Sync + 'static,
    {
        let builder = GeminiRequestBuilder;

        // If tools are present and we have a registry, handle automatic tool calling
        let has_tools = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.tools.as_ref())
            .is_some();

        if has_tools && tool_registry.is_some() {
            let mut guard = self.config.get_tool_calling_guard();
            let provider_response = self
                .completion_client
                .handle_tool_calling_loop::<_, Ctx>(
                    &builder,
                    request,
                    tool_registry.unwrap(),
                    &mut guard,
                    format,
                )
                .await?;
            return T::parse_response(provider_response);
        }

        // Single request without tool calling loop
        let conversation = convert_messages_to_conversation(&request.messages)?;
        let api_request = builder.build_request(&request, &format, &conversation)?;
        let api_response = self
            .completion_client
            .make_api_request(&builder, api_request, &request.model)
            .await?;
        let provider_response = builder.parse_response(api_response)?;
        T::parse_response(provider_response)
    }
}

fn convert_messages_to_conversation(
    messages: &[crate::core::ConversationMessage],
) -> Result<Vec<ConversationItem>, LlmError> {
    messages
        .iter()
        .map(|msg| match msg {
            crate::core::ConversationMessage::Chat(m) => {
                let role = match m.role {
                    crate::core::ChatRole::System => "system",
                    crate::core::ChatRole::User => "user",
                    crate::core::ChatRole::Assistant => "assistant",
                };
                Ok(ConversationItem::Message {
                    role: role.to_string(),
                    content: m.content.clone(),
                })
            }
            crate::core::ConversationMessage::ToolCall(tc) => Ok(ConversationItem::FunctionCall {
                id: tc.call_id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            }),
            crate::core::ConversationMessage::ToolCallResult(tr) => {
                Ok(ConversationItem::FunctionResult {
                    call_id: tr.tool_call_id.clone(),
                    result: tr.content.clone(),
                })
            }
        })
        .collect()
}

// ============================================================================
// Builder Integration
// ============================================================================

pub fn create_gemini_client_from_builder<State, Ctx>(
    builder: &LlmBuilder<State, Ctx>,
) -> Result<GeminiClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("GEMINI_API_KEY not set.".to_string()))?
        .to_string();

    let mut client = GeminiClient::new(api_key)?;

    if let Some(http_config) = builder.get_http_config() {
        client = client.with_http_config(http_config.clone())?;
    }

    Ok(client)
}
