use async_trait::async_trait;
use futures::Stream;
use reqwest::{Client, header};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;

use crate::AIError;
use crate::model::chat::{self, FinishReason, LanguageModelUsage, ProviderOptions, Temperature};
use crate::model::chat::{StructuredOutput, StructuredOutputParameters};
use crate::model::{ChatRole, ChatSettings, TextCompletion, TextStream};

use super::Provider;

const DEFAULT_MODEL: &str = "gpt-4.1-mini-2025-04-14";
const API_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI-specific provider settings
#[derive(Debug, Clone, Default)]
pub struct OpenAISettings {
    /// Override the model specified in the provider
    pub model: Option<String>,

    /// Number between -2.0 and 2.0. Positive values penalize tokens that appear
    /// frequently in the text so far, decreasing their likelihood.
    pub frequency_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize tokens that appear
    /// in the text so far, increasing the likelihood of new topics.
    pub presence_penalty: Option<f32>,

    /// If specified, OpenAI will make a best effort to sample deterministically.
    pub seed: Option<i64>,

    pub temperature: Option<f32>,

    /// A unique identifier representing the end-user, to help OpenAI monitor and detect abuse.
    pub user: Option<String>,
}

impl OpenAISettings {
    /// Create a new instance with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set frequency penalty
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Set presence penalty
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty.clamp(-2.0, 2.0));
        self
    }

    /// Set random seed for deterministic generation
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set user identifier
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Convert to a Box<dyn ProviderOptions> for use with ChatSettings
    pub fn into_provider_options(self) -> Box<dyn ProviderOptions> {
        Box::new(self)
    }
}

impl ProviderOptions for OpenAISettings {
    fn clone_box(&self) -> Box<dyn ProviderOptions> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct OpenAIProvider {
    api_key: String,
    model: String,
    base_url: String,
    client: Client,
}

// OpenAI API request and response types
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<Audio>,

    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,

    // logit_bias
    // logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<Modality>>,

    /// Default is 1
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    prediction: Option<StaticContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,

    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    // tool_choice
    // tools
    // top_logprobs
    // top_p
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_options: Option<WebSearchOptions>,
}

#[derive(Debug, Serialize)]
struct StaticContent {
    content: Content,

    #[serde(rename = "type")]
    static_content_type: ContentType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum ContentType {
    Content,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Content {
    TextContent(String),
    ArrayOfContent(Vec<String>),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    Low,
    Medium,
    Hight,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum ServiceTier {
    Default,
    Auto,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    /// If set, an additional chunk will be streamed before the data: \[DONE\] message.
    /// The usage field on this chunk shows the token usage statistics for the entire request,
    /// and the choices field will always be an empty array.
    #[serde(skip_serializing_if = "Option::is_none")]
    include_usage: Option<bool>,
}

#[derive(Debug, Serialize)]
struct WebSearchOptions {
    /// Reusing `ResoningEffort` here, since the options are also `low`,`medium` and `high`.
    search_context_size: Option<ReasoningEffort>,

    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<UserLocation>,
}

#[derive(Debug, Serialize)]
struct UserLocation {
    approximate: ApproximateLocation,

    location_type: LocationType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum LocationType {
    Approximate,
}

#[derive(Debug, Serialize)]
struct ApproximateLocation {
    city: Option<String>,

    /// Two-letter country code.
    country: Option<String>,

    region: Option<String>,

    /// IANA timezone.
    timezone: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Audio {
    format: AudioFormat,
    voice: Voice,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    WAV,
    MP3,
    FLAC,
    OPUS,
    PCM16,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    ALLOY,
    ASH,
    BALLAD,
    CORAL,
    ECHO,
    SAGE,
    SHIMMER,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    choices: Vec<ChatCompletionChoice>,
    created: i32,
    model: String,
    object: CompletionType,
    service_tier: Option<String>,
    system_fingerprint: String,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
enum CompletionType {
    #[serde(rename = "chat.completion")]
    ChatCompletion,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    TEXT,
    AUDIO,
}

/// Represents a streamed chunk of a chat completion response.
#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    id: String,
    choices: Vec<ChatCompletionChunkChoice>,
    created: i32,
    model: String,
    object: CompletionType,
    service_tier: Option<String>,
    system_fingerprint: String,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
enum CompletionChunkType {
    #[serde(rename = "chat.completion.chunk")]
    ChatCompletionChunk,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunkChoice {
    delta: CompletionDelta,
    finish_reason: Option<String>,
    index: i32,
    // logprobs
}

#[derive(Debug, Deserialize)]
struct CompletionDelta {
    content: Option<String>,
    refusal: Option<String>,
    role: String,
    // tool_calls
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    finish_reason: Option<String>,
    index: i32,
    // logprobs
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
    refusal: Option<String>,
    role: String,
    annotations: Option<Vec<Annotation>>,

    audio: Option<AudioResponse>,
    // tool_calls
}

#[derive(Debug, Deserialize)]
struct AudioResponse {
    id: String,

    /// Base64 encoded audio bytes
    data: String,

    /// The Unix timestamp (in seconds) for when this audio response will no longer be
    /// accessible on the server for use in multi-turn conversations.
    expires_at: i32,

    /// Transcript of the audio generated by the model.
    transcript: String,
}

#[derive(Debug, Deserialize)]
struct Annotation {
    #[serde(rename = "type")]
    annotation_type: AnnotationType,
    url_citation: UrlCitation,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnnotationType {
    UrlCitation,
}

#[derive(Debug, Deserialize)]
struct UrlCitation {
    end_index: i32,
    start_index: i32,
    title: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    completion_tokens: i32,
    prompt_tokens: i32,
    total_tokens: i32,
    completion_tokens_details: CompletionTokensDetails,
    prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    accepted_prediction_tokens: i32,
    audio_tokens: i32,
    reasoning_tokens: i32,
    rejected_prediction_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    audio_tokens: i32,
    cached_tokens: i32,
}

impl OpenAIProvider {
    /// Creates a new `OpenAIProvider` with the specified API key and model
    pub fn new(api_key: &str, model: &str) -> Self {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .expect("Invalid API key format"),
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        OpenAIProvider {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: API_BASE_URL.to_string(),
            client: Client::builder()
                .default_headers(headers)
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Creates an `OpenAIProvider` with the default model (`gpt-4o`)
    pub fn default(api_key: &str) -> Self {
        Self::new(api_key, DEFAULT_MODEL)
    }

    /// Sets a custom base URL for the API
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    /// Constructs the chat completions endpoint URL
    fn get_chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

// Helper functions for OpenAI provider

/// Builds the request payload for the OpenAI API
fn build_request(
    prompt: &str,
    settings: &ChatSettings,
    default_model: &str,
) -> Result<ChatCompletionRequest, AIError> {
    let mut messages = Vec::new();

    // Add system prompt if available
    if let Some(system_prompt) = &settings.system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system_prompt.clone(),
        });
    }

    // Add conversation history if available
    if let Some(history) = &settings.messages {
        for msg in history {
            let role = match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::System => "system",
            };

            messages.push(ChatMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            });
        }
    }

    // Add the current prompt as a user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    });

    // Extract OpenAI-specific settings if available
    let provider_settings = settings
        .provider_options
        .as_ref()
        .and_then(|options| options.as_any().downcast_ref::<OpenAISettings>());

    // Use provider-specified model or fallback to the default
    let model = provider_settings
        .and_then(|s| s.model.clone())
        .unwrap_or_else(|| default_model.to_string());

    // Map temperature from ChatSettings
    let temperature = settings.temperature.as_ref().map(|temp| match temp {
        Temperature::Raw(val) => *val,
        Temperature::Normalized(val) => val.clamp(0.0, 1.0) * 2.0, // Scale to OpenAI's 0-2 range
    });

    // Create request
    let request = ChatCompletionRequest {
        model,
        messages,
        frequency_penalty: provider_settings.and_then(|s| s.frequency_penalty),
        max_completion_tokens: settings.max_tokens.map(|t| t as i64),
        parallel_tool_calls: None,
        presence_penalty: provider_settings.and_then(|s| s.presence_penalty),
        response_format: None,
        seed: provider_settings.and_then(|s| s.seed),
        stop: settings.stop_sequences.clone(),
        store: None,
        stream: None,
        temperature,
        user: provider_settings.and_then(|s| s.user.clone()),
        audio: todo!(),
        metadata: todo!(),
        modalities: todo!(),
        n: todo!(),
        prediction: todo!(),
        reasoning_effort: todo!(),
        service_tier: todo!(),
        stream_options: todo!(),
        web_search_options: todo!(),
    };

    Ok(request)
}

/// Maps OpenAI finish reason to our FinishReason enum
fn map_finish_reason(reason: String) -> FinishReason {
    match reason.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => {
            FinishReason::ContentFilter("OpenAI content filter triggered".to_string())
        }
        "tool_calls" => FinishReason::ToolCalls,
        "function_call" => FinishReason::ToolCalls, // Legacy, map to tool calls
        _ => FinishReason::Other,
    }
}

/// Maps OpenAI usage information to our LanguageModelUsage type
fn map_usage(usage: &Usage) -> LanguageModelUsage {
    LanguageModelUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

/// Parses an SSE chunk into a TextStream
fn parse_sse_chunk(bytes: bytes::Bytes) -> Result<TextStream, AIError> {
    use std::str::from_utf8;

    let chunk_str = from_utf8(&bytes)
        .map_err(|e| AIError::ConversionError(format!("Invalid UTF-8 in SSE chunk: {}", e)))?
        .trim();

    if chunk_str.is_empty() {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Unknown,
            usage: None,
        });
    }

    // SSE format: each message is prefixed with "data: "
    let Some(data) = chunk_str.strip_prefix("data: ") else {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Other,
            usage: None,
        });
    };

    let data = data.trim();

    // Check for the end of stream marker
    if data == "[DONE]" {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Stop,
            usage: None,
        });
    }

    // Parse the JSON data
    let chunk: ChatCompletionChunk = serde_json::from_str(data)
        .map_err(|e| AIError::ConversionError(format!("Invalid JSON in SSE chunk: {}", e)))?;

    if chunk.choices.is_empty() {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Other,
            usage: None,
        });
    }

    // Extract the content delta
    let choice = &chunk.choices[0];
    let text = choice.delta.content.clone().unwrap_or_default();

    // Map finish reason if provided
    let finish_reason = choice
        .finish_reason
        .clone()
        .map_or(chat::FinishReason::Unknown, map_finish_reason);

    // If available, map usage info
    let usage = chunk.usage.as_ref().map(map_usage);

    Ok(TextStream {
        text,
        finish_reason,
        usage,
    })
}
type PinnedStream<'a> = Pin<Box<dyn Stream<Item = Result<TextStream, AIError>> + 'a>>;

#[async_trait]
impl Provider for OpenAIProvider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError> {
        let url = self.get_chat_completions_url();
        let request = build_request(prompt, settings, &self.model)?;

        let response = self
            .client
            .post(&url)
            .headers(settings.headers.clone())
            .json(&request)
            .send()
            .await
            .map_err(|e| AIError::RequestError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AIError::ApiError(format!(
                "API returned error status {}: {}",
                status, error_text
            )));
        }

        let response_body = response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| AIError::ConversionError(format!("Failed to parse response: {}", e)))?;

        let choice = response_body
            .choices
            .first()
            .ok_or_else(|| AIError::ApiError("No response choices returned".to_string()))?;

        // Extract content, handling refusal
        let text = match (&choice.message.content, &choice.message.refusal) {
            (Some(content), _) => content.clone(),
            (None, Some(refusal)) => {
                return Err(AIError::ContentFilterError(refusal.clone()));
            }
            (None, None) => {
                return Err(AIError::ApiError(
                    "Response missing both content and refusal".to_string(),
                ));
            }
        };

        let finish_reason = choice
            .finish_reason
            .clone()
            .map_or(chat::FinishReason::Unknown, map_finish_reason);

        let usage = map_usage(&response_body.usage);

        Ok(TextCompletion {
            text,
            finish_reason,
            usage,
        })
    }

    async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
        settings: &'a ChatSettings,
    ) -> Result<PinnedStream, AIError> {
        todo!("To implement");
    }

    async fn generate_object<T: DeserializeOwned + JsonSchema + Sync>(
        &self,
        prompt: &str,
        settings: &ChatSettings,
        parameters: &StructuredOutputParameters<T>,
    ) -> Result<StructuredOutput<T>, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality(
            "Not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn test_chat_completion_request_full_json() {
        let request = ChatCompletionRequest {
            model: "gpt-4.1-mini-2025-04-14".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello, world!".to_string(),
            }],
            audio: None,
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(128),
            metadata: None,
            modalities: None,
            n: None,
            parallel_tool_calls: None,
            prediction: None,
            presence_penalty: None,
            reasoning_effort: None,
            response_format: None,
            seed: None,
            service_tier: None,
            stop: None,
            store: None,
            stream: None,
            stream_options: None,
            temperature: Some(1.0),
            user: Some("test-user".to_string()),
            web_search_options: None,
        };

        let actual_json = serde_json::to_value(&request).unwrap();

        let expected_json = json!({
            "model": "gpt-4.1-mini-2025-04-14",
            "messages": [
                { "role": "user", "content": "Hello, world!" }
            ],
            "frequency_penalty": 0.5,
            "max_completion_tokens": 128,
            "temperature": 1.0,
            "user": "test-user"
        });

        assert_eq!(actual_json, expected_json);
    }
}
