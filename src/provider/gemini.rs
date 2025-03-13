// Remove this in the future
#![allow(dead_code)]

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashMap;
use std::str::from_utf8;

use crate::AIError;
use crate::model::chat;
use crate::model::chat::FinishReason as ChatFinishReason;
use crate::model::chat::LanguageModelUsage;
use crate::model::{ChatMessage, ChatRole, ChatSettings, TextCompletion, TextStream};

use super::Provider;

const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models/";

pub struct GeminiProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl GeminiProvider {
    /// Creates a new `GeminiProvider` with the specified API key and model
    pub fn new(api_key: &str, model: &str) -> Self {
        GeminiProvider {
            api_key: api_key.to_string(),
            client: Client::new(),
            model: model.to_string(),
        }
    }

    /// Creates a `GeminiProvider` with the default model (`gemini-2.0-flash`)/
    pub fn default(api_key: &str) -> Self {
        Self::new(api_key, DEFAULT_MODEL)
    }

    /// Constructs the model URL for the Gemini API
    fn get_model_url(&self) -> String {
        format!("{}{}", API_BASE_URL, self.model)
    }
}

/// Builds the request payload for the Gemini API
fn build_request(prompt: &str, settings: &ChatSettings) -> Result<Request, AIError> {
    let mut contents = Vec::new();

    if let Some(messages) = &settings.messages {
        for msg in messages {
            let content = Content::try_from(msg.clone())?;
            contents.push(content);
        }
    }

    contents.push(Content {
        role: Role::User,
        parts: vec![Part {
            text: prompt.to_string(),
        }],
    });

    let system_instruction = settings
        .system_prompt
        .as_ref()
        .map(|prompt| SystemInstruction {
            parts: vec![Part {
                text: prompt.to_string(),
            }],
        });

    let generation_config = Option::<GenerationConfig>::from(settings.clone());
    let provider_settings = Option::<GeminiSettings>::from(settings.clone());
    let safety_settings = provider_settings.and_then(|s| s.safety_settings.clone());

    Ok(Request {
        contents,
        system_instruction,
        generation_config,
        safety_settings,
    })
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError> {
        let url = format!("{}:generateContent", self.get_model_url());
        let request = build_request(prompt, settings)?;

        let response = self
            .client
            .post(&url)
            .headers(settings.headers.clone())
            .query(&[("key", &self.api_key)])
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

        let parsed_response = response
            .json::<GenerateContentResponse>()
            .await
            .map_err(|e| AIError::ConversionError(format!("Failed to Content response: {}", e)))?;

        let candidate = parsed_response
            .candidates
            .first()
            .ok_or_else(|| AIError::ApiError("No response candidates returned".to_string()))?;

        let text = candidate
            .content
            .parts
            .first()
            .ok_or_else(|| AIError::ApiError("No content parts in response".to_string()))?
            .text
            .clone();

        let finish_reason = candidate
            .finish_reason
            .clone()
            .map_or(ChatFinishReason::Unknown, ChatFinishReason::from);

        let usage = LanguageModelUsage::try_from(parsed_response.usage_metadata)?;

        Ok(TextCompletion {
            text: text.to_string(),
            finish_reason,
            usage,
        })
    }

    async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
        settings: &'a ChatSettings,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>> + 'a, AIError> {
        let url = format!("{}:streamGenerateContent", self.get_model_url());
        let request = build_request(prompt, settings)?;

        let response = self
            .client
            .post(url)
            .headers(settings.headers.clone())
            .query(&[("key", &self.api_key), ("alt", &"sse".to_string())])
            .json(&request)
            .send()
            .await
            .map_err(|e| AIError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(AIError::ApiError(format!(
                "error status {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let bytes_stream = response.bytes_stream();
        Ok(bytes_stream.map(|result| match result {
            Ok(bytes) => match parse_sse_chunk(bytes) {
                Ok(text_stream) => Ok(text_stream),
                Err(e) => Err(AIError::ConversionError(format!(
                    "Failed to parse SSE: {}",
                    e
                ))),
            },
            Err(e) => Err(AIError::RequestError(format!("Stream error: {}", e))),
        }))
    }
}

/// Parses an SSE chunk into a `TextStream`
fn parse_sse_chunk(bytes: Bytes) -> Result<TextStream, AIError> {
    let chunk_str = from_utf8(&bytes)
        .map_err(|e| AIError::ConversionError(format!("Invalid UTF-8 in SSE chunk: {}", e)))?
        .trim();

    if chunk_str.is_empty() {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: ChatFinishReason::Unknown,
            usage: None,
        });
    }

    let Some(data) = chunk_str.strip_prefix("data: ") else {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: ChatFinishReason::Other,
            usage: None,
        });
    };

    let data = data.trim();
    if data == "[DONE]" {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Stop,
            usage: None,
        });
    }

    let chunk = serde_json::from_str::<GenerateContentResponse>(data)
        .map_err(|e| AIError::ConversionError(format!("Invalid JSON: {}", e)))?;

    if chunk.candidates.is_empty() || chunk.candidates[0].content.parts.is_empty() {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: ChatFinishReason::Other,
            usage: None,
        });
    }

    let text = chunk.candidates[0].content.parts[0].text.clone();
    let finish_reason = chunk.candidates[0]
        .finish_reason
        .clone()
        .map_or(ChatFinishReason::Unknown, ChatFinishReason::from);
    let usage = LanguageModelUsage::try_from(chunk.usage_metadata).ok();

    Ok(TextStream {
        text,
        finish_reason,
        usage,
    })
}

impl From<FinishReason> for ChatFinishReason {
    fn from(finish_reason: FinishReason) -> Self {
        match finish_reason {
            FinishReason::Unspecified => ChatFinishReason::Unknown,
            FinishReason::Stop => ChatFinishReason::Stop,
            FinishReason::MaxTokens => ChatFinishReason::Length,
            FinishReason::Safety => ChatFinishReason::ContentFilter("The response candidate content was flagged for safety reasons.".to_string()),
            FinishReason::Recitation => ChatFinishReason::ContentFilter("The response candidate content was flagged for recitation reasons.".to_string()),
            FinishReason::Language => ChatFinishReason::Other,
            FinishReason::Other => ChatFinishReason::Other,
            FinishReason::Blocklist => ChatFinishReason::ContentFilter("Token generation stopped because the content contains forbidden terms.".to_string()),
            FinishReason::ProhibitedContent => ChatFinishReason::ContentFilter("Token generation stopped for potentially containing prohibited content.".to_string()),
            FinishReason::Spii => ChatFinishReason::ContentFilter("Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).".to_string()),
            FinishReason::MalformedFunctionCall => ChatFinishReason::Error,
            FinishReason::ImageSafety => ChatFinishReason::ContentFilter("Token generation stopped because generated images contain safety violations.".to_string()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    User,
    Model,
}

/// !INCOMPLETE!
/// Containing media that is part of a multi-part Content message
///
///[Google API Docs](https://ai.google.dev/api/caching#Part)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Part {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Content {
    parts: Vec<Part>,
    role: Role,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<Vec<SafetySetting>>,
}

// Response structs for parsing Gemini API responses
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    candidates: Vec<Candidate>,
    #[serde(default)]
    prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    model_version: Option<String>,
}

/// !INCOMPLETE!
/// A response candidate generated from the model.
///
///[Google API Docs](https://ai.google.dev/api/generate-content#v1beta.Candidate)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Content,
    /// If this is `None` the model hasn't stopped generating tokens yet.
    finish_reason: Option<FinishReason>,
    token_count: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    #[serde(default)]
    block_reason: Option<BlockReason>,
    #[serde(default)]
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum BlockReason {
    #[serde(rename = "BLOCK_REASON_UNSPECIFIED")]
    Unspecified,
    Safety,
    Other,
    Blocklist,
    ProhibitedContent,
    ImageSafety,
}

/// TODO: add missing fields https://ai.google.dev/api/generate-content#UsageMetadata
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: Option<i32>,
    #[serde(default)]
    candidates_token_count: Option<i32>,
    #[serde(default)]
    total_token_count: Option<i32>,
}

/// TODO: add category and probability https://ai.google.dev/api/generate-content#v1beta.SafetyRating
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SafetyRating {
    category: String,
    probability: String,
    #[serde(default)]
    blocked: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingMetadata {
    #[serde(default)]
    web_search_queries: Option<Vec<String>>,
    #[serde(default)]
    retrieval_queries: Option<Vec<String>>,
    #[serde(default)]
    search_entry_point: Option<SearchEntryPoint>,
    #[serde(default)]
    grounding_chunks: Option<Vec<GroundingChunk>>,
    #[serde(default)]
    grounding_supports: Option<Vec<GroundingSupport>>,
    #[serde(default)]
    retrieval_metadata: Option<RetrievalMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchEntryPoint {
    rendered_content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingChunk {
    #[serde(default)]
    web: Option<WebSource>,
    #[serde(default)]
    retrieved_context: Option<RetrievedContext>,
}

#[derive(Debug, Deserialize)]
struct WebSource {
    uri: String,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RetrievedContext {
    uri: String,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroundingSupport {
    segment: Segment,
    #[serde(default)]
    segment_text: Option<String>,
    #[serde(default)]
    grounding_chunk_indices: Option<Vec<i32>>,
    #[serde(default)]
    support_chunk_indices: Option<Vec<i32>>,
    #[serde(default)]
    confidence_scores: Option<Vec<f32>>,
    #[serde(default)]
    confidence_score: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Segment {
    #[serde(default)]
    start_index: Option<i32>,
    #[serde(default)]
    end_index: Option<i32>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RetrievalMetadata {
    #[serde(default)]
    web_dynamic_retrieval_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_timestamp: Option<bool>,
}

impl From<ChatSettings> for Option<GenerationConfig> {
    fn from(settings: ChatSettings) -> Self {
        if settings.max_tokens.is_none()
            && settings.temperature.is_none()
            && settings.stop_sequences.is_none()
            && settings.seed.is_none()
        {
            return None;
        }

        Some(GenerationConfig {
            max_output_tokens: settings.max_tokens,
            temperature: settings.temperature,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: settings.stop_sequences.clone(),
            seed: settings.seed,
            response_mime_type: None,
            response_schema: None,
            audio_timestamp: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: SafetyCategory,
    pub threshold: HarmBlockThreshold,
}

impl SafetySetting {
    /// Create a new SafetySetting with the given category and threshold
    pub fn new(category: SafetyCategory, threshold: HarmBlockThreshold) -> Self {
        Self {
            category,
            threshold,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyCategory {
    HarmCategoryUnspecified,
    HarmCategoryHateSpeech,
    HarmCategoryDangerousContent,
    HarmCategoryHarassment,
    HarmCategorySexuallyExplicit,
    HarmCategoryCivicIntegrity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tools {
    #[serde(skip_serializing_if = "Option::is_none")]
    function_declarations: Option<Vec<FunctionDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    google_search: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    google_search_retrieval: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    function_calling_config: Option<FunctionCallingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FunctionCallingConfig {
    mode: FunctionCallingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum FunctionCallingMode {
    Auto,
    None,
    Any,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
    prompt_feedback: Option<PromptFeedback>,
    usage_metadata: UsageMetadata,
    model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum FinishReason {
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    Unspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Other,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
    ImageSafety,
}

#[derive(Debug, Deserialize)]
struct StreamResponseContent {
    parts: Vec<StreamResponsePart>,
}

#[derive(Debug, Deserialize)]
struct StreamResponsePart {
    text: String,
    role: Role,
}

/// Gemini Provider specific settings
#[derive(Debug, Clone, Default)]
pub struct GeminiSettings {
    /// Optional. The name of the cached content used as context to serve the prediction.
    /// Format: cachedContents/{cachedContent}
    pub cached_content: Option<String>,

    /// Optional. Enable structured output. Default is true.
    ///
    /// This is useful when the JSON Schema contains elements that are
    /// not supported by the OpenAPI schema version that
    /// Google Generative AI uses. You can use this to disable
    /// structured outputs if you need to.
    pub structured_outputs: Option<bool>,

    /// Optional. A list of unique safety settings for blocking unsafe content.
    pub safety_settings: Option<Vec<SafetySetting>>,

    /// Optional. Enables timestamp understanding for audio-only files.
    pub audio_timestamp: Option<bool>,

    /// Optional. When enabled, the model will use Google search to ground the response.
    pub use_search_grounding: Option<bool>,
}

impl GeminiSettings {
    /// Create a new GeminiSettings instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cached content
    pub fn cached_content(mut self, cached_content: impl Into<String>) -> Self {
        self.cached_content = Some(cached_content.into());
        self
    }

    /// Enable or disable structured outputs
    pub fn structured_outputs(mut self, enabled: bool) -> Self {
        self.structured_outputs = Some(enabled);
        self
    }

    /// Set safety settings
    pub fn safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Add a single safety setting
    pub fn add_safety_setting(
        mut self,
        category: SafetyCategory,
        threshold: HarmBlockThreshold,
    ) -> Self {
        let settings = self.safety_settings.get_or_insert_with(Vec::new);
        settings.push(SafetySetting {
            category,
            threshold,
        });
        self
    }

    /// Enable or disable audio timestamp
    pub fn audio_timestamp(mut self, enabled: bool) -> Self {
        self.audio_timestamp = Some(enabled);
        self
    }

    /// Enable or disable search grounding
    pub fn use_search_grounding(mut self, enabled: bool) -> Self {
        self.use_search_grounding = Some(enabled);
        self
    }

    /// Convert to a Box<dyn ProviderOptions> for use with ChatSettings
    pub fn into_provider_options(self) -> Box<dyn crate::model::chat::ProviderOptions> {
        Box::new(self)
    }
}

impl From<ChatSettings> for Option<GeminiSettings> {
    fn from(settings: ChatSettings) -> Self {
        settings
            .provider_options
            .as_ref()
            .and_then(move |options| options.as_any().downcast_ref::<GeminiSettings>())
            .cloned()
    }
}

impl crate::model::chat::ProviderOptions for GeminiSettings {
    fn clone_box(&self) -> Box<dyn crate::model::chat::ProviderOptions> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl TryFrom<ChatMessage> for Content {
    type Error = AIError;

    fn try_from(msg: ChatMessage) -> Result<Self, Self::Error> {
        let role = match msg.role {
            // System messages are handled separately via SystemInstruction
            ChatRole::System => {
                return Err(AIError::UnsupportedFunctionality(
                    "system messages should be set in ChatSettings.system_prompt, not in messages"
                        .to_string(),
                ));
            }
            ChatRole::User => Role::User,
            ChatRole::Assistant => Role::Model,
        };

        Ok(Content {
            role,
            parts: vec![Part { text: msg.content }],
        })
    }
}

impl TryFrom<UsageMetadata> for LanguageModelUsage {
    type Error = AIError;

    fn try_from(value: UsageMetadata) -> Result<Self, Self::Error> {
        match (
            value.prompt_token_count,
            value.candidates_token_count,
            value.total_token_count,
        ) {
            (Some(prompt_tokens), None, None) => Ok(LanguageModelUsage {
                prompt_tokens,
                completion_tokens: 0,
                total_tokens: prompt_tokens,
            }),
            (Some(prompt_tokens), Some(completion_tokens), _) => Ok(LanguageModelUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            _ => Err(AIError::ConversionError(
                "Missing token usage metadata".to_string(),
            )),
        }
    }
}
