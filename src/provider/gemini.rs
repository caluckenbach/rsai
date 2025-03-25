use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use reqwest::Client;
use schemars::JsonSchema;
use schemars::schema::SchemaObject;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::str::from_utf8;

use crate::AIError;
use crate::model::chat;
use crate::model::chat::StructuredOutput;
use crate::model::chat::StructuredOutputParameters;
use crate::model::chat::StructuredResult;
use crate::model::chat::{LanguageModelUsage, Temperature};
use crate::model::{ChatRole, ChatSettings, Message, TextCompletion, TextStream};

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
            .map_or(chat::FinishReason::Unknown, from_provider_finish_reason);

        let usage = from_provider_usage(parsed_response.usage_metadata)?;

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

    async fn generate_object<T: DeserializeOwned + JsonSchema + Sync>(
        &self,
        prompt: &str,
        settings: &ChatSettings,
        parameters: &StructuredOutputParameters<T>,
    ) -> Result<StructuredOutput<T>, AIError> {
        if parameters.output == chat::OutputType::NoSchema
            && parameters.mode != Some(chat::Mode::Json)
        {
            return Err(AIError::InvalidInput(
                "Mode must be 'json' for 'OutputType::NoSchema'".to_string(),
            ));
        }

        let url = format!("{}:generateContent", self.get_model_url());
        let mut request = build_request(prompt, settings)?;

        request.generation_config = to_structured_generation_config(settings, parameters);

        if let Some(provider_settings) = to_provider_settings(settings) {
            if let Some(false) = provider_settings.structured_outputs {
                return Err(AIError::UnsupportedFunctionality(
                    "Structured output is disabled".to_string(),
                ));
            }
        }

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
            return Err(AIError::ApiError(format!(
                "error status {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let parsed_response = response
            .json::<GenerateContentResponse>()
            .await
            .map_err(|e| AIError::ConversionError(format!("Failed to parse response: {}", e)))?;

        let candidate = parsed_response
            .candidates
            .first()
            .ok_or_else(|| AIError::ApiError("No response candidates returned".to_string()))?;

        let content_text = candidate
            .content
            .parts
            .first()
            .ok_or_else(|| AIError::ApiError("No content parts in response".to_string()))?
            .text
            .clone();

        eprint!("{}", content_text);

        let value = match parameters.output {
            chat::OutputType::Object => {
                let parsed = serde_json::from_str(&content_text).map_err(|e| {
                    AIError::ConversionError(format!("Failed to parse JSON object: {}", e))
                })?;
                StructuredResult::Object(parsed)
            }
            chat::OutputType::Array => {
                let parsed = serde_json::from_str(&content_text).map_err(|e| {
                    AIError::ConversionError(format!("Failed to parse JSON array: {}", e))
                })?;
                StructuredResult::Array(parsed)
            }
            chat::OutputType::Enum => todo!(),
            chat::OutputType::NoSchema => {
                let parsed = serde_json::from_str(&content_text)
                    .map_err(|e| AIError::ConversionError(format!("Filed to parse JSON: {}", e)))?;
                StructuredResult::NoSchema(parsed)
            }
        };

        let finish_reason = candidate
            .finish_reason
            .clone()
            .map_or(chat::FinishReason::Unknown, from_provider_finish_reason);

        let usage = from_provider_usage(parsed_response.usage_metadata)?;

        Ok(StructuredOutput {
            value,
            finish_reason,
            usage,
        })
    }
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

impl chat::ProviderOptions for GeminiSettings {
    fn clone_box(&self) -> Box<dyn crate::model::chat::ProviderOptions> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

/// Configuration options for model generation and outputs.
///
/// Not all parameters are configurable for every model.
///
/// [Google API Docs](https://ai.google.dev/api/generate-content#generationconfig)
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<MimeType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<SchemaObject>,
}

#[derive(Debug, Clone, Serialize)]
enum MimeType {
    /// Default
    #[serde(rename = "text/plain")]
    Text,
    #[serde(rename = "application/json")]
    Json,
    #[serde(rename = "text/x.enum")]
    Enum,
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

/// Builds the request payload for the Gemini API
fn build_request(prompt: &str, settings: &ChatSettings) -> Result<Request, AIError> {
    let mut contents = Vec::new();

    if let Some(messages) = &settings.messages {
        for msg in messages {
            let content = to_content(msg.clone())?;
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
        .clone()
        .map(|prompt| SystemInstruction {
            parts: vec![Part { text: prompt }],
        });

    let generation_config = to_generation_config(settings);
    let safety_settings = to_provider_settings(settings).and_then(|s| s.safety_settings);

    Ok(Request {
        contents,
        system_instruction,
        generation_config,
        safety_settings,
    })
}

/// Parses an SSE chunk into a `TextStream`
fn parse_sse_chunk(bytes: Bytes) -> Result<TextStream, AIError> {
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

    let Some(data) = chunk_str.strip_prefix("data: ") else {
        return Ok(TextStream {
            text: String::new(),
            finish_reason: chat::FinishReason::Other,
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
            finish_reason: chat::FinishReason::Other,
            usage: None,
        });
    }

    let text = chunk.candidates[0].content.parts[0].text.clone();
    let finish_reason = chunk.candidates[0]
        .finish_reason
        .clone()
        .map_or(chat::FinishReason::Unknown, from_provider_finish_reason);

    let usage = from_provider_usage(chunk.usage_metadata).ok();

    Ok(TextStream {
        text,
        finish_reason,
        usage,
    })
}

fn to_generation_config(settings: &ChatSettings) -> Option<GenerationConfig> {
    // Early return if no generation config parameters are set
    if settings.max_tokens.is_none()
        && settings.temperature.is_none()
        && settings.stop_sequences.is_none()
    {
        return None;
    }

    let temperature = settings.temperature.clone().map(to_provider_temperature);

    Some(GenerationConfig {
        stop_sequences: settings.stop_sequences.clone(),
        max_output_tokens: settings.max_tokens,
        temperature,
        response_mime_type: None,
        response_schema: None,
    })
}

fn to_structured_generation_config<T: DeserializeOwned + JsonSchema + Sync>(
    settings: &ChatSettings,
    params: &StructuredOutputParameters<T>,
) -> Option<GenerationConfig> {
    let mut cfg = to_generation_config(settings).unwrap_or(GenerationConfig {
        stop_sequences: None,
        max_output_tokens: None,
        temperature: None,
        response_mime_type: None,
        response_schema: None,
    });

    cfg.response_mime_type = to_provider_mime_type(params);
    cfg.response_schema = to_provider_response_schema(params);

    if cfg.response_mime_type.is_none() && cfg.response_schema.is_none() {
        return None;
    }

    Some(cfg)
}

fn to_provider_mime_type<T: DeserializeOwned>(
    params: &StructuredOutputParameters<T>,
) -> Option<MimeType> {
    match params.output {
        chat::OutputType::Object | chat::OutputType::Array => {
            if params.schema.is_some() {
                Some(MimeType::Json)
            } else {
                None
            }
        }
        chat::OutputType::Enum => Some(MimeType::Enum),
        chat::OutputType::NoSchema => None,
    }
}

fn to_provider_response_schema<T: DeserializeOwned + JsonSchema>(
    params: &StructuredOutputParameters<T>,
) -> Option<SchemaObject> {
    match params.output {
        chat::OutputType::Object | chat::OutputType::Array => {
            // Only generate schema if a schema instance is provided
            params
                .schema
                .as_ref()
                .map(|_| schemars::schema_for!(T).schema)
        }
        _ => None,
    }
}

/// Map Temperature enum to Gemini's temperature scale [0.0-2.0]
fn to_provider_temperature(temperature: Temperature) -> f32 {
    match temperature {
        Temperature::Raw(val) => val,
        Temperature::Normalized(val) => val * 2.0,
    }
}

fn to_provider_settings(settings: &ChatSettings) -> Option<GeminiSettings> {
    settings
        .provider_options
        .as_ref()
        .and_then(|options| options.as_any().downcast_ref::<GeminiSettings>())
        .cloned()
}

fn to_content(message: Message) -> Result<Content, AIError> {
    let role = match message.role {
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
        parts: vec![Part {
            text: message.content,
        }],
    })
}

fn from_provider_finish_reason(reason: FinishReason) -> chat::FinishReason {
    match reason {
            FinishReason::Unspecified => chat::FinishReason::Unknown,
            FinishReason::Stop => chat::FinishReason::Stop,
            FinishReason::MaxTokens => chat::FinishReason::Length,
            FinishReason::Safety => chat::FinishReason::ContentFilter("The response candidate content was flagged for safety reasons.".to_string()),
            FinishReason::Recitation => chat::FinishReason::ContentFilter("The response candidate content was flagged for recitation reasons.".to_string()),
            FinishReason::Language =>chat::FinishReason::Other,
            FinishReason::Other =>chat::FinishReason::Other,
            FinishReason::Blocklist => chat::FinishReason::ContentFilter("Token generation stopped because the content contains forbidden terms.".to_string()),
            FinishReason::ProhibitedContent => chat::FinishReason::ContentFilter("Token generation stopped for potentially containing prohibited content.".to_string()),
            FinishReason::Spii => chat::FinishReason::ContentFilter("Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).".to_string()),
            FinishReason::MalformedFunctionCall => chat::FinishReason::Error,
            FinishReason::ImageSafety => chat::FinishReason::ContentFilter("Token generation stopped because generated images contain safety violations.".to_string())}
}

fn from_provider_usage(usage: UsageMetadata) -> Result<LanguageModelUsage, AIError> {
    match (
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
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
