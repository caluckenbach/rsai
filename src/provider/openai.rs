use async_trait::async_trait;
use futures::Stream;
use reqwest::{Client, header};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::any::Any;
use std::collections::HashMap;

use crate::AIError;
use crate::model::chat::{self, FinishReason, LanguageModelUsage, ProviderOptions, Temperature};
use crate::model::chat::{StructuredOutput, StructuredOutputParameters};
use crate::model::{ChatRole, ChatSettings, Message, TextCompletion, TextStream};

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
    // audio
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    // logit_bias
    // logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<i64>,
    // metadata
    // modalities
    parallel_tool_calls: Option<bool>,
    // prediction
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    // reasoning_effort
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    // service_tier
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    // stream_options
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // tool_choice
    // tools
    // top_logprobs
    // top_p
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    // web_search_options
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

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    choices: Vec<ChatCompletionChoice>,
    created: i32,
    model: String,
    /// Is always `chat.completion`
    object: String,
    service_tier: Option<String>,
    system_fingerprint: String,
    usage: Usage,
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
    // audio
    // tool_calls
}

#[derive(Debug, Deserialize)]
struct Annotation {
    /// Is always `url_citation`
    #[serde(rename = "type")]
    annotation_type: String,
    url_citation: UrlCitation,
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

#[async_trait]
impl Provider for OpenAIProvider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality(
            "Not yet implemented".to_string(),
        ))
    }

    async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
        settings: &'a ChatSettings,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>> + 'a, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality(
            "Not yet implemented".to_string(),
        ))
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

