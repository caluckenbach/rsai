use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashMap;

use crate::AIError;
use crate::model::chat::{FinishReason, calculate_language_model_usage};
use crate::model::{ChatMessage, ChatRole, ChatSettings, TextCompletion};

use super::Provider;

pub struct GeminiProvider {
    pub api_key: String,
    pub model: String,
    pub client: Client,
}

impl GeminiProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        GeminiProvider {
            api_key: api_key.to_string(),
            client: Client::new(),
            model: model.to_string(),
        }
    }

    /// Create a new GeminiProvider with default model
    ///
    /// default model: `gemini-2.0-flash`
    pub fn default(api_key: &str) -> Self {
        Self::new(api_key, "gemini-2.0-flash")
    }

    fn get_base_url(&self, stream: bool) -> String {
        let model_url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            self.model
        );

        if stream {
            format!(
                "{}:streamGenerateContent?alt=sse&key={}",
                model_url, self.api_key
            )
        } else {
            format!("{}:generateContent?key={}", model_url, self.api_key)
        }
    }

    fn build_generation_config(&self, settings: &ChatSettings) -> Option<GenerationConfig> {
        // Only create the config if at least one setting is provided
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

    fn get_provider_settings<'a>(&self, settings: &'a ChatSettings) -> Option<&'a GeminiSettings> {
        settings
            .provider_options
            .as_ref()
            .and_then(move |options| options.as_any().downcast_ref::<GeminiSettings>())
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError> {
        let url = self.get_base_url(false);

        let mut contents = Vec::new();

        // Handle regular chat messages
        if let Some(messages) = &settings.messages {
            for msg in messages {
                let content = Content::try_from(msg.clone())?;
                contents.push(content);
            }
        }

        // Add the current prompt as a user message
        contents.push(Content {
            role: Role::User,
            parts: vec![Part {
                text: Some(prompt.to_string()),
                function_call: None,
                inline_data: None,
            }],
        });

        // Create system instruction if a system prompt is provided
        let system_instruction = settings
            .system_prompt
            .as_ref()
            .map(|prompt| SystemInstruction {
                parts: vec![Part {
                    text: Some(prompt.clone()),
                    function_call: None,
                    inline_data: None,
                }],
            });

        // Build generation config from general settings
        let generation_config = self.build_generation_config(settings);

        // Get provider-specific settings
        let provider_settings = self.get_provider_settings(settings);

        // Extract provider-specific settings
        let safety_settings = provider_settings.and_then(|s| s.safety_settings.clone());
        let cached_content = provider_settings.and_then(|s| s.cached_content.clone());

        // Build search tools if grounding is enabled
        let tools = if provider_settings
            .and_then(|s| s.use_search_grounding)
            .unwrap_or(false)
        {
            // Detect if this is a Gemini 2.0 model
            let is_gemini2 = self.model.contains("gemini-2");

            if is_gemini2 {
                Some(Tools {
                    function_declarations: None,
                    google_search: Some(HashMap::new()),
                    google_search_retrieval: None,
                })
            } else {
                Some(Tools {
                    function_declarations: None,
                    google_search: None,
                    google_search_retrieval: Some(HashMap::new()),
                })
            }
        } else {
            None
        };

        // Create the full request
        let message = Request {
            contents,
            system_instruction,
            generation_config,
            safety_settings,
            tools,
            tool_config: None, // No tool config for now
            cached_content,
        };

        // Send the request
        let response = self
            .client
            .post(&url)
            .json(&message)
            .send()
            .await
            .map_err(|e| AIError::RequestError(e.to_string()))?;

        // Check if the request was successful
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AIError::ApiError(format!(
                "API returned error status {}: {}",
                status, error_text
            )));
        }

        // Parse the response JSON directly
        let gemini_response = response.json::<GeminiResponse>().await.map_err(|e| {
            AIError::ConversionError(format!("Failed to parse Gemini response: {}", e))
        })?;

        // Check if we got any candidates
        if gemini_response.candidates.is_empty() {
            return Err(AIError::ApiError(
                "No response candidates returned".to_string(),
            ));
        }

        // Get the first candidate
        let candidate = &gemini_response.candidates[0];

        // Check if we got blocked by safety filters
        if let Some(reason) = &candidate.finish_reason {
            if reason == "SAFETY" {
                return Err(AIError::ApiError(
                    "Response blocked by safety filters".to_string(),
                ));
            }
        }

        // Extract the text from the parts
        if candidate.content.parts.is_empty() {
            return Err(AIError::ApiError(
                "No content parts in response".to_string(),
            ));
        }

        // Check for function call response
        //if let Some(function_call) = &candidate.content.parts[0].function_call {
        //    return Ok(format!(
        //        "Function call: {} with args: {}",
        //        function_call.name,
        //        function_call.args.to_string()
        //    ));
        //}

        // Get text from the first part (if available)
        if let Some(text) = &candidate.content.parts[0].text {
            // Extract finish reason
            let finish_reason = match &candidate.finish_reason {
                Some(reason) => match reason.as_str() {
                    "STOP" => FinishReason::Stop,
                    "MAX_TOKENS" => FinishReason::Length,
                    "SAFETY" => FinishReason::ContentFilter,
                    "RECITATION" => FinishReason::Other,
                    "TOOL_CALLS" => FinishReason::ToolCalls,
                    "ERROR" => FinishReason::Error,
                    _ => FinishReason::Unknown,
                },
                None => FinishReason::Unknown,
            };

            // Extract usage
            let usage = calculate_language_model_usage(
                gemini_response
                    .usage_metadata
                    .as_ref()
                    .and_then(|m| m.prompt_token_count)
                    .unwrap_or(0),
                gemini_response
                    .usage_metadata
                    .as_ref()
                    .and_then(|m| m.candidates_token_count)
                    .unwrap_or(0),
            );

            let completion = TextCompletion {
                text: text.to_string(),
                reasoning_text: None,
                finish_reason,
                usage,
            };

            return Ok(completion);
        }

        // If we get here, we don't have text or function call
        Err(AIError::ApiError(
            "Response contained no text or function call".to_string(),
        ))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: FunctionResponseContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponseContent {
    pub name: String,
    pub content: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: Role,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Tools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
}

// Response structs for parsing Gemini API responses

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Content,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: Option<i32>,
    #[serde(default)]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    #[serde(default)]
    pub grounding_metadata: Option<GroundingMetadata>,
    #[serde(default)]
    pub avg_logprobs: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(default)]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    #[serde(default)]
    pub block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: Option<i32>,
    #[serde(default)]
    pub candidates_token_count: Option<i32>,
    #[serde(default)]
    pub total_token_count: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
    #[serde(default)]
    pub probability_score: Option<f32>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub severity_score: Option<f32>,
    #[serde(default)]
    pub blocked: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    #[serde(default)]
    pub web_search_queries: Option<Vec<String>>,
    #[serde(default)]
    pub retrieval_queries: Option<Vec<String>>,
    #[serde(default)]
    pub search_entry_point: Option<SearchEntryPoint>,
    #[serde(default)]
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
    #[serde(default)]
    pub grounding_supports: Option<Vec<GroundingSupport>>,
    #[serde(default)]
    pub retrieval_metadata: Option<RetrievalMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    pub rendered_content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    #[serde(default)]
    pub web: Option<WebSource>,
    #[serde(default)]
    pub retrieved_context: Option<RetrievedContext>,
}

#[derive(Debug, Deserialize)]
pub struct WebSource {
    pub uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RetrievedContext {
    pub uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    pub segment: Segment,
    #[serde(default)]
    pub segment_text: Option<String>,
    #[serde(default)]
    pub grounding_chunk_indices: Option<Vec<i32>>,
    #[serde(default)]
    pub support_chunk_indices: Option<Vec<i32>>,
    #[serde(default)]
    pub confidence_scores: Option<Vec<f32>>,
    #[serde(default)]
    pub confidence_score: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    #[serde(default)]
    pub start_index: Option<i32>,
    #[serde(default)]
    pub end_index: Option<i32>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    #[serde(default)]
    pub web_dynamic_retrieval_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_timestamp: Option<bool>,
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
pub struct Tools {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_retrieval: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: FunctionCallingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    Auto,
    None,
    Any,
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
            parts: vec![Part {
                text: Some(msg.content),
                function_call: None,
                inline_data: None,
            }],
        })
    }
}
