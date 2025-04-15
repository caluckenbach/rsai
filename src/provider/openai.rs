use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::AIError;
use crate::model::chat::StructuredOutput;
use crate::model::chat::StructuredOutputParameters;
use crate::model::{ChatSettings, TextCompletion, TextStream};

use super::Provider;

const DEFAULT_MODEL: &str = "gpt-4o";
const API_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAIProvider {
    api_key: String,
    model: String,
    base_url: String,
    client: Client,
}

impl OpenAIProvider {
    /// Creates a new `OpenAIProvider` with the specified API key and model
    pub fn new(api_key: &str, model: &str) -> Self {
        OpenAIProvider {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: API_BASE_URL.to_string(),
            client: Client::new(),
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
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality("Not yet implemented".to_string()))
    }

    async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
        settings: &'a ChatSettings,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>> + 'a, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality("Not yet implemented".to_string()))
    }

    async fn generate_object<T: DeserializeOwned + JsonSchema + Sync>(
        &self,
        prompt: &str,
        settings: &ChatSettings,
        parameters: &StructuredOutputParameters<T>,
    ) -> Result<StructuredOutput<T>, AIError> {
        // Implementation will be added later
        Err(AIError::UnsupportedFunctionality("Not yet implemented".to_string()))
    }
}