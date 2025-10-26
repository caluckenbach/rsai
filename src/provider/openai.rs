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

use crate::core::types::{StructuredRequest, StructuredResponse, ToolRegistry};
use crate::provider::constants::openai;
use crate::responses::{ResponsesClient, ResponsesProviderConfig};

use crate::core::{builder::LlmBuilder, error::LlmError, traits::LlmProvider};
use async_trait::async_trait;

/// OpenAI-specific configuration for the responses client
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
}

impl OpenAiConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openai::API_BASE.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

impl ResponsesProviderConfig for OpenAiConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self) -> &str {
        openai::RESPONSES_ENDPOINT
    }

    fn auth_header(&self) -> (String, String) {
        (
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        )
    }

    fn default_model(&self) -> &str {
        "gpt-4o"
    }

    fn provider(&self) -> super::Provider {
        self.provider()
    }
}

impl OpenAiConfig {
    /// Get the provider type for this configuration
    pub fn provider(&self) -> crate::provider::Provider {
        crate::provider::Provider::OpenAI
    }
}

pub struct OpenAiClient {
    responses_client: ResponsesClient<OpenAiConfig>,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        let config = OpenAiConfig::new(api_key);
        Self {
            responses_client: ResponsesClient::new(config),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        // Create a new config with the updated base_url using the current API key
        let current_api_key = &self.responses_client.config.api_key;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url,
        };
        self.responses_client = ResponsesClient::new(new_config);
        self
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
        let has_tools = request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.tools.as_ref())
            .is_some();

        if has_tools && tool_registry.is_some() {
            return self
                .responses_client
                .handle_tool_calling_loop(request, tool_registry.unwrap())
                .await;
        }

        // Otherwise, make a single request expecting structured content
        let messages_clone = request.messages.clone();
        let responses_request = self.responses_client.build_request::<T>(
            &request,
            &crate::responses::convert_messages_to_responses_format(messages_clone)?,
        )?;
        let api_response = self
            .responses_client
            .make_api_request(responses_request)
            .await?;
        crate::responses::create_core_structured_response(api_response, super::Provider::OpenAI)
    }
}

pub fn create_openai_client_from_builder<State>(
    builder: &LlmBuilder<State>,
) -> Result<OpenAiClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENAI_API_KEY not set.".to_string()))?
        .to_string();

    let client = OpenAiClient::new(api_key);
    Ok(client)
}
