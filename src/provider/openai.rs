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

use crate::provider::ToolCallingConfig;
use crate::provider::constants::openai;

use crate::core::{
    LlmBuilder, LlmError, LlmProvider, StructuredRequest, StructuredResponse, ToolRegistry,
};
use crate::responses::{ResponsesClient, ResponsesProviderConfig, ToolCallingGuard};
use async_trait::async_trait;

/// OpenAI-specific configuration for the responses client
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
    /// Configuration for tool calling limits
    pub tool_calling_config: Option<ToolCallingConfig>,
}

impl OpenAiConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openai::API_BASE.to_string(),
            tool_calling_config: Some(ToolCallingConfig::default()),
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

    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::with_limits(config.max_iterations, config.timeout)
        } else {
            ToolCallingGuard::new()
        }
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
            tool_calling_config: self.responses_client.config.tool_calling_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config);
        self
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Self {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: Some(config),
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

        if has_tools && let Some(tool_registry) = tool_registry {
            let mut guard = self.responses_client.config.get_tool_calling_guard();
            return self
                .responses_client
                .handle_tool_calling_loop(request, tool_registry, &mut guard)
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
