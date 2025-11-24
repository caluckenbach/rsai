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

use crate::provider::constants::openai;

use crate::core::{
    LlmBuilder, LlmError, LlmProvider, StructuredRequest, ToolCallingConfig, ToolCallingGuard,
    ToolRegistry,
};
use crate::responses::{HttpClientConfig, ResponsesClient, ResponsesProviderConfig};
use async_trait::async_trait;

/// OpenAI-specific configuration for the responses client
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
    /// Configuration for tool calling limits
    pub tool_calling_config: Option<ToolCallingConfig>,
    pub http_config: HttpClientConfig,
}

impl OpenAiConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openai::API_BASE.to_string(),
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

    pub fn get_tool_calling_guard(&self) -> ToolCallingGuard {
        if let Some(ref config) = self.tool_calling_config {
            ToolCallingGuard::with_limits(config.max_iterations, config.timeout)
        } else {
            ToolCallingGuard::new()
        }
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Self {
        self.http_config = config;
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

    fn provider(&self) -> super::Provider {
        self.provider()
    }

    fn http_config(&self) -> HttpClientConfig {
        self.http_config.clone()
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
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let config = OpenAiConfig::new(api_key);
        Ok(Self {
            responses_client: ResponsesClient::new(config)?,
        })
    }

    pub fn with_base_url(mut self, base_url: String) -> Result<Self, LlmError> {
        // Create a new config with the updated base_url using the current API key
        let current_api_key = &self.responses_client.config.api_key;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url,
            tool_calling_config: self.responses_client.config.tool_calling_config.clone(),
            http_config: self.responses_client.config.http_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_tool_calling_config(mut self, config: ToolCallingConfig) -> Result<Self, LlmError> {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: Some(config),
            http_config: self.responses_client.config.http_config.clone(),
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }

    pub fn with_http_config(mut self, config: HttpClientConfig) -> Result<Self, LlmError> {
        let current_api_key = &self.responses_client.config.api_key;
        let base_url = &self.responses_client.config.base_url;
        let tool_config = &self.responses_client.config.tool_calling_config;

        let new_config = OpenAiConfig {
            api_key: current_api_key.clone(),
            base_url: base_url.clone(),
            tool_calling_config: tool_config.clone(),
            http_config: config,
        };
        self.responses_client = ResponsesClient::new(new_config)?;
        Ok(self)
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn generate_completion<T>(
        &self,
        request: StructuredRequest,
        format: crate::responses::Format,
        tool_registry: Option<&ToolRegistry>,
    ) -> Result<T::Output, LlmError>
    where
        T: crate::CompletionTarget + Send,
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
                .handle_tool_calling_loop::<T>(request, tool_registry, &mut guard, format)
                .await;
        }

        // Otherwise, make a single request expecting structured content
        let messages_clone = request.messages.clone();
        let responses_request = self.responses_client.build_request_with_format(
            &request,
            &crate::responses::convert_messages_to_responses_format(messages_clone)?,
            format,
        )?;
        let api_response = self
            .responses_client
            .make_api_request(responses_request)
            .await?;
        T::parse_response(api_response, super::Provider::OpenAI)
    }
}

pub fn create_openai_client_from_builder<State>(
    builder: &LlmBuilder<State>,
) -> Result<OpenAiClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENAI_API_KEY not set.".to_string()))?
        .to_string();

    let mut config = OpenAiConfig::new(api_key);

    if let Some(http_config) = &builder.get_http_config() {
        config = config.with_http_config((*http_config).clone());
    }

    let client = ResponsesClient::new(config)?;
    Ok(OpenAiClient {
        responses_client: client,
    })
}
