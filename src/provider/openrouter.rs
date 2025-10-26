//! OpenRouter provider implementation.
//!
//! # API Compatibility
//!
//! This module preserves all fields from the OpenRouter API responses, even those not currently used.
//! Fields marked with `#[allow(dead_code)]` are retained for:
//! - API contract completeness
//! - Future compatibility without breaking changes
//! - Debugging and logging purposes
//!
//! When adding new API structs, include all fields from the OpenRouter documentation and mark
//! unused ones with `#[allow(dead_code)]` rather than omitting them.

use crate::core::types::{StructuredRequest, StructuredResponse, ToolRegistry};
use crate::provider::constants::openrouter;
use crate::responses::{ResponsesClient, ResponsesProviderConfig};

use crate::core::{builder::LlmBuilder, error::LlmError, traits::LlmProvider};
use async_trait::async_trait;

/// OpenRouter-specific configuration for the responses client
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub http_referer: Option<String>,
    pub x_title: Option<String>,
}

impl OpenRouterConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: openrouter::API_BASE.to_string(),
            http_referer: None,
            x_title: None,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.http_referer = Some(http_referer);
        self
    }

    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.x_title = Some(x_title);
        self
    }
}

impl ResponsesProviderConfig for OpenRouterConfig {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self) -> &str {
        openrouter::RESPONSES_ENDPOINT
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

    fn extra_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        if let Some(referer) = &self.http_referer {
            headers.push(("HTTP-Referer".to_string(), referer.clone()));
        }

        if let Some(title) = &self.x_title {
            headers.push(("X-Title".to_string(), title.clone()));
        }

        headers
    }
}

impl OpenRouterConfig {
    /// Get the provider type for this configuration
    pub fn provider(&self) -> crate::provider::Provider {
        crate::provider::Provider::OpenRouter
    }
}

pub struct OpenRouterClient {
    responses_client: ResponsesClient<OpenRouterConfig>,
}

impl OpenRouterClient {
    pub fn new(api_key: String) -> Self {
        let config = OpenRouterConfig::new(api_key);
        Self {
            responses_client: ResponsesClient::new(config),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        // Create a new config with the updated base_url using the current API key
        let current_api_key = &self.responses_client.config.api_key;
        let http_referer = self.responses_client.config.http_referer.clone();
        let x_title = self.responses_client.config.x_title.clone();

        let new_config = OpenRouterConfig {
            api_key: current_api_key.clone(),
            base_url,
            http_referer,
            x_title,
        };
        self.responses_client = ResponsesClient::new(new_config);
        self
    }

    pub fn with_http_referer(mut self, http_referer: String) -> Self {
        self.responses_client.config.http_referer = Some(http_referer);
        self
    }

    pub fn with_x_title(mut self, x_title: String) -> Self {
        self.responses_client.config.x_title = Some(x_title);
        self
    }
}

#[async_trait]
impl LlmProvider for OpenRouterClient {
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
        crate::responses::create_core_structured_response(api_response, super::Provider::OpenRouter)
    }
}

pub fn create_openrouter_client_from_builder<State>(
    builder: &LlmBuilder<State>,
) -> Result<OpenRouterClient, LlmError> {
    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENROUTER_API_KEY not set.".to_string()))?
        .to_string();

    let client = OpenRouterClient::new(api_key);
    Ok(client)
}
