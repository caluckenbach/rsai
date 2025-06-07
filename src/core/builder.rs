use std::{env, marker::PhantomData};

use serde::de::Deserialize;

use crate::provider::openai::{self};

use super::{
    error::LlmError,
    types::{Message, StructuredRequest, Tool, ToolChoice},
};

mod private {
    use crate::core::types::{LlmResponse, StructuredRequest, StructuredResponse};
    use crate::core::error::LlmError;
    use crate::core::traits::LlmProvider;
    use crate::provider::openai::OpenAiClient;
    
    pub struct Init;
    pub struct ProviderSet;
    pub struct ApiKeySet;
    pub struct Configuring;
    pub struct MessagesSet;
    pub struct ToolsSet;

    /// Marker trait for states that can execute complete() requests
    pub trait CompletableState {
        type Output<T>;
        
        async fn execute_request<T>(
            client: &OpenAiClient,
            request: StructuredRequest,
        ) -> Result<Self::Output<T>, LlmError>
        where
            T: serde::de::DeserializeOwned + Send + schemars::JsonSchema;
    }
    
    impl CompletableState for MessagesSet {
        type Output<T> = StructuredResponse<T>;
        
        async fn execute_request<T>(
            client: &OpenAiClient,
            request: StructuredRequest,
        ) -> Result<Self::Output<T>, LlmError>
        where
            T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
        {
            client.generate_structured(request).await
        }
    }
    
    impl CompletableState for ToolsSet {
        type Output<T> = LlmResponse<T>;
        
        async fn execute_request<T>(
            client: &OpenAiClient,
            request: StructuredRequest,
        ) -> Result<Self::Output<T>, LlmError>
        where
            T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
        {
            client.generate(request).await
        }
    }
}

/// A type-safe builder for constructing LLM requests using the builder pattern.
/// The builder enforces correct construction order through phantom types.
pub struct LlmBuilder<State> {
    provider: Option<String>,
    model: Option<String>,
    messages: Option<Vec<Message>>,
    api_key: Option<String>,
    tools: Option<Box<[Tool]>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    _state: PhantomData<State>,
}

impl<State> LlmBuilder<State> {
    pub(crate) fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub(crate) fn set_api_key(&mut self, api_key: &str) {
        self.api_key = Some(api_key.to_string());
    }

    pub(crate) fn get_api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

impl LlmBuilder<private::Init> {
    /// Set the provider for the LLM request.
    /// Currently supports "openai".
    pub fn provider(self, provider: &str) -> Result<LlmBuilder<private::ProviderSet>, LlmError> {
        match provider {
            "openai" => Ok(LlmBuilder {
                provider: Some(provider.to_string()),
                model: None,
                messages: None,
                api_key: None,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                _state: PhantomData,
            }),
            _ => Err(LlmError::Builder("Unsupported Provider".to_string())),
        }
    }
}

/// Configuration for API key source
pub enum ApiKey {
    /// Use the default environment variable for the provider
    Default,
    /// Use a custom API key string
    Custom(String),
}

impl LlmBuilder<private::ProviderSet> {
    /// Set the API key for the provider.
    /// Use `ApiKey::Default` to load from environment variables or `ApiKey::Custom` for a custom key.
    pub fn api_key(self, api_key: ApiKey) -> Result<LlmBuilder<private::ApiKeySet>, LlmError> {
        let key = match api_key {
            ApiKey::Default => match self.provider.as_deref() {
                Some("openai") => env::var("OPENAI_API_KEY").map_err(|_| {
                    LlmError::Builder("Missing OPENAI_API_KEY environment variable".to_string())
                })?,
                _ => {
                    return Err(LlmError::Builder(
                        "Can't load API Key for unsupported Provider".to_string(),
                    ));
                }
            },
            ApiKey::Custom(custom_key) => custom_key,
        };

        Ok(LlmBuilder {
            provider: self.provider,
            model: None,
            messages: None,
            api_key: Some(key),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            _state: PhantomData,
        })
    }
}

impl LlmBuilder<private::ApiKeySet> {
    /// Set the model to use for the LLM request.
    pub fn model(self, model_id: &str) -> LlmBuilder<private::Configuring> {
        LlmBuilder {
            provider: self.provider,
            model: Some(model_id.to_string()),
            messages: None,
            api_key: self.api_key,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            _state: PhantomData,
        }
    }
}

impl LlmBuilder<private::Configuring> {
    /// Set the messages for the conversation.
    pub fn messages(self, messages: Vec<Message>) -> LlmBuilder<private::MessagesSet> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: Some(messages),
            api_key: self.api_key,
            tools: self.tools,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            _state: PhantomData,
        }
    }
}

fn validate_builder<State>(
    builder: &LlmBuilder<State>,
) -> Result<(&Vec<Message>, &str, &str), LlmError> {
    builder.api_key.as_ref().ok_or(LlmError::Builder(
        "Missing API key. Make sure to specify an API key.".into(),
    ))?;

    let messages = builder
        .messages
        .as_ref()
        .filter(|messages| !messages.is_empty())
        .ok_or(LlmError::Builder(
            "Missing messages. Make sure to add at least one message.".to_string(),
        ))?;

    let provider = builder.provider.as_ref().ok_or(LlmError::Builder(
        "Missing provider. Make sure to specify a provider.".into(),
    ))?;

    let model = builder.model.as_ref().ok_or(LlmError::Builder(
        "Missing model. Make sure to specify a model.".into(),
    ))?;

    Ok((messages, provider, model))
}

impl<State> LlmBuilder<State>
where
    State: private::CompletableState,
{
    /// Execute the LLM request and return structured output of type T.
    /// The type T must implement Deserialize and JsonSchema for structured output generation as well as
    /// be annotated with `#[schemars(deny_unknown_fields)]`.
    /// Use the `completion_schema` attribute macro to easily define structured output types.
    ///
    /// # Example
    /// ```
    /// use ai::{completion_schema, llm, Message, ChatRole, ApiKey};
    ///
    /// #[completion_schema]
    /// struct Analysis {
    ///     sentiment: String,
    ///     confidence: f32,
    /// }
    ///
    /// let analysis = llm::call()
    ///     .provider("openai")?
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Analyze: 'This library is amazing!'".to_string(),
    ///     }])
    ///     .complete::<Analysis>()
    ///     .await?;
    /// ```
    pub async fn complete<T>(mut self) -> Result<State::Output<T>, LlmError>
    where
        T: for<'a> Deserialize<'a> + Send + schemars::JsonSchema,
    {
        let (messages, provider, model) = validate_builder(&self)?;
        let model_string = model.to_string();
        let messages = messages.to_vec();

        match provider {
            "openai" => {
                if self.api_key.is_none() {
                    let api_key = env::var("OPENAI_API_KEY")
                        .map_err(|_| LlmError::Builder("Missing OPENAI_API_KEY.".to_string()))?;
                    self.set_api_key(&api_key);
                }

                let req = StructuredRequest {
                    model: model_string,
                    messages,
                    tools: self.tools.clone(),
                    tool_choice: self.tool_choice.clone(),
                    parallel_tool_calls: self.parallel_tool_calls,
                };
                let client = openai::create_openai_client_from_builder(&self)?;
                State::execute_request(&client, req).await
            }
            _ => todo!(),
        }
    }
}

impl LlmBuilder<private::MessagesSet> {
    /// Set the tools for the LLM request.
    /// This transitions to the ToolsSet state where tool_choice and parallel_tool_calls can be configured.
    ///
    /// By default parallel tool calling is enabled. This can be changed by calling `parallel_tool_calls` with `false`.
    pub fn tools(self, tools: impl Into<Box<[Tool]>>) -> LlmBuilder<private::ToolsSet> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: self.messages,
            api_key: self.api_key,
            tools: Some(tools.into()),
            tool_choice: self.tool_choice,
            parallel_tool_calls: Some(true),
            _state: PhantomData,
        }
    }
}

impl LlmBuilder<private::ToolsSet> {
    /// Set the tool choice using a type-safe enum that implements Into<ToolChoice>.
    /// This accepts the generated toolset::Choice enum for compile-time validation.
    pub fn tool_choice<TC>(mut self, choice: TC) -> Self
    where
        TC: Into<ToolChoice>,
    {
        self.tool_choice = Some(choice.into());
        self
    }

    /// Set whether to enable parallel tool calls.
    /// When true, the model can call multiple tools simultaneously.
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }
}

/// Module containing the main entry point for building LLM requests
pub mod llm {
    use super::*;

    /// Create a new LLM builder to start constructing a request
    pub fn call() -> LlmBuilder<private::Init> {
        LlmBuilder {
            provider: None,
            model: None,
            messages: None,
            api_key: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            _state: PhantomData,
        }
    }
}
