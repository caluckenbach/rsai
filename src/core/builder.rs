use std::{env, marker::PhantomData};

use serde::de::Deserialize;

use crate::provider::{Provider, openai, openrouter};

use super::{
    error::LlmError,
    traits::LlmProvider,
    types::{
        ConversationMessage, GenerationConfig, Message, StructuredRequest, StructuredResponse,
        Tool, ToolChoice, ToolConfig, ToolRegistry,
    },
};

mod private {
    pub struct ProviderSet;
    pub struct ApiKeySet;
    pub struct Configuring;
    pub struct MessagesSet;
    pub struct ToolsSet;

    /// Marker trait for states that can call complete()
    pub trait Completable {}
    impl Completable for MessagesSet {}
    impl Completable for ToolsSet {}
}

/// Builder fields that are shared across all states
struct BuilderFields {
    // Core configuration
    provider: Option<Provider>,
    api_key: Option<String>,
    model: Option<String>,

    // Request content
    messages: Option<Vec<Message>>,

    // Tool configuration
    tools: Option<Box<[Tool]>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    tool_registry: Option<ToolRegistry>,

    // Generation parameters
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl BuilderFields {
    fn new() -> Self {
        Self {
            provider: None,
            api_key: None,
            model: None,
            messages: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            tool_registry: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Validate that all required fields are present
    fn validate(&self) -> Result<(&Vec<Message>, Provider, &str), LlmError> {
        self.api_key.as_ref().ok_or(LlmError::Builder(
            "Missing API key. Make sure to specify an API key.".into(),
        ))?;

        let messages = self
            .messages
            .as_ref()
            .filter(|messages| !messages.is_empty())
            .ok_or(LlmError::Builder(
                "Missing messages. Make sure to add at least one message.".to_string(),
            ))?;

        let provider = self.provider.ok_or(LlmError::Builder(
            "Missing provider. Make sure to specify a provider.".into(),
        ))?;

        let model = self.model.as_ref().ok_or(LlmError::Builder(
            "Missing model. Make sure to specify a model.".into(),
        ))?;

        Ok((messages, provider, model))
    }
}

/// A type-safe builder for constructing LLM requests using the builder pattern.
/// The builder enforces correct construction order through phantom types.
pub struct LlmBuilder<State> {
    fields: BuilderFields,
    _state: PhantomData<State>,
}

impl<State> LlmBuilder<State> {
    /// Transition to a new builder state while preserving all field values
    fn transition_state<NewState>(self) -> LlmBuilder<NewState> {
        LlmBuilder {
            fields: self.fields,
            _state: PhantomData,
        }
    }

    pub(crate) fn get_api_key(&self) -> Option<&str> {
        self.fields.api_key.as_deref()
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
    pub fn api_key(mut self, api_key: ApiKey) -> Result<LlmBuilder<private::ApiKeySet>, LlmError> {
        let key = match api_key {
            ApiKey::Default => {
                let provider = self.fields.provider.ok_or(LlmError::Builder(
                    "Provider must be set before API key".into(),
                ))?;

                env::var(provider.default_api_key_env_var()).map_err(|_| {
                    LlmError::Builder(format!(
                        "Missing {} environment variable",
                        provider.default_api_key_env_var()
                    ))
                })?
            }
            ApiKey::Custom(custom_key) => custom_key,
        };

        self.fields.api_key = Some(key);
        Ok(self.transition_state())
    }
}

impl LlmBuilder<private::ApiKeySet> {
    /// Set the model to use for the LLM request.
    pub fn model(mut self, model_id: &str) -> LlmBuilder<private::Configuring> {
        self.fields.model = Some(model_id.to_string());
        self.transition_state()
    }
}

impl LlmBuilder<private::Configuring> {
    /// Set the messages for the conversation.
    pub fn messages(mut self, messages: Vec<Message>) -> LlmBuilder<private::MessagesSet> {
        self.fields.messages = Some(messages);
        self.transition_state()
    }
}

impl<State: private::Completable> LlmBuilder<State> {
    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.fields.max_tokens = Some(max_tokens);
        self
    }

    /// Set the temperature for generation (0.0 to 2.0).
    /// Lower values make output more focused and deterministic.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.fields.temperature = Some(temperature);
        self
    }

    /// Set the top_p value for nucleus sampling (0.0 to 1.0).
    /// An alternative to temperature; use one or the other, not both.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.fields.top_p = Some(top_p);
        self
    }

    /// Execute the LLM request and return structured output of type T.
    /// The type T must implement Deserialize and JsonSchema for structured output generation as well as
    /// be annotated with `#[schemars(deny_unknown_fields)]`.
    /// Use the `completion_schema` attribute macro to easily define structured output types.
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{completion_schema, llm, Message, ChatRole, ApiKey, Provider};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// #[completion_schema]
    /// struct Analysis {
    ///     sentiment: String,
    ///     confidence: f32,
    /// }
    ///
    /// let analysis = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini")
    ///     .messages(vec![Message {
    ///         role: ChatRole::User,
    ///         content: "Analyze: 'This library is amazing!'".to_string(),
    ///     }])
    ///     .complete::<Analysis>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn complete<T>(self) -> Result<StructuredResponse<T>, LlmError>
    where
        T: for<'a> Deserialize<'a> + Send + schemars::JsonSchema,
    {
        let (messages, provider, model) = self.fields.validate()?;
        let model_string = model.to_string();
        let messages = messages.to_vec();

        match provider {
            Provider::OpenAI => {
                let conversation_messages: Vec<ConversationMessage> = messages
                    .into_iter()
                    .map(ConversationMessage::Chat)
                    .collect();

                let req = StructuredRequest {
                    model: model_string,
                    messages: conversation_messages,
                    tool_config: Some(ToolConfig {
                        tools: self.fields.tools.clone(),
                        tool_choice: self.fields.tool_choice.clone(),
                        parallel_tool_calls: self.fields.parallel_tool_calls,
                    }),
                    generation_config: Some(GenerationConfig {
                        max_tokens: self.fields.max_tokens,
                        temperature: self.fields.temperature,
                        top_p: self.fields.top_p,
                    }),
                };
                let client = openai::create_openai_client_from_builder(&self)?;
                client
                    .generate_structured(req, self.fields.tool_registry.as_ref())
                    .await
            }
            Provider::OpenRouter => {
                let conversation_messages: Vec<ConversationMessage> = messages
                    .into_iter()
                    .map(ConversationMessage::Chat)
                    .collect();

                let req = StructuredRequest {
                    model: model_string,
                    messages: conversation_messages,
                    tool_config: Some(ToolConfig {
                        tools: self.fields.tools.clone(),
                        tool_choice: self.fields.tool_choice.clone(),
                        parallel_tool_calls: self.fields.parallel_tool_calls,
                    }),
                    generation_config: Some(GenerationConfig {
                        max_tokens: self.fields.max_tokens,
                        temperature: self.fields.temperature,
                        top_p: self.fields.top_p,
                    }),
                };
                let client = openrouter::create_openrouter_client_from_builder(&self)?;
                client
                    .generate_structured(req, self.fields.tool_registry.as_ref())
                    .await
            }
        }
    }
}

impl LlmBuilder<private::MessagesSet> {
    /// Set the tools for the LLM request with automatic execution support.
    /// This transitions to the ToolsSet state where tool_choice and parallel_tool_calls can be configured.
    ///
    /// By default parallel tool calling is enabled. This can be changed by calling `parallel_tool_calls` with `false`.
    pub fn tools(mut self, toolset: super::types::ToolSet) -> LlmBuilder<private::ToolsSet> {
        self.fields.tools = Some(toolset.tools().into_boxed_slice());
        self.fields.parallel_tool_calls = Some(true);
        self.fields.tool_registry = Some(toolset.registry);
        self.transition_state()
    }
}

impl LlmBuilder<private::ToolsSet> {
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.fields.tool_choice = Some(choice);
        self
    }

    /// Set whether to enable parallel tool calls.
    /// When true, the model can call multiple tools simultaneously.
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.fields.parallel_tool_calls = Some(enabled);
        self
    }
}

/// Module containing the main entry point for building LLM requests
pub mod llm {
    use super::*;

    /// Create a new LLM builder with the specified provider.
    ///
    /// # Example
    /// ```no_run
    /// # use rsai::{llm, ApiKey, Provider};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let builder = llm::with(Provider::OpenAI)
    ///     .api_key(ApiKey::Default)?
    ///     .model("gpt-4o-mini");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with(provider: Provider) -> LlmBuilder<private::ProviderSet> {
        let mut fields = BuilderFields::new();
        fields.provider = Some(provider);

        LlmBuilder {
            fields,
            _state: PhantomData,
        }
    }
}
