use std::{env, marker::PhantomData};

use serde::de::Deserialize;

use crate::provider::openai::{self};

use super::{
    error::LlmError,
    traits::LlmProvider,
    types::{Message, StructuredRequest},
};

mod private {
    pub struct Init;
    pub struct ProviderSet;
    pub struct ApiKeySet;
    pub struct Configuring;
    pub struct MessagesSet;
}

/// A type-safe builder for constructing LLM requests using the builder pattern.
/// The builder enforces correct construction order through phantom types.
pub struct LlmBuilder<State> {
    provider: Option<String>,
    model: Option<String>,
    messages: Option<Vec<Message>>,
    api_key: Option<String>,
    _state: PhantomData<State>,
}

impl<State> LlmBuilder<State> {
    pub(crate) fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub(crate) fn get_messages(&self) -> Option<&Vec<Message>> {
        self.messages.as_ref()
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
            _state: PhantomData,
        }
    }
}

fn validate_builder(
    builder: &LlmBuilder<private::MessagesSet>,
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

impl LlmBuilder<private::MessagesSet> {
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
    pub async fn complete<T>(mut self) -> Result<T, LlmError>
    where
        T: for<'a> Deserialize<'a> + Send + schemars::JsonSchema,
    {
        let (messages, provider, model) = validate_builder(&self)?;
        // Yes.. I am also throwing up.. let me fix that later.
        let model_string = model.to_string();
        // Cloning is fine, since we don't want to modify the original messages.
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
                };
                let client = openai::create_openai_client_from_builder(&self)?;
                let res = client.generate_structured::<T>(req).await?;
                Ok(res.content)
            }
            _ => todo!(),
        }
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
            _state: PhantomData,
        }
    }
}
