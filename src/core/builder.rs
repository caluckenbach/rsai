use std::{env, marker::PhantomData};

use serde::de::Deserialize;

use crate::provider::openai::{self};

use super::{
    error::LlmError,
    traits::LlmProvider,
    types::{Message, StructuredRequest},
};

// TODO: Hide the states
pub struct Init;
pub struct ProviderSet;
pub struct ApiKeySet;
pub struct Configuring;
pub struct MessagesSet;

// Is it fine to init this with the unit type?
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

impl LlmBuilder<Init> {
    pub fn provider(self, provider: &str) -> Result<LlmBuilder<ProviderSet>, LlmError> {
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

pub enum ApiKey {
    Default,
    Custom(String),
}

impl LlmBuilder<ProviderSet> {
    pub fn api_key(self, api_key: ApiKey) -> Result<LlmBuilder<ApiKeySet>, LlmError> {
        let key = match api_key {
            ApiKey::Default => {
                match self.provider.as_deref() {
                    Some("openai") => env::var("OPENAI_API_KEY")
                        .map_err(|_| LlmError::Builder("Missing OPENAI_API_KEY environment variable".to_string()))?,
                    _ => return Err(LlmError::Builder("Can't load API Key for unsupported Provider".to_string())),
                }
            }
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

impl LlmBuilder<ApiKeySet> {
    pub fn model(self, model_id: &str) -> LlmBuilder<Configuring> {
        LlmBuilder {
            provider: self.provider,
            model: Some(model_id.to_string()),
            messages: None,
            api_key: self.api_key,
            _state: PhantomData,
        }
    }
}

impl LlmBuilder<Configuring> {
    pub fn messages(self, messages: Vec<Message>) -> LlmBuilder<MessagesSet> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: Some(messages),
            api_key: self.api_key,
            _state: PhantomData,
        }
    }
}

fn validate_builder(builder: &LlmBuilder<MessagesSet>) -> Result<(&Vec<Message>, &str), LlmError> {
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

    Ok((messages, provider))
}

impl LlmBuilder<MessagesSet> {
    pub async fn complete<T>(mut self) -> Result<T, LlmError>
    where
        // TODO: Understand why this is necesary here.
        T: for<'a> Deserialize<'a> + Send,
    {
        let (messages, provider) = validate_builder(&self)?;
        // Cloning is fine, since we don't want to modify the original messages.
        let messages = messages.to_vec();

        match provider {
            "openai" => {
                if self.api_key.is_none() {
                    let api_key = env::var("OPENAI_API_KEY")
                        .map_err(|_| LlmError::Builder("Missing OPENAI_API_KEY.".to_string()))?;
                    self.set_api_key(&api_key);
                }

                let req = StructuredRequest { messages };
                let client = openai::create_openai_client_from_builder(&self)?;
                let res = client.generate_structured::<T>(req).await?;
                Ok(res.content)
            }
            _ => todo!(),
        }
    }
}

pub mod llm {
    use super::*;

    pub fn call() -> LlmBuilder<Init> {
        LlmBuilder {
            provider: None,
            model: None,
            messages: None,
            api_key: None,
            _state: PhantomData,
        }
    }
}
