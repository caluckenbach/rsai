use std::marker::PhantomData;

use crate::provider::openai::{self};

use super::{
    error::LlmError,
    traits::LlmProvider,
    types::{Message, StructuredRequest},
};

// TODO: Hide the states
pub struct Init;
pub struct ProviderSet;
pub struct Configuring;
pub struct MessagesSet;

// Is it fine to init this with the unit type?
pub struct LlmBuilder<State, T = ()> {
    provider: Option<String>,
    model: Option<String>,
    messages: Option<Vec<Message>>,
    api_key: Option<String>,
    _state: PhantomData<State>,
    _response: PhantomData<T>,
}

impl<State> LlmBuilder<State> {
    pub(crate) fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub(crate) fn get_messages(&self) -> Option<&Vec<Message>> {
        self.messages.as_ref()
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
                _response: PhantomData,
            }),
            _ => Err(LlmError::Builder("Unsupported Provider".to_string())),
        }
    }
}

impl LlmBuilder<ProviderSet> {
    pub fn model(self, model_id: &str) -> LlmBuilder<Configuring> {
        LlmBuilder {
            provider: self.provider,
            model: Some(model_id.to_string()),
            messages: None,
            api_key: None,
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}

impl<T> LlmBuilder<Configuring, T> {
    pub fn response_model<U>(self) -> LlmBuilder<Configuring, U> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: self.messages,
            api_key: None,
            _state: PhantomData,
            _response: PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LlmBuilder<MessagesSet, T> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: Some(messages),
            api_key: None,
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}

fn validate_builder<T>(
    builder: &LlmBuilder<MessagesSet, T>,
) -> Result<(&Vec<Message>, &str), LlmError> {
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

impl<T> LlmBuilder<MessagesSet, T>
where
    T: serde::de::DeserializeOwned + Send + 'static,
{
    pub async fn send(self) -> Result<T, LlmError> {
        let (messages, provider) = validate_builder(&self).await?;

        match provider {
            "openai" => {
                let req = StructuredRequest {
                    messages: messages.to_vec(),
                };
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
            _response: PhantomData,
        }
    }
}
