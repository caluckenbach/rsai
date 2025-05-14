use std::marker::PhantomData;

use crate::provider::openai;

use super::{
    error::LlmError,
    traits::ChatCompletion,
    types::{ChatCompletionRequest, Message},
};

// TODO: Hide the states
pub struct Init;
pub struct ProviderSet;
pub struct Configuring;
pub struct MessagesSet;

pub struct LlmBuilder<State, T = String> {
    provider: Option<String>,
    model: Option<String>,
    messages: Option<Vec<Message>>,
    _state: PhantomData<State>,
    _response: PhantomData<T>,
}

impl LlmBuilder<Init> {
    pub fn provider(self, provider: &str) -> Result<LlmBuilder<ProviderSet>, LlmError> {
        match provider {
            "openai" => Ok(LlmBuilder {
                provider: Some(provider.to_string()),
                model: None,
                messages: None,
                _state: PhantomData,
                _response: PhantomData,
            }),
            _ => Err(LlmError::BuilderError("Unsupported Provider".to_string())),
        }
    }
}

impl LlmBuilder<ProviderSet> {
    pub fn model(self, model_id: &str) -> LlmBuilder<Configuring> {
        LlmBuilder {
            provider: self.provider,
            model: Some(model_id.to_string()),
            messages: None,
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
            _state: PhantomData,
            _response: PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LlmBuilder<MessagesSet, T> {
        LlmBuilder {
            provider: self.provider,
            model: self.model,
            messages: Some(messages),
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}

impl<T> LlmBuilder<MessagesSet, T>
where
    T: serde::de::DeserializeOwned,
{
    pub async fn send(self) -> Result<T, LlmError> {
        let request = ChatCompletionRequest {
            messages: self.messages.ok_or(LlmError::BuilderError(
                "Missing messages. Make sure to define at least one message.".into(),
            ))?,
        };
        let provider = self.provider.ok_or(LlmError::BuilderError(
            "Missing provider. Make sure to specify a provider.".into(),
        ))?;
        match provider.as_str() {
            "openai" => {
                let prov = openai::create_provider_from_builder(&self)?;
                let response = prov.complete(&request).await?;
                Ok(response)
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
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}
