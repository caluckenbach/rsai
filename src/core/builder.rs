use std::marker::PhantomData;

use super::{error::LlmError, types::Message};

// TODO: Hide the states
pub struct Init;
pub struct ProviderSet;
pub struct Configuring;
pub struct MessagesSet;

pub struct LlmBuilder<State, T = String> {
    provider: Option<String>,
    model_id: Option<String>,
    system_prompt: Option<String>,
    messages: Option<Vec<Message>>,
    _state: PhantomData<State>,
    _response: PhantomData<T>,
}

impl LlmBuilder<Init> {
    pub fn provider(self, provider: &str) -> Result<LlmBuilder<ProviderSet>, LlmError> {
        match provider {
            "openai" => Ok(LlmBuilder {
                provider: Some(provider.to_string()),
                model_id: None,
                system_prompt: None,
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
            model_id: Some(model_id.to_string()),
            system_prompt: None,
            messages: None,
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}

impl<T> LlmBuilder<Configuring, T> {
    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = Some(system_prompt.to_string());
        self
    }

    pub fn response_model<U>(self) -> LlmBuilder<Configuring, U> {
        LlmBuilder {
            provider: self.provider,
            model_id: self.model_id,
            system_prompt: self.system_prompt,
            messages: self.messages,
            _state: PhantomData,
            _response: PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LlmBuilder<MessagesSet, T> {
        LlmBuilder {
            provider: self.provider,
            model_id: self.model_id,
            system_prompt: self.system_prompt,
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
        todo!()
    }
}

pub mod llm {
    use super::*;

    pub fn call() -> LlmBuilder<Init> {
        LlmBuilder {
            provider: None,
            model_id: None,
            system_prompt: None,
            messages: None,
            _state: PhantomData,
            _response: PhantomData,
        }
    }
}
