use std::sync::Arc;

use crate::core::{
    builder::{LlmBuilder, MessagesSet},
    error::LlmError,
    traits::ChatCompletion,
    types::{ChatCompletionRequest, ChatCompletionResponse},
};

pub struct OpenAiProvider<
    C: async_openai::config::Config + Default = async_openai::config::OpenAIConfig,
> {
    client: Arc<async_openai::Client<C>>,
}

impl OpenAiProvider<C> {
    pub fn new() -> Self {
        OpenAiProvider {
            client: async_openai::Client::new(),
        }
    }
}

impl ChatCompletion for OpenAiProvider {
    async fn complete<T>(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
    }
}

pub fn create_provider_from_builder<T>(
    builder: &LlmBuilder<MessagesSet, T>,
) -> Result<OpenAiProvider, LlmError> {
    Ok(OpenAiProvider::new())
}
