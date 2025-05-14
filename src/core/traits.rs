use async_trait::async_trait;

use super::{
    error::LlmError,
    types::{ChatCompletionRequest, ChatCompletionResponse},
};

#[async_trait]
pub trait ChatCompletion: Send + Sync {
    async fn complete(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;
}
