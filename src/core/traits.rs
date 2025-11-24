use async_trait::async_trait;

use crate::{
    Provider,
    responses::{request::Format, response::Response},
};

use super::{
    error::LlmError,
    types::{BoxFuture, StructuredRequest, Tool, ToolRegistry},
};

#[async_trait]
pub trait LlmProvider {
    async fn generate_completion<T>(
        &self,
        request: StructuredRequest,
        format: Format,
        tool_registry: Option<&ToolRegistry>,
    ) -> Result<T::Output, LlmError>
    where
        T: CompletionTarget + Send;
}

pub trait ToolFunction: Send + Sync {
    fn schema(&self) -> Tool;
    fn execute<'a>(
        &'a self,
        params: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>>;
}

pub trait CompletionTarget: Sized + Send {
    type Output;

    fn format() -> Result<Format, LlmError>;

    fn parse_response(res: Response, provider: Provider) -> Result<Self::Output, LlmError>;

    fn supports_tools() -> bool {
        true
    }
}
