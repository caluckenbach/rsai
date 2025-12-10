use async_trait::async_trait;

use crate::responses::request::Format;

use super::{
    error::LlmError,
    types::{BoxFuture, ProviderResponse, StructuredRequest, Tool, ToolRegistry},
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

pub trait ToolFunction<Ctx = ()>: Send + Sync {
    fn schema(&self) -> Tool;
    fn execute<'a>(
        &'a self,
        ctx: &'a Ctx,
        params: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>>;
}

pub trait CompletionTarget: Sized + Send {
    type Output;

    fn format() -> Result<Format, LlmError>;

    /// Parse a provider-agnostic response into the target output type.
    fn parse_response(res: ProviderResponse) -> Result<Self::Output, LlmError>;

    fn supports_tools() -> bool {
        true
    }
}
