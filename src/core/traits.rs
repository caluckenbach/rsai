use async_trait::async_trait;
use serde_json::Value;

use super::{
    error::LlmError,
    types::{BoxFuture, StructuredRequest, StructuredResponse, Tool, ToolRegistry},
};

#[async_trait]
pub trait LlmProvider {
    async fn generate_structured<T>(
        &self,
        request: StructuredRequest,
        tool_registry: Option<&ToolRegistry>,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema;
}

pub trait ToolFunction: Send + Sync {
    fn schema(&self) -> Tool;
    fn execute<'a>(&'a self, params: Value) -> BoxFuture<'a, Result<Value, LlmError>>;
}
