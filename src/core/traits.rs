use async_trait::async_trait;

use super::{
    error::LlmError,
    types::{StructuredRequest, StructuredResponse},
};

#[async_trait]
pub trait LlmProvider {
    async fn generate_structured<T>(
        &self,
        request: StructuredRequest,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema;
}
