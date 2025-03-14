pub mod gemini;

use crate::error::AIError;
use crate::model::chat::{StructuredOutput, StructuredOutputParameters, TextStream};
use crate::model::{ChatSettings, TextCompletion};
use async_trait::async_trait;
use futures::Stream;
use serde::de::DeserializeOwned;

#[async_trait]
pub trait Provider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError>;

    async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
        settings: &'a ChatSettings,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>> + 'a, AIError>;

    async fn generate_object<T: DeserializeOwned>(
        &self,
        prompt: &str,
        settings: &ChatSettings,
        parameters: &StructuredOutputParameters<T>,
    ) -> Result<StructuredOutput<T>, AIError>;
}
