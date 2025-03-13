pub mod gemini;

use crate::error::AIError;
use crate::model::chat::TextStream;
use crate::model::{ChatSettings, TextCompletion};
use async_trait::async_trait;
use futures::Stream;

#[async_trait]
pub trait Provider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError>;

    async fn stream_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>>, AIError>;
}
