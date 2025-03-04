pub mod gemini;

use crate::core::TextStream;
use crate::core::llm::GenerationOptions;
use crate::error::AIError;
use async_trait::async_trait;

#[async_trait]
pub trait Provider {
    async fn generate_text(&self, prompt: &str, options: &GenerationOptions)
    -> Result<String, AIError>;
    async fn stream_text(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<TextStream, AIError>;
}
