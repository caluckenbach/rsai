pub mod gemini;

use crate::error::AIError;
use crate::model::{ChatSettings, TextCompletion};
use async_trait::async_trait;

#[async_trait]
pub trait Provider {
    async fn generate_text(
        &self,
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<TextCompletion, AIError>;
}
