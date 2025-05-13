use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("LLM-Builder error: {0}")]
    BuilderError(String),
}
