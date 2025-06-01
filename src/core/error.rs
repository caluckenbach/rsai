use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("LLM-Builder error: {0}")]
    Builder(String),

    #[error("Provider configuration error: {0}")]
    ProviderConfiguration(String),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Parse error: {0}")]
    Parse(String),
}
