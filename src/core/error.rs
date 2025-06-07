use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM-Builder error: {0}")]
    Builder(String),

    #[error("Provider configuration error: {0}")]
    ProviderConfiguration(String),

    #[error("Provider error: {message}")]
    Provider {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Network error: {message}")]
    Network {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("API error: {message}")]
    Api {
        message: String,
        status_code: Option<u16>,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Parse error: {message}")]
    Parse {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Tool execution error: {message}")]
    ToolExecution {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
}
