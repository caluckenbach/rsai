mod constants;
pub(crate) mod openai;
pub(crate) mod openrouter;

pub use openai::{OpenAiClient, OpenAiConfig};
pub use openrouter::{OpenRouterClient, OpenRouterConfig};

use std::time::Duration;

/// Configuration for tool calling behavior and limits
#[derive(Debug, Clone)]
pub struct ToolCallingConfig {
    /// Maximum number of iterations in tool calling loop (default: 50)
    pub max_iterations: u32,
    /// Timeout for tool calling loop (default: 5 minutes)
    pub timeout: Duration,
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    OpenRouter,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
            Provider::OpenRouter => write!(f, "OpenRouter"),
        }
    }
}

impl Provider {
    /// Get the default environment variable name for this provider's API key
    pub fn default_api_key_env_var(&self) -> &'static str {
        match self {
            Provider::OpenAI => constants::openai::API_KEY_ENV_VAR,
            Provider::OpenRouter => constants::openrouter::API_KEY_ENV_VAR,
        }
    }
}
