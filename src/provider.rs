mod constants;
pub mod openai;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
        }
    }
}

impl Provider {
    /// Parse a provider from a string
    pub fn from_str(s: &str) -> Result<Self, crate::core::error::LlmError> {
        match s {
            "openai" => Ok(Provider::OpenAI),
            _ => Err(crate::core::error::LlmError::Builder(format!(
                "Unsupported provider: {}",
                s
            ))),
        }
    }

    /// Get the default environment variable name for this provider's API key
    pub fn default_api_key_env_var(&self) -> &'static str {
        match self {
            Provider::OpenAI => constants::openai::API_KEY_ENV_VAR,
        }
    }
}
