use std::error::Error as StdError;
use std::fmt;

/// Represents errors specifically related to AI/LLM operations
#[derive(Debug)]
pub enum AIError {
    /// Request error
    RequestError(String),
    /// Response parsing error
    ParseError(String),
    /// API returned an error
    ApiError(String),
    /// Rate limit exceeded
    RateLimited(String),
    /// Authentication error
    AuthError(String),
}

impl StdError for AIError {}

impl fmt::Display for AIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AIError::RequestError(e) => write!(f, "Request error: {}", e),
            AIError::ParseError(e) => write!(f, "Parse error: {}", e),
            AIError::ApiError(e) => write!(f, "API error: {}", e),
            AIError::RateLimited(e) => write!(f, "Rate limited: {}", e),
            AIError::AuthError(e) => write!(f, "Authentication error: {}", e),
        }
    }
}