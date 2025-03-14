use thiserror::Error;

#[derive(Error, Debug)]
pub enum AIError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("the request failed due to: {0}")]
    RequestError(String),

    // TODO: This needs to have a more fitting name
    #[error("conversion failed due to: {0}")]
    ConversionError(String),

    #[error("this functionality isn't supported by the current provider: {0}")]
    UnsupportedFunctionality(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}
