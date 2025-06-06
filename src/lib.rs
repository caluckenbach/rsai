pub mod core;
pub mod error;
pub mod provider;

pub use ai_macros::completion_schema;
pub use core::{builder::llm, types::{Message, ChatRole}};
pub use core::builder::ApiKey;
pub use error::AIError;
