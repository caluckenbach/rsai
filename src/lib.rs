//! # rsai
//!
//! Predictable development for unpredictable models. Let the compiler handle the chaos.
//!
//! ## ⚠️ WARNING
//!
//! This is a pre-release version with an unstable API. Breaking changes may occur between versions.
//! Use with caution and pin to specific versions in production applications.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rsai::{llm, Message, ChatRole, ApiKey, Provider, completion_schema};
//!
//! #[completion_schema]
//! struct Analysis {
//!     sentiment: String,
//!     confidence: f32,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let analysis = llm::with(Provider::OpenAI)
//!     .api_key(ApiKey::Default)?
//!     .model("gpt-4o-mini")
//!     .messages(vec![Message {
//!         role: ChatRole::User,
//!         content: "Analyze: 'This library is amazing!'".to_string(),
//!     }])
//!     .complete::<Analysis>()
//!     .await?;
//! Ok(())
//! }
//! ```
//!
//! ## Known Issues
//!
//! Currently, the `tool_choice` parameter in the builder pattern is not functional due to an issue in the macro implementation.
//! This will be fixed in a future release.

pub mod core;
pub mod provider;
pub mod responses;

pub use core::builder::ApiKey;
pub use core::{
    builder::llm,
    types::{ChatRole, Message},
};
pub use provider::Provider;
pub use rsai_macros::completion_schema;
