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
pub mod core;
pub mod provider;
pub mod responses;

// Core types
pub use core::types::{ChatRole, ConversationMessage, Message};
pub use core::types::{Tool, ToolCall, ToolCallResult, ToolRegistry, ToolSet};

// Configuration types
pub use core::builder::ApiKey;
pub use core::types::{GenerationConfig, ToolChoice, ToolConfig};

// Response types
pub use core::types::{LanguageModelUsage, ResponseMetadata, StructuredResponse};

// Error handling
pub use core::error::LlmError;
pub type Result<T> = std::result::Result<T, LlmError>;

// Gen AI request builders
pub use core::builder::llm;

// Gen AI providers
pub use provider::Provider;

// Traits
pub use core::traits::{LlmProvider, ToolFunction};

// Macros from `rsai-macros`
pub use rsai_macros::{completion_schema, tool, toolset};
