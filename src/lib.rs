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
mod core;
mod provider;
mod responses;

// Core types
pub use core::{ChatRole, ConversationMessage, Message};
pub use core::{Tool, ToolCall, ToolCallResult, ToolRegistry, ToolSet};
pub use core::{ToolCallingConfig, ToolCallingGuard};

// Configuration types
pub use core::{ApiKey, GenerationConfig, LlmBuilder, ToolChoice, ToolConfig};
pub use responses::HttpClientConfig;

// Response types
pub use core::{LanguageModelUsage, ResponseMetadata, StructuredResponse};

// Async helpers
pub use core::BoxFuture;

// Error handling
pub use core::LlmError;
pub type Result<T> = std::result::Result<T, LlmError>;

// Gen AI request builders
pub use core::llm;

// Gen AI providers
pub use provider::{OpenAiClient, OpenAiConfig, OpenRouterClient, OpenRouterConfig, Provider};

// Traits
pub use core::{LlmProvider, ToolFunction};

// Macros from `rsai-macros`
pub use rsai_macros::{completion_schema, tool, toolset};
