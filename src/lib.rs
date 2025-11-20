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
//! use rsai::{ApiKey, Provider};
//! use rsai::text::{llm, Message, ChatRole, completion_schema};
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

// The core module currently implements text-based LLM functionality.
// We re-export it as `text` to align with future modules (image, audio, video).
// Actual file system refactoring (renaming core/ -> text/) is deferred until
// those modalities are added to minimize churn.
pub mod text {
    // Core types
    pub use crate::core::{ChatRole, ConversationMessage, Message};
    pub use crate::core::{Tool, ToolCall, ToolCallResult, ToolRegistry, ToolSet};
    pub use crate::core::{ToolCallingConfig, ToolCallingGuard};

    // Configuration types
    pub use crate::core::{GenerationConfig, LlmBuilder, ToolChoice, ToolConfig};

    // Response types
    pub use crate::core::{
        LanguageModelUsage, ResponseMetadata, StructuredRequest, StructuredResponse,
    };

    // Async helpers
    pub use crate::core::BoxFuture;

    // Error handling
    pub use crate::core::LlmError;
    pub type Result<T> = std::result::Result<T, LlmError>;

    // Gen AI request builders
    pub use crate::core::llm;

    // Traits
    pub use crate::core::{LlmProvider, ToolFunction};

    // Macros
    pub use rsai_macros::{completion_schema, tool, toolset};
}

// Shared Configuration types
pub use core::ApiKey;
pub use responses::HttpClientConfig;

// Shared Providers
pub use provider::{OpenAiClient, OpenAiConfig, OpenRouterClient, OpenRouterConfig, Provider};
