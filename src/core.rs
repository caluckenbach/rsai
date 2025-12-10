mod builder;
mod error;
pub mod http;
mod tool_guard;
mod traits;
mod types;

pub use builder::{ApiKey, LlmBuilder, llm};

pub use error::LlmError;
pub use http::{HttpClient, HttpClientConfig};
pub use tool_guard::{ToolCallingConfig, ToolCallingGuard};
pub use traits::{CompletionTarget, LlmProvider, ToolFunction};

pub use types::StructuredRequest;
pub use types::{
    BoxFuture, ChatRole, ConversationMessage, Ctx, FunctionCallData, GenerationConfig,
    LanguageModelUsage, Message, ProviderResponse, ResponseContent, ResponseMetadata,
    StructuredResponse, TextResponse, Tool, ToolCall, ToolCallResult, ToolChoice, ToolConfig,
    ToolRegistry, ToolSet, ToolSetBuilder,
};
