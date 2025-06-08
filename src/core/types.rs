use crate::core::traits::ToolFunction;
use crate::provider::Provider;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallResult {
    pub id: String,
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationMessage {
    Chat(Message),
    ToolCall(ToolCall),
    ToolCallResult(ToolCallResult),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredRequest {
    pub model: String,
    pub messages: Vec<ConversationMessage>,
    pub tools: Option<Box<[Tool]>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredResponse<T> {
    pub content: T,
    pub usage: LanguageModelUsage,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageModelUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseMetadata {
    pub provider: Provider,
    pub model: String,
    pub id: String,
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolFunction>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn ToolFunction>) {
        let schema = tool.schema();
        self.tools.insert(schema.name, tool);
    }

    pub fn get_schemas(&self) -> Vec<Tool> {
        self.tools.values().map(|tool| tool.schema()).collect()
    }

    pub async fn execute(
        &self,
        tool_call: &ToolCall,
    ) -> Result<String, crate::core::error::LlmError> {
        if let Some(tool) = self.tools.get(&tool_call.name) {
            let result = tool.execute(tool_call.arguments.clone()).await?;
            // If result is a JSON string, extract the actual string value
            // Otherwise, serialize the result as JSON
            match result {
                serde_json::Value::String(s) => Ok(s),
                other => Ok(other.to_string()),
            }
        } else {
            Err(crate::core::error::LlmError::ToolNotFound(
                tool_call.name.clone(),
            ))
        }
    }
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct ToolSet {
    pub registry: ToolRegistry,
}

impl ToolSet {
    pub fn tools(&self) -> Vec<Tool> {
        self.registry.get_schemas()
    }
}
