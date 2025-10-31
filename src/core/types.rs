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
    pub content: serde_json::Value,
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
    pub tool_config: Option<ToolConfig>,
    pub generation_config: Option<GenerationConfig>,
}

/// Configuration for tool calling behavior
#[derive(Debug, Clone, PartialEq)]
pub struct ToolConfig {
    /// Available tools for the model to call
    pub tools: Option<Box<[Tool]>>,
    /// Strategy for choosing which tools to call
    pub tool_choice: Option<ToolChoice>,
    /// Whether to allow parallel tool calls (default: true)
    pub parallel_tool_calls: Option<bool>,
}

/// Configuration for text generation parameters
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter (0.0 to 1.0)
    pub top_p: Option<f32>,
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
    ) -> Result<serde_json::Value, crate::core::error::LlmError> {
        if let Some(tool) = self.tools.get(&tool_call.name) {
            tool.execute(tool_call.arguments.clone()).await
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

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use std::sync::Arc;

    struct ObjectTool;
    #[async_trait]
    impl ToolFunction for ObjectTool {
        fn schema(&self) -> Tool {
            Tool {
                name: "object_tool".to_string(),
                description: Some("Returns an object".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                strict: Some(true),
            }
        }

        fn execute<'a>(
            &'a self,
            _params: serde_json::Value,
        ) -> BoxFuture<'a, Result<serde_json::Value, crate::core::error::LlmError>> {
            Box::pin(async move {
                Ok(serde_json::json!({
                    "name": "test",
                    "value":42,
                    "active": true
                }))
            })
        }
    }

    #[tokio::test]
    async fn test_tool_registry_preservers_object_types() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ObjectTool));

        let tool_call = ToolCall {
            id: "test_Id".to_string(),
            call_id: "call_123".to_string(),
            name: "object_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let result = registry.execute(&tool_call).await.unwrap();

        // Verify the result is still structured data an not just a string
        assert!(result.is_object());
        assert_eq!(result["name"], "test");
        assert_eq!(result["value"], 42);
        assert_eq!(result["active"], true);
    }
}
