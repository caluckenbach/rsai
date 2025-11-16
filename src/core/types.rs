use crate::core::{LlmError, traits::ToolFunction};
use crate::provider::Provider;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

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
    tools: Arc<RwLock<HashMap<String, Arc<dyn ToolFunction>>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers a new tool in the registry.
    ///
    /// # Arguments
    /// * `tool` - The tool to register, wrapped in an Arc
    ///
    /// # Returns
    /// * `Ok(())` if the tool was successfully registered
    /// * `Err(LlmError::ToolRegistration)` if a tool with the same name already exists
    ///
    /// # Errors
    /// This function will return an error if:
    /// - A tool with the same name is already registered
    /// - The registry's write lock is poisoned (indicates a panic in another thread)
    ///
    /// # Examples
    /// ```ignore
    /// use std::sync::Arc;
    /// use rsai::ToolRegistry;
    ///
    /// // Assuming you have a tool implementation
    /// let registry = ToolRegistry::new();
    /// // let tool: Arc<dyn rsai::ToolFunction> = Arc::new(your_tool);
    /// // registry.register(tool)?;
    /// # Ok::<(), rsai::LlmError>(())
    /// ```
    ///
    /// # Thread Safety
    /// This method is thread-safe. Multiple threads can register tools concurrently,
    /// but attempting to register the same tool name from multiple threads will
    /// result in only one success and the rest will return errors.
    pub fn register(&self, tool: Arc<dyn ToolFunction>) -> Result<(), LlmError> {
        let schema = tool.schema();
        let schema_name = schema.name.clone();

        let mut w_tools = self.tools.write().map_err(|_| LlmError::ToolRegistration {
            tool_name: schema_name.clone(),
            message: "Failed to acquire write lock on tool registry".to_string(),
        })?;

        if w_tools.contains_key(&schema.name) {
            return Err(LlmError::ToolRegistration {
                tool_name: schema.name.clone(),
                message: format!("Tool {} already registered", schema.name),
            });
        }

        w_tools.insert(schema.name, tool);
        Ok(())
    }

    pub fn overwrite(&self, tool: Arc<dyn ToolFunction>) -> Result<(), LlmError> {
        let schema = tool.schema();
        let schema_name = schema.name.clone();

        let mut w_tools = self.tools.write().map_err(|_| LlmError::ToolRegistration {
            tool_name: schema_name.clone(),
            message: "Failed to acquire write lock on tool registry".to_string(),
        })?;

        let overwritten_tool = w_tools.insert(schema.name, tool);

        if overwritten_tool.is_some() {
            // TODO: Log warning
        }

        Ok(())
    }

    pub fn get_schemas(&self) -> Result<Vec<Tool>, LlmError> {
        let r_tools = self
            .tools
            .read()
            .map_err(|_| LlmError::ToolRegistryAccess {
                message: "Failed to acquire read lock (lock poisoned)".to_string(),
            })?;
        let schema = r_tools.values().map(|tool| tool.schema()).collect();
        Ok(schema)
    }

    pub async fn execute(&self, tool_call: &ToolCall) -> Result<serde_json::Value, LlmError> {
        let tool = {
            let r_tools = self
                .tools
                .read()
                .map_err(|_| LlmError::ToolRegistryAccess {
                    message: "Failed to acquire read lock (lock poisoned)".to_string(),
                })?;
            r_tools.get(&tool_call.name).cloned()
        };

        if let Some(tool) = tool {
            tool.execute(tool_call.arguments.clone()).await
        } else {
            Err(LlmError::ToolNotFound(tool_call.name.clone()))
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct ToolSet {
    pub registry: ToolRegistry,
}

impl ToolSet {
    pub fn tools(&self) -> Result<Vec<Tool>, LlmError> {
        self.registry.get_schemas()
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
        ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>> {
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
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(ObjectTool))
            .expect("Failed to regiter object_tool");

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
