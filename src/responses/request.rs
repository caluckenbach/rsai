use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{core::types, responses::common::FunctionToolCall};

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    pub input: Vec<InputItem>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of total calls to built-in tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    pub store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    pub text: Format,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Box<[Tool]>>,

    /// An integer between 0 and 20
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Alter this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    pub truncation: Option<bool>,

    /// Used to boost cache hit rates by better bucketing similar requests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum InputItem {
    Message(InputMessage),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionToolCallOutput),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function { name: String },
}

impl From<types::ToolChoice> for ToolChoice {
    fn from(value: types::ToolChoice) -> Self {
        match value {
            types::ToolChoice::None => ToolChoice::None,
            types::ToolChoice::Auto => ToolChoice::Auto,
            types::ToolChoice::Required => ToolChoice::Required,
            types::ToolChoice::Function { name } => ToolChoice::Function { name },
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct Format {
    pub format: FormatType,
}

#[derive(Debug, Serialize, Clone)]
pub struct InputMessage {
    pub role: InputMessageRole,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionToolCallOutput {
    pub call_id: String,
    pub output: serde_json::Value,
    #[serde(rename = "type")]
    pub r#type: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum FormatType {
    // TODO: remove this, once text input is supported
    #[allow(dead_code)]
    Text {
        #[serde(rename = "type")]
        r#type: TextType,
    },
    JsonSchema(JsonSchema),
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum TextType {
    // TODO: Remove this, once text input is supported
    #[allow(dead_code)]
    Text,
}

#[derive(Debug, Serialize)]
pub struct JsonSchema {
    pub name: String,

    pub schema: serde_json::Value,

    #[serde(rename = "type")]
    pub r#type: JsonSchemaType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonSchemaType {
    JsonSchema,
}
