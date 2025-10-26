use serde::Deserialize;

use crate::responses::types::FunctionToolCall;

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: String,
    pub model: String,
    pub output: Vec<OutputContent>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OutputContent {
    OutputMessage(OutputMessage),
    FunctionCall(FunctionToolCall),
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
}

// TODO: Remove this, once text input is supported
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct OutputMessage {
    pub id: String,

    #[allow(dead_code)]
    /// This is always `message`
    #[serde(rename = "type")]
    pub r#type: String,

    pub status: Status,

    pub content: Vec<MessageContent>,

    /// This is always `assistant`
    pub role: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum MessageContent {
    OutputText(OutputText),
    Refusal(Refusal),
}

#[derive(Debug, Deserialize)]
pub struct OutputText {
    #[allow(dead_code)]
    /// Always `output_text`
    #[serde(rename = "type")]
    pub r#type: String,

    pub text: String,
    // TODO
    // annotations
}

#[derive(Debug, Deserialize)]
pub struct Refusal {
    /// The refusal explanation from the model.
    pub refusal: String,

    #[allow(dead_code)]
    /// Always `refusal`
    #[serde(rename = "type")]
    pub r#type: String,
}
