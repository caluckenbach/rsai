#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionRequest {
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatCompletionResponse {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter(String),
    ToolCalls,
    Error,
    Other,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageModelUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}
