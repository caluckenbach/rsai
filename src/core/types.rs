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
pub struct StructuredRequest {
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredResponse<T> {
    pub content: T,
    pub usage: LanguageModelUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LanguageModelUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

