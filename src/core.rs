pub mod llm;

#[derive(Debug)]
pub enum AIError {
    RequestError(String),
}

pub struct ChatSettings {}

pub struct TextStream {}
