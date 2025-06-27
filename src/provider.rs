mod constants;
pub mod openai;

#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    OpenAI,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
        }
    }
}
