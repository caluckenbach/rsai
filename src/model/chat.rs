use reqwest::header;
use std::any::Any;
use std::fmt::Debug;

use crate::AIError;
use crate::provider::Provider;

pub struct ChatModel<P: Provider> {
    provider: P,
    settings: ChatSettings,
}

impl<P: Provider> ChatModel<P> {
    pub fn new(provider: P, settings: ChatSettings) -> Self {
        Self { provider, settings }
    }

    pub async fn generate_text(&self, prompt: &str) -> Result<TextCompletion, AIError> {
        self.provider.generate_text(prompt, &self.settings).await
    }
}

#[derive(Debug, Clone)]
pub struct ChatSettings {
    pub system_prompt: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<i32>,
    pub max_retries: Option<i32>,
    // TODO: add abort signal
    // TODO: hide this implementation detail
    pub headers: reqwest::header::HeaderMap,
    pub provider_options: Option<Box<dyn ProviderOptions>>,
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            system_prompt: None,
            messages: None,
            max_tokens: None,
            temperature: None,
            stop_sequences: None,
            seed: None,
            max_retries: None,
            headers: reqwest::header::HeaderMap::new(),
            provider_options: None,
        }
    }
}

impl ChatSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn add_message(mut self, message: ChatMessage) -> Self {
        let messages = self.messages.get_or_insert_with(Vec::new);
        messages.push(message);
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn add_stop_sequence(mut self, stop_sequence: impl Into<String>) -> Self {
        let sequences = self.stop_sequences.get_or_insert_with(Vec::new);
        sequences.push(stop_sequence.into());
        self
    }

    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_retries(mut self, max_retries: i32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    pub fn add_header(mut self, key: &str, value: &str) -> Self {
        if let (Ok(key), Ok(value)) = (key.parse::<header::HeaderName>(), value.parse()) {
            self.headers.insert(key, value);
        }
        self
    }

    pub fn provider_options(mut self, options: Box<dyn ProviderOptions>) -> Self {
        self.provider_options = Some(options);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

pub trait ProviderOptions: Debug + Send + Sync + Any {
    fn clone_box(&self) -> Box<dyn ProviderOptions>;

    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn ProviderOptions> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextCompletion {
    pub text: String,
    pub reasoning_text: Option<String>,
    //pub reasoning:
    //pub sources:
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
    // warnings
    // steps
    //pub request: String,
    //pub response: String,
}

#[derive(Debug)]
pub struct TextStream {
    pub text: String,
    pub reasoning_text: Option<String>,
    pub finish_reason: FinishReason,
    pub usage: Option<LanguageModelUsage>,
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

pub fn calculate_language_model_usage(
    prompt_tokens: i32,
    completion_tokens: i32,
) -> LanguageModelUsage {
    LanguageModelUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;

    struct MockProvider {
        last_prompt: Arc<Mutex<Option<String>>>,
        last_settings: Arc<Mutex<Option<ChatSettings>>>,
        response: TextCompletion,
    }

    impl MockProvider {
        fn new(response: TextCompletion) -> Self {
            Self {
                last_prompt: Arc::new(Mutex::new(None)),
                last_settings: Arc::new(Mutex::new(None)),
                response,
            }
        }

        fn with_error() -> Self {
            Self {
                last_prompt: Arc::new(Mutex::new(None)),
                last_settings: Arc::new(Mutex::new(None)),
                response: TextCompletion {
                    text: String::new(),
                    reasoning_text: None,
                    finish_reason: FinishReason::Error,
                    usage: LanguageModelUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                },
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn generate_text(
            &self,
            prompt: &str,
            settings: &ChatSettings,
        ) -> Result<TextCompletion, AIError> {
            *self.last_prompt.lock().unwrap() = Some(prompt.to_string());
            *self.last_settings.lock().unwrap() = Some(settings.clone());

            if matches!(self.response.finish_reason, FinishReason::Error) {
                return Err(AIError::ApiError("Mock error".to_string()));
            }

            Ok(TextCompletion {
                text: self.response.text.clone(),
                reasoning_text: self.response.reasoning_text.clone(),
                finish_reason: self.response.finish_reason.clone(),
                usage: LanguageModelUsage {
                    prompt_tokens: self.response.usage.prompt_tokens,
                    completion_tokens: self.response.usage.completion_tokens,
                    total_tokens: self.response.usage.total_tokens,
                },
            })
        }
    }

    #[test]
    fn test_chat_settings_default() {
        let settings = ChatSettings::default();

        assert_eq!(settings.system_prompt, None);
        assert_eq!(settings.messages, None);
        assert_eq!(settings.max_tokens, None);
        assert_eq!(settings.temperature, None);
        assert_eq!(settings.stop_sequences, None);
        assert_eq!(settings.seed, None);
        assert_eq!(settings.max_retries, None);
        assert!(settings.headers.is_empty());
        assert!(settings.provider_options.is_none());
    }

    #[test]
    fn test_chat_settings_builder() {
        let settings = ChatSettings::new()
            .system_prompt("Hello, AI!")
            .max_tokens(100)
            .temperature(0.7)
            .seed(42)
            .max_retries(3)
            .add_stop_sequence("END")
            .add_stop_sequence("STOP")
            .add_header("X-Test", "test-value");

        assert_eq!(settings.system_prompt, Some("Hello, AI!".to_string()));
        assert_eq!(settings.max_tokens, Some(100));
        assert_eq!(settings.temperature, Some(0.7));
        assert_eq!(settings.seed, Some(42));
        assert_eq!(settings.max_retries, Some(3));
        assert_eq!(
            settings.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );

        // Header check is limited because HeaderMap doesn't implement Eq
        assert!(!settings.headers.is_empty());
    }

    #[test]
    fn test_add_message() {
        let message = ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        };

        let settings = ChatSettings::new().add_message(message);

        assert!(settings.messages.is_some());
        let messages = settings.messages.unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0].role, ChatRole::User));
        assert_eq!(messages[0].content, "Hello");
    }

    #[test]
    fn test_messages() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "System prompt".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "User message".to_string(),
            },
        ];

        let settings = ChatSettings::new().messages(messages.clone());

        assert!(settings.messages.is_some());
        let stored_messages = settings.messages.unwrap();
        assert_eq!(stored_messages.len(), 2);
        assert!(matches!(stored_messages[0].role, ChatRole::System));
        assert!(matches!(stored_messages[1].role, ChatRole::User));
    }

    #[tokio::test]
    async fn test_chat_model_generate_text() {
        let expected_response = TextCompletion {
            text: "Hello, human!".to_string(),
            reasoning_text: None,
            finish_reason: FinishReason::Stop,
            usage: LanguageModelUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };

        let provider = MockProvider::new(expected_response);
        let prompt_tracker = provider.last_prompt.clone();
        let settings_tracker = provider.last_settings.clone();

        let chat_settings = ChatSettings::new()
            .system_prompt("Be helpful")
            .max_tokens(100);

        let chat_model = ChatModel::new(provider, chat_settings);

        let result = chat_model.generate_text("Tell me a joke").await.unwrap();

        // Check the response
        assert_eq!(result.text, "Hello, human!");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 20);
        assert_eq!(result.usage.total_tokens, 30);

        // Check that the provider was called with the correct arguments
        let captured_prompt = prompt_tracker.lock().unwrap();
        assert_eq!(*captured_prompt, Some("Tell me a joke".to_string()));

        let captured_settings = settings_tracker.lock().unwrap();
        assert!(captured_settings.is_some());
        let settings = captured_settings.as_ref().unwrap();
        assert_eq!(settings.system_prompt, Some("Be helpful".to_string()));
        assert_eq!(settings.max_tokens, Some(100));
    }

    #[tokio::test]
    async fn test_chat_model_generate_text_error() {
        let provider = MockProvider::with_error();
        let chat_model = ChatModel::new(provider, ChatSettings::new());

        let result = chat_model.generate_text("Tell me a joke").await;

        assert!(result.is_err());
        if let Err(AIError::ApiError(msg)) = result {
            assert_eq!(msg, "Mock error");
        } else {
            panic!("Expected AIError::ApiError");
        }
    }

    #[test]
    fn test_language_model_usage_calculation() {
        let usage = calculate_language_model_usage(15, 25);

        assert_eq!(usage.prompt_tokens, 15);
        assert_eq!(usage.completion_tokens, 25);
        assert_eq!(usage.total_tokens, 40);
    }
}
