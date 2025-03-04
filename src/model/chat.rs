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

    pub async fn generate_text(&self, prompt: &str) -> Result<String, AIError> {
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

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone)]
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
