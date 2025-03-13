use reqwest::header;
use std::any::Any;
use std::fmt::Debug;

use crate::AIError;
use crate::provider::Provider;

use futures::Stream;

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

    pub async fn stream_text<'a>(
        &'a self,
        prompt: &'a str,
    ) -> Result<impl Stream<Item = Result<TextStream, AIError>> + 'a, AIError> {
        self.provider.stream_text(prompt, &self.settings).await
    }
}

#[derive(Debug, Clone)]
pub struct ChatSettings {
    /// System prompt specifying model behavior.
    pub system_prompt: Option<String>,

    /// List of `Message` representing a conversation.
    pub messages: Option<Vec<Message>>,

    /// Maximum number of tokens to generate.
    pub max_tokens: Option<i32>,

    /// Temperature differs between providers.
    ///
    /// Use the `temperature()` method to set a normalized temperature value.
    /// For provider-specific temperature values, use the `raw_temperature()` method instead.
    pub temperature: Option<Temperature>,

    /// If the model generates any of these sequences, it will stop generating further text.
    pub stop_sequences: Option<Vec<String>>,

    /// Additional HTTP request headers.
    pub headers: reqwest::header::HeaderMap,

    /// Provider-specific options.
    pub provider_options: Option<Box<dyn ProviderOptions>>,
}

/// Temperature setting.
///
/// Disable normalization by setting `raw` to `true`.
#[derive(Debug, Clone)]
pub enum Temperature {
    Raw(f32),
    Normalized(f32),
}

/// A validated temperature value that is guaranteed to be between 0.0 and 1.0.
pub struct ValidTemperature(f32);

impl ValidTemperature {
    /// Creates a new temperature value that is checked at compile time.
    /// Will cause a compile error if the value is outside the valid range.
    pub const fn new(value: f32) -> Self {
        assert!(
            value >= 0.0 && value <= 1.0,
            "Temperature must be between 0.0 and 1.0"
        );
        Self(value)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            system_prompt: None,
            messages: None,
            max_tokens: None,
            temperature: None,
            stop_sequences: None,
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

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn add_message(mut self, message: Message) -> Self {
        let messages = self.messages.get_or_insert_with(Vec::new);
        messages.push(message);
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets a normalized temperature between 0.0 and 1.0.
    /// Returns an error if the temperature is outside this range.
    /// For temperatures outside this range, use `raw_temperature()`.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(Temperature::Normalized(
            ValidTemperature::new(temperature).value(),
        ));
        self
    }

    /// Sets a raw temperature value without normalization.
    /// Use this for provider-specific temperature values that may be outside the 0.0 - 1.0 range.
    pub fn raw_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(Temperature::Raw(temperature));
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
    //pub reasoning_text: Option<String>,
    //pub reasoning:
    //pub sources:
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
}

#[derive(Debug)]
pub struct TextStream {
    pub text: String,
    //pub reasoning_text: Option<String>,
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
