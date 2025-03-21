use reqwest::header;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
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

    pub async fn generate_object<T: DeserializeOwned + JsonSchema + Sync>(
        &self,
        prompt: &str,
        parameters: &StructuredOutputParameters<T>,
    ) -> Result<StructuredOutput<T>, AIError> {
        self.provider
            .generate_object(prompt, &self.settings, parameters)
            .await
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

#[derive(Debug, Clone)]
pub struct StructuredOutputParameters<T: DeserializeOwned> {
    pub output: OutputType,
    pub mode: Option<Mode>,
    pub schema: Option<T>,
    pub schema_name: Option<String>,
    pub schema_description: Option<String>,
    pub enum_values: Option<Vec<String>>,
}

#[derive(Debug)]
pub struct StructuredOutput<T: DeserializeOwned> {
    pub value: StructuredResult<T>,
    pub finish_reason: FinishReason,
    pub usage: LanguageModelUsage,
}

#[derive(Debug)]
pub enum StructuredResult<T> {
    Object(T),
    Array(Vec<T>),
    Enum(String),
    NoSchema(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputType {
    Object,
    Array,
    Enum,
    NoSchema,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Auto,
    Json,
    Tool,
}

/// Temperature setting.
///
/// Disable normalization by setting `raw` to `true`.
#[derive(Debug, Clone)]
pub enum Temperature {
    Raw(f32),
    Normalized(f32),
}

impl Temperature {
    /// Creates a temperature.
    ///
    /// Since different providers use different `temperature` values, this is normalized across
    /// providers.
    ///
    /// Values smaller than `0.0` and larger than `1.0` are clamped.
    pub fn new(value: f32) -> Self {
        Temperature::Normalized(value.clamp(0.0, 1.0))
    }

    /// Creates a raw temperature that is used as is.
    ///
    /// You should check if the provided values is valid for the provider you are using.
    pub fn raw(value: f32) -> Self {
        Temperature::Raw(value)
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

    pub fn temperature(mut self, temperature: Temperature) -> Self {
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
