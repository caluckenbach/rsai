use crate::core::{AIError, TextStream};
use crate::core::llm::GenerationOptions;
use crate::provider::Provider;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Gemini provider implementation
pub struct GeminiProvider {
    pub api_key: String,
    pub client: Client,
    pub model: String,
}

impl GeminiProvider {
    /// Create a new GeminiProvider instance
    pub fn new(api_key: &str, model: &str) -> Self {
        GeminiProvider {
            api_key: api_key.to_string(),
            client: Client::new(),
            model: model.to_string(),
        }
    }

    /// Create a new GeminiProvider with default model
    ///
    /// default model: `gemini-2.0-flash`
    pub fn default(api_key: &str) -> Self {
        Self::new(api_key, "gemini-2.0-flash")
    }

    fn get_base_url(&self, stream: bool) -> String {
        let model_url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}",
            self.model
        );

        if stream {
            format!(
                "{}:streamGenerateContent?alt=sse&key={}",
                model_url, self.api_key
            )
        } else {
            format!("{}:generateContent?key={}", model_url, self.api_key)
        }
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn generate_text(
        &self,
        contents: &str,
        options: &GenerationOptions,
    ) -> Result<String, AIError> {
        let url = self.get_base_url(false);

        // Parse the contents into a GeminiMessage
        let message = Request {
            contents: vec![Content {
                role: Role::User,
                parts: vec![Part {
                    text: contents.to_string(),
                }],
            }],
        };

        let completion = self
            .client
            .post(&url)
            .json(&message)
            .send()
            .await
            .map_err(|e| AIError::RequestError(e.to_string()));

        todo!("Implement the actual API request")
    }

    async fn stream_text(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<TextStream, AIError> {
        todo!()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Part {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: Role,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub contents: Vec<Content>,
}

// Helper function to create a conversation
pub fn create_conversation(messages: Vec<(Role, &str)>) -> Request {
    let contents = messages
        .into_iter()
        .map(|(role, text)| Content {
            role,
            parts: vec![Part {
                text: text.to_string(),
            }],
        })
        .collect();

    Request { contents }
}
