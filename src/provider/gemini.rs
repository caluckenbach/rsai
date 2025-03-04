use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    AIError,
    model::{
        ChatSettings,
        chat::{ChatMessage, ChatRole},
    },
};

use super::Provider;

pub struct GeminiProvider {
    pub api_key: String,
    pub model: String,
    pub client: Client,
}

impl GeminiProvider {
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
        prompt: &str,
        settings: &ChatSettings,
    ) -> Result<String, AIError> {
        let url = self.get_base_url(false);

        let mut contents = Vec::new();

        if let Some(messages) = &settings.messages {
            for msg in messages {
                let content = Content::try_from(msg.clone())?;
                contents.push(content);
            }
        }

        contents.push(Content {
            role: Role::User,
            parts: vec![Part {
                text: prompt.to_string(),
            }],
        });

        let message = Request { contents };

        let response = self
            .client
            .post(&url)
            .json(&message)
            .send()
            .await
            .map_err(|e| AIError::RequestError(e.to_string()))?;

        let json_response = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| AIError::ConversionError(e.to_string()))?;

        // Extract response text from Gemini API structure
        let response_text = json_response["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| AIError::ApiError("Failed to parse response".to_string()))?
            .to_string();

        Ok(response_text)
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

impl TryFrom<ChatMessage> for Content {
    type Error = AIError;

    fn try_from(msg: ChatMessage) -> Result<Self, Self::Error> {
        let role = match msg.role {
            ChatRole::System => {
                return Err(AIError::UnsuportedFunctionality(
                    "system messages are only supported at the beginning of the converstation"
                        .to_string(),
                ));
            }
            ChatRole::User => Role::User,
            ChatRole::Assistant => Role::Model,
        };

        Ok(Content {
            role,
            parts: vec![Part { text: msg.content }],
        })
    }
}
