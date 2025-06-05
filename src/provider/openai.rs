use crate::core::{
    self,
    types::{StructuredRequest, StructuredResponse},
};
use async_trait::async_trait;
use schemars::schema_for;
use serde::{Deserialize, Serialize};

use crate::core::{
    builder::{LlmBuilder, MessagesSet},
    error::LlmError,
    traits::LlmProvider,
};

pub struct OpenAiClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    default_model: String,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_model: "gpt-4.1".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_default_model(mut self, model: String) -> Self {
        self.default_model = model;
        self
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn generate_structured<T>(
        &self,
        request: StructuredRequest,
    ) -> Result<StructuredResponse<T>, LlmError>
    where
        T: serde::de::DeserializeOwned + Send + schemars::JsonSchema,
    {
        // TODO: Fix error handling.

        let request = create_openai_structured_request::<T>(request)?;

        let url = format!("{}/responses", self.base_url);
        let res = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Network(format!("Failed to complete request: {}", e)))?;

        if !res.status().is_success() {
            // TODO: Handle error response from OpenAI API
            return Err(LlmError::Api(format!(
                "OpenAI API returned error: status code {}",
                res.status()
            )));
        }

        let response_text = res
            .text()
            .await
            .map_err(|e| LlmError::Network(format!("Failed to read response body: {}", e)))?;

        eprintln!("OpenAI API Response: {}", response_text);

        let api_res: OpenAiStructuredResponse = serde_json::from_str(&response_text)
            .map_err(|e| LlmError::Parse(format!("Failed to parse OpenAI response: {}", e)))?;

        create_core_structured_response(api_res)
    }
}

#[derive(Debug, Serialize)]
struct OpenAiStructuredRequest {
    model: String,
    input: Vec<InputMessage>,
    text: Format,
}

#[derive(Debug, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
enum Format {
    Text(TextType),
    JsonSchema(JsonSchema),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TextType {
    Text,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,

    schema: serde_json::Value,

    #[serde(rename = "type")]
    j_type: JsonSchemaType,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum JsonSchemaType {
    JsonSchema,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum InputMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
struct InputMessage {
    role: InputMessageRole,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiStructuredResponse {
    id: String,
    model: String,
    output: Vec<Message>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Message {
    id: String,

    /// This is always `message`
    #[serde(rename = "type")]
    m_type: String,

    status: MessageStatus,

    content: Vec<MessageContent>,

    /// This is always `assistant`
    role: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
enum MessageContent {
    OutputText(OutputText),
    Refusal(Refusal),
}

#[derive(Debug, Deserialize)]
struct OutputText {
    /// Always `output_text`
    #[serde(rename = "type")]
    c_type: String,

    text: String,
    // TODO
    // annotations
}

#[derive(Debug, Deserialize)]
struct Refusal {
    /// The refusal explanationfrom the model.
    refusal: String,

    /// Always `refusal`
    #[serde(rename = "type")]
    r_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MessageStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: i32,
    output_tokens: i32,
    total_tokens: i32,
}

pub fn create_openai_client_from_builder(
    builder: &LlmBuilder<MessagesSet>,
) -> Result<OpenAiClient, LlmError> {
    // Setting the model should be optional
    let model = builder
        .get_model()
        .ok_or_else(|| LlmError::ProviderConfiguration("Model not set".to_string()))?
        .to_string();

    let api_key = builder
        .get_api_key()
        .ok_or_else(|| LlmError::ProviderConfiguration("OPENAI_API_KEY not set.".to_string()))?
        .to_string();

    let client = OpenAiClient::new(api_key).with_default_model(model);
    Ok(client)
}

fn create_openai_structured_request<T>(
    req: StructuredRequest,
) -> Result<OpenAiStructuredRequest, LlmError>
where
    T: schemars::JsonSchema,
{
    let input = req
        .messages
        .into_iter()
        .map(|m| InputMessage {
            role: match m.role {
                core::types::ChatRole::System => InputMessageRole::System,
                core::types::ChatRole::User => InputMessageRole::User,
                core::types::ChatRole::Assistant => InputMessageRole::Assistant,
            },
            content: m.content,
        })
        .collect();

    let s = schema_for!(T);

    let schema_name = s
        .schema
        .metadata
        .as_ref()
        .and_then(|meta| meta.title.as_ref())
        .ok_or_else(|| {
            LlmError::Parse("Failed to build JSON Schema: Missing schema name".to_string())
        })?
        .clone();

    let schema = JsonSchema {
        name: schema_name,
        schema: serde_json::to_value(&s)
            .map_err(|e| LlmError::Parse(format!("Failed to build JSON Schema: {}", e)))?,
        j_type: JsonSchemaType::JsonSchema,
    };

    Ok(OpenAiStructuredRequest {
        model: req.model,
        input,
        text: Format::JsonSchema(schema),
    })
}

fn create_core_structured_response<T>(
    res: OpenAiStructuredResponse,
) -> Result<StructuredResponse<T>, LlmError>
where
    T: serde::de::DeserializeOwned,
{
    let message = res
        .output
        .first()
        .ok_or_else(|| LlmError::Parse("No messages in response".to_string()))?;

    let content = message
        .content
        .first()
        .ok_or_else(|| LlmError::Parse("No content in message".to_string()))?;

    let text = match content {
        MessageContent::OutputText(output) => &output.text,
        MessageContent::Refusal(refusal) => {
            return Err(LlmError::Api(format!("Model refused: {}", refusal.refusal)));
        }
    };

    let parsed_content: T = serde_json::from_str(&text)
        .map_err(|e| LlmError::Parse(format!("Failed to parse structured output: {}", e)))?;

    Ok(StructuredResponse {
        content: parsed_content,
        usage: core::types::LanguageModelUsage {
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.total_tokens,
        },
    })
}
