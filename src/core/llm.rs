use crate::{
    error::AIError,
    provider::Provider,
    tool::{FunctionDeclaration, Tool},
};
use reqwest::Client;
use serde_json::{Value, json};

pub struct GenerationOptions {
    pub system_prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    // Add other optional parameters here (temperature, max tokens, etc.)
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<i32>,
    pub max_retries: Option<i32>,
    // TODO: add abort signal
    // TODO: hide this implementation detail
    pub headers: reqwest::header::HeaderMap,
    // TODO: provider options
}

impl GenerationOptions {
    pub fn new() -> GenerationOptionsBuilder {
        GenerationOptionsBuilder::default()
    }
}

#[derive(Default)]
pub struct GenerationOptionsBuilder {
    system_prompt: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    seed: Option<i32>,
    max_retries: Option<i32>,
    headers: reqwest::header::HeaderMap,
}

impl GenerationOptionsBuilder {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
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

    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_retries(mut self, max_retries: i32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    pub fn headers(mut self, headers: reqwest::header::HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    pub fn build(self) -> GenerationOptions {
        GenerationOptions {
            system_prompt: self.system_prompt,
            messages: self.messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stop_sequences: self.stop_sequences,
            seed: self.seed,
            max_retries: self.max_retries,
            headers: self.headers,
        }
    }
}

pub async fn generate_text(
    provider: &dyn Provider,
    prompt: &str,
    options: &GenerationOptions,
) -> Result<String, AIError> {
    let response = provider.generate_text(prompt, options).await?;

    Ok(response)
}

/// Generate content using the LLM with tools
pub async fn generate_content_with_tools(
    client: &Client,
    api_key: &str,
    prompt: &str,
    tools: &[Tool],
) -> Result<Value, AIError> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
        api_key
    );

    // Convert our Tool structs to what Gemini expects
    let function_declarations: Vec<FunctionDeclaration> = tools
        .iter()
        .map(|tool| FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect();

    let request_body = json!({
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "tools": [{
            "function_declarations": function_declarations
        }],
        "tool_config": {
            "function_calling_config": {
                "mode": "AUTO"
            }
        }
    });

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(response)
}

/// Send function results back to the LLM for a second turn
pub async fn send_function_results(
    client: &Client,
    api_key: &str,
    prompt: &str,
    function_name: &str,
    function_result: &Value,
    tools: &[Tool],
) -> Result<Value, AIError> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
        api_key
    );

    // Convert our Tool structs to what Gemini expects
    let function_declarations: Vec<FunctionDeclaration> = tools
        .iter()
        .map(|tool| FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect();

    // Create request with conversation history including function result
    let request_body = json!({
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            },
            {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": function_name,
                        "args": {}  // We don't have the original args here
                    }
                }]
            },
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": function_name,
                        "response": {
                            "name": function_name,
                            "content": function_result
                        }
                    }
                }]
            }
        ],
        "tools": [{
            "function_declarations": function_declarations
        }],
        "tool_config": {
            "function_calling_config": {
                "mode": "AUTO"
            }
        }
    });

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(response)
}

/// Extract function call information from LLM response
pub fn extract_function_call(response: &Value) -> Option<(String, Value)> {
    response
        .get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?
        .first()?
        .get("functionCall")
        .and_then(|function_call| {
            let name = function_call.get("name")?.as_str()?.to_string();
            let args = function_call.get("args")?.clone();
            Some((name, args))
        })
}

/// Extract text response from LLM
pub fn extract_text_response(response: &Value) -> Option<String> {
    response
        .get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?
        .first()?
        .get("text")?
        .as_str()
        .map(|s| s.to_string())
}
