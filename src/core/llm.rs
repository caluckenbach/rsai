use reqwest::Client;
use serde_json::{Value, json};
use crate::tool::{Tool, FunctionDeclaration};
use std::error::Error;

/// Generate content using the LLM with tools
pub async fn generate_content_with_tools(
    client: &Client,
    api_key: &str,
    prompt: &str,
    tools: &[Tool],
) -> Result<Value, Box<dyn Error>> {
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
) -> Result<Value, Box<dyn Error>> {
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
        .get("candidates")?.as_array()?
        .first()?
        .get("content")?
        .get("parts")?.as_array()?
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
        .get("candidates")?.as_array()?
        .first()?
        .get("content")?
        .get("parts")?.as_array()?
        .first()?
        .get("text")?.as_str()
        .map(|s| s.to_string())
}