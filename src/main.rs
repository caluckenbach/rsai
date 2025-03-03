use dotenv::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone)]
struct Tool {
    name: String,
    description: String,
    parameters: Parameters,
    execute: fn(HashMap<String, String>) -> serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Parameters {
    #[serde(rename = "type")]
    param_type: String,
    properties: HashMap<String, Property>,
    required: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Property {
    #[serde(rename = "type")]
    property_type: String,
    description: String,
}

// This will be sent to the LLM
#[derive(Serialize, Deserialize, Debug)]
struct FunctionDeclaration {
    name: String,
    description: String,
    parameters: Parameters,
}

fn tool(
    name: &str,
    description: &str,
    parameters: HashMap<String, (String, String)>,
    required: Vec<&str>,
    execute: fn(HashMap<String, String>) -> serde_json::Value,
) -> Tool {
    let mut properties = HashMap::new();
    for (key, (property_type, description)) in parameters {
        properties.insert(
            key.to_string(),
            Property {
                property_type,
                description,
            },
        );
    }

    Tool {
        name: name.to_string(),
        description: description.to_string(),
        parameters: Parameters {
            param_type: "object".to_string(),
            properties,
            required: required.iter().map(|&s| s.to_string()).collect(),
        },
        execute,
    }
}

async fn generate_content_with_tools(
    client: &Client,
    api_key: &str,
    prompt: &str,
    tools: &[Tool],
) -> Result<Value, Box<dyn std::error::Error>> {
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
            "parts": [{"text": prompt}]
        }],
        "tools": [{
            "function_declarations": function_declarations
        }],
        //"tool_config": {
        //    "function_calling_config": {
        //        "mode": "auto",
        //        "allowed_function_names": tools.iter().map(|t| t.name.clone()).collect::<Vec<String>>()
        //    }
        //}
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

fn process_function_call(
    tools: &[Tool],
    function_name: &str,
    args: &Value,
) -> Option<serde_json::Value> {
    let tool = tools.iter().find(|t| t.name == function_name)?;

    // Convert args to HashMap<String, String>
    let mut params = HashMap::new();
    if let Some(obj) = args.as_object() {
        for (key, value) in obj {
            if let Some(value_str) = value.as_str() {
                params.insert(key.clone(), value_str.to_string());
            }
        }
    }

    Some((tool.execute)(params))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let client = Client::new();

    let weather_tool = tool(
        "get_weather",
        "Get the weather in a location",
        {
            let mut params = HashMap::new();
            params.insert(
                "location".to_string(),
                (
                    "string".to_string(),
                    "The location to get the weather for".to_string(),
                ),
            );
            params
        },
        vec!["location"],
        |params| {
            let binding = "unknown".to_string();
            let location = params.get("location").unwrap_or(&binding);
            let temperature = 24;

            json!({
                "location": location,
                "temperature": temperature,
                "unit": "celsius",
                "condition": "sunny"
            })
        },
    );

    let prompt = "What's the weather like in San Francisco?";
    let response =
        generate_content_with_tools(&client, &api_key, prompt, &[weather_tool.clone()]).await?;

    // Check for function calls in response
    if let Some(candidates) = response.get("candidates").and_then(|c| c.as_array()) {
        if let Some(candidate) = candidates.first() {
            if let Some(function_calls) = candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(|parts| parts.as_array())
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("functionCall"))
            {
                println!("Function call detected:");

                // Extract function name and arguments
                if let (Some(name), Some(args)) = (
                    function_calls.get("name").and_then(|n| n.as_str()),
                    function_calls.get("args"),
                ) {
                    println!("Function: {}", name);
                    println!("Args: {}", args);

                    // Execute the tool
                    if let Some(result) = process_function_call(&[weather_tool], name, args) {
                        println!("\nTool execution result:");
                        println!("{}", result);

                        // You could send this result back to Gemini for a final response
                    }
                }
            } else if let Some(text) = candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(|parts| parts.as_array())
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("text"))
                .and_then(|text| text.as_str())
            {
                println!("AI Response:\n{}", text);
            }
        }
    } else {
        println!("Failed to parse response. Raw response:\n{:#?}", response);
    }

    Ok(())
}
