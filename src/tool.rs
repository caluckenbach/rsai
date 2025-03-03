use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// Core tool definition
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Parameters,
    pub execute: fn(HashMap<String, String>) -> Value,
}

// Parameter definitions for tools
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Parameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: HashMap<String, Property>,
    pub required: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Property {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: String,
}

// Function declaration for LLM API
#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Parameters,
}

// Helper function to create a new tool
pub fn tool(
    name: &str,
    description: &str,
    parameters: HashMap<String, (String, String)>,
    required: Vec<&str>,
    execute: fn(HashMap<String, String>) -> Value,
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

// Process a function call by finding the appropriate tool and executing it
pub fn process_function_call(
    tools: &[Tool],
    function_name: &str,
    args: &Value,
) -> Option<Value> {
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