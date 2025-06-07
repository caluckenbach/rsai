// Mock the ai_rs types for testing the macro in isolation
mod ai_rs {
    pub mod core {
        pub mod types {
            #[derive(Debug, Clone, PartialEq)]
            pub struct Tool {
                pub name: String,
                pub description: Option<String>,
                pub parameters: serde_json::Value,
                pub strict: Option<bool>,
            }
        }
    }
}

use ai_macros::tool;

#[tool]
/// Get current temperature for a given location.
/// location: City and country e.g. Bogotá, Colombia
fn get_weather(location: String) -> f64 {
    22.5
}

#[test]
fn test_get_weather_tool_schema() {
    let tool = get_weather_tool();
    
    // Check the tool name
    assert_eq!(tool.name, "get_weather");
    
    // Check the description
    assert_eq!(tool.description, Some("Get current temperature for a given location.".to_string()));
    
    // Check that strict mode is enabled
    assert_eq!(tool.strict, Some(true));
    
    // Parse the parameters schema
    let params = &tool.parameters;
    
    // Check that it's an object type
    assert_eq!(params["type"], "object");
    
    // Check properties
    let properties = &params["properties"];
    let location_prop = &properties["location"];
    assert_eq!(location_prop["type"], "string");
    assert_eq!(location_prop["description"], "City and country e.g. Bogotá, Colombia");
    
    // Check required fields
    let required = params["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "location");
    
    // Check additionalProperties
    assert_eq!(params["additionalProperties"], false);
    
    // Print the full schema for debugging
    println!("Generated schema: {}", serde_json::to_string_pretty(&tool.parameters).unwrap());
}

#[tool]
/// This function does multiple things with various parameters.
/// It's a complex function for testing.
/// param1: First parameter description
/// param2: Second parameter that is optional
/// param3: Third parameter with special chars: symbols, numbers, etc.
fn complex_function(
    param1: String,
    param2: Option<i32>,
    param3: bool,
) -> String {
    format!("{} {} {}", param1, param2.unwrap_or(0), param3)
}

#[test]
fn test_complex_function_docstring_parsing() {
    let tool = complex_function_tool();
    
    // Check function description (should exclude parameter lines)
    assert_eq!(
        tool.description, 
        Some("This function does multiple things with various parameters. It's a complex function for testing.".to_string())
    );
    
    let params = &tool.parameters;
    let properties = &params["properties"];
    
    // Check param1
    let param1 = &properties["param1"];
    assert_eq!(param1["type"], "string");
    assert_eq!(param1["description"], "First parameter description");
    
    // Check param2 (optional)
    let param2 = &properties["param2"];
    assert_eq!(param2["type"], "integer");
    assert_eq!(param2["description"], "Second parameter that is optional");
    
    // Check param3
    let param3 = &properties["param3"];
    assert_eq!(param3["type"], "boolean");
    assert_eq!(param3["description"], "Third parameter with special chars: symbols, numbers, etc.");
    
    // Check required array (should only have param1 and param3, not param2 since it's Option<T>)
    let required = params["required"].as_array().unwrap();
    assert_eq!(required.len(), 2);
    assert!(required.contains(&serde_json::Value::String("param1".to_string())));
    assert!(required.contains(&serde_json::Value::String("param3".to_string())));
    assert!(!required.contains(&serde_json::Value::String("param2".to_string())));
}

