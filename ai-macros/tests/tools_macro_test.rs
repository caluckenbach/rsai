use ai_macros::{tool, toolset};

/// Get the current weather for a city
/// _city: The city to get weather for
/// _unit: Temperature unit (celsius or fahrenheit)
#[tool]
async fn get_weather(_city: String, _unit: Option<String>) -> f64 {
    22.0
}

/// Calculate distance between two locations
/// _from: Starting location
/// _to: Destination location
#[tool]
fn calculate_distance(_from: String, _to: String) -> f64 {
    42.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai::core::types::{ToolCall, ToolChoice};
    use serde_json::json;

    #[test]
    fn test_single_tool_macro() {
        // Test that the macro compiles and creates a toolset
        let toolset = toolset![get_weather];

        // Basic assertions to verify the structure works
        assert_eq!(toolset.tools.len(), 1);
        assert_eq!(toolset.tools[0].name, "get_weather");
        assert!(toolset.tools[0].description.is_some());
    }

    #[test]
    fn test_multiple_tools_macro() {
        // Test with multiple tools
        let toolset = toolset![get_weather, calculate_distance];

        assert_eq!(toolset.tools.len(), 2);
        assert_eq!(toolset.tools[0].name, "get_weather");
        assert_eq!(toolset.tools[1].name, "calculate_distance");
    }

    #[test]
    fn test_registry_execution_functionality() {
        let toolset = toolset![get_weather, calculate_distance];
        
        // Test that registry has correct tools
        let schemas = toolset.registry.get_schemas();
        assert_eq!(schemas.len(), 2);
        
        // Verify schema names match
        let names: Vec<String> = schemas.iter().map(|s| s.name.clone()).collect();
        assert!(names.contains(&"get_weather".to_string()));
        assert!(names.contains(&"calculate_distance".to_string()));
    }

    #[test]
    fn test_tool_parameter_schemas_in_registry() {
        let toolset = toolset![get_weather];
        let schemas = toolset.registry.get_schemas();
        
        assert_eq!(schemas.len(), 1);
        let weather_schema = &schemas[0];
        
        // Verify parameter schema structure
        assert_eq!(weather_schema.name, "get_weather");
        assert!(weather_schema.description.is_some());
        assert_eq!(weather_schema.description.as_ref().unwrap(), "Get the current weather for a city");
        
        // Check parameter structure
        let params = &weather_schema.parameters;
        assert_eq!(params["type"], "object");
        
        let properties = &params["properties"];
        assert!(properties["_city"].is_object());
        assert!(properties["_unit"].is_object());
        
        // Verify required parameters
        let required = &params["required"];
        assert!(required.is_array());
        let required_array = required.as_array().unwrap();
        assert!(required_array.contains(&json!("_city")));
        assert!(!required_array.contains(&json!("_unit"))); // Optional parameter
    }

    #[tokio::test]
    async fn test_registry_execute_method() {
        let toolset = toolset![get_weather, calculate_distance];
        
        // Test successful execution of get_weather
        let weather_call = ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "New York",
                "_unit": "celsius"
            }),
        };
        
        let result = toolset.registry.execute(&weather_call).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "22.0");
        
        // Test successful execution of calculate_distance
        let distance_call = ToolCall {
            id: "call_2".to_string(),
            name: "calculate_distance".to_string(),
            arguments: json!({
                "_from": "New York",
                "_to": "Boston"
            }),
        };
        
        let result = toolset.registry.execute(&distance_call).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42.5");
    }

    #[tokio::test]
    async fn test_registry_execute_method_error_cases() {
        let toolset = toolset![get_weather];
        
        // Test execution with non-existent tool
        let invalid_call = ToolCall {
            id: "call_invalid".to_string(),
            name: "non_existent_tool".to_string(),
            arguments: json!({}),
        };
        
        let result = toolset.registry.execute(&invalid_call).await;
        assert!(result.is_err());
        
        // Verify error message contains tool name
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("non_existent_tool"));
    }

    #[test]
    fn test_choice_enum_generation() {
        let toolset = toolset![get_weather, calculate_distance];
        
        // The macro should generate a Choice enum, but we need to access it properly
        // This tests that the macro compiles with choice enum syntax
        // We can't directly test the enum without exposing it, but we can verify
        // the toolset structure that depends on it works correctly
        
        assert_eq!(toolset.tools.len(), 2);
        assert_eq!(toolset.registry.get_schemas().len(), 2);
    }

    #[test]
    fn test_choice_enum_to_tool_choice_conversion() {
        // Test within the macro scope where Choice enum is available
        let toolset = toolset![get_weather, calculate_distance];
        
        // We can't directly access the Choice enum from outside the macro,
        // but we can verify that the tool choice functionality works
        // by checking that the schemas support the expected tool names
        let _schemas = toolset.registry.get_schemas();
        
        // Verify that we can create ToolChoice::Function variants for our tools
        let weather_choice = ToolChoice::Function { 
            name: "get_weather".to_string() 
        };
        let distance_choice = ToolChoice::Function { 
            name: "calculate_distance".to_string() 
        };
        
        match weather_choice {
            ToolChoice::Function { name } => assert_eq!(name, "get_weather"),
            _ => panic!("Expected Function variant"),
        }
        
        match distance_choice {
            ToolChoice::Function { name } => assert_eq!(name, "calculate_distance"),
            _ => panic!("Expected Function variant"),
        }
    }

    #[tokio::test]
    async fn test_tool_invocation_through_registry() {
        let toolset = toolset![get_weather, calculate_distance];
        
        // Test complete workflow: schema -> call -> execution
        let schemas = toolset.registry.get_schemas();
        assert_eq!(schemas.len(), 2);
        
        // Find weather tool schema
        let weather_schema = schemas.iter()
            .find(|s| s.name == "get_weather")
            .expect("Should find get_weather schema");
        
        // Create tool call based on schema
        let tool_call = ToolCall {
            id: "test_call".to_string(),
            name: weather_schema.name.clone(),
            arguments: json!({
                "_city": "San Francisco",
                "_unit": "fahrenheit"
            }),
        };
        
        // Execute through registry
        let result = toolset.registry.execute(&tool_call).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "22.0");
        
        // Test synchronous tool (calculate_distance)
        let distance_schema = schemas.iter()
            .find(|s| s.name == "calculate_distance")
            .expect("Should find calculate_distance schema");
        
        let distance_call = ToolCall {
            id: "distance_call".to_string(),
            name: distance_schema.name.clone(),
            arguments: json!({
                "_from": "Los Angeles",
                "_to": "San Diego"
            }),
        };
        
        let result = toolset.registry.execute(&distance_call).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42.5");
    }

    #[tokio::test]
    async fn test_registry_with_optional_parameters() {
        let toolset = toolset![get_weather];
        
        // Test with optional parameter provided
        let call_with_unit = ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "Tokyo",
                "_unit": "celsius"
            }),
        };
        
        let result = toolset.registry.execute(&call_with_unit).await;
        assert!(result.is_ok());
        
        // Test without optional parameter (unit should default to None)
        let call_without_unit = ToolCall {
            id: "call_2".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({
                "_city": "Tokyo"
            }),
        };
        
        let result = toolset.registry.execute(&call_without_unit).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "22.0");
    }
}
