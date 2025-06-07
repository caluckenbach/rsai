use ai_macros::{tool, tools};

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

    #[test]
    fn test_single_tool_macro() {
        // Test that the macro compiles and creates a toolset
        let toolset = tools![get_weather];

        // Basic assertions to verify the structure works
        assert_eq!(toolset.tools.len(), 1);
        assert_eq!(toolset.tools[0].name, "get_weather");
        assert!(toolset.tools[0].description.is_some());
    }

    #[test]
    fn test_multiple_tools_macro() {
        // Test with multiple tools
        let toolset = tools![get_weather, calculate_distance];

        assert_eq!(toolset.tools.len(), 2);
        assert_eq!(toolset.tools[0].name, "get_weather");
        assert_eq!(toolset.tools[1].name, "calculate_distance");
    }
}
