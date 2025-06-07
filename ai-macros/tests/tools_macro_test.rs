// Mock the ai_rs types for testing the macro in isolation
mod ai_rs {
    pub mod core {
        pub mod types {
            use std::future::Future;
            use std::pin::Pin;

            #[derive(Debug, Clone, PartialEq)]
            pub struct Tool {
                pub name: String,
                pub description: Option<String>,
                pub parameters: serde_json::Value,
                pub strict: Option<bool>,
            }

            pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
        }

        pub mod error {
            #[derive(Debug)]
            pub enum LlmError {
                ToolExecution {
                    message: String,
                    source: Option<Box<dyn std::error::Error + Send + Sync>>,
                },
            }

            impl std::fmt::Display for LlmError {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        LlmError::ToolExecution { message, .. } => write!(f, "{}", message),
                    }
                }
            }

            impl std::error::Error for LlmError {}
        }

        pub trait ToolFunction: Send + Sync {
            fn schema(&self) -> types::Tool;
            fn execute<'a>(
                &'a self,
                params: serde_json::Value,
            ) -> types::BoxFuture<'a, Result<serde_json::Value, error::LlmError>>;
        }
    }
}

use ai_macros::{tool, tools};
use ai_rs::core::error::LlmError;

/// Get the current weather for a city
/// city: The city to get weather for
/// unit: Temperature unit (celsius or fahrenheit)
#[tool]
async fn get_weather(
    city: String,
    unit: Option<String>,
) -> f64 {
    22.0
}

/// Calculate distance between two locations
/// from: Starting location
/// to: Destination location  
#[tool]
fn calculate_distance(
    from: String,
    to: String,
) -> f64 {
    42.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_rs::core::ToolFunction;

    #[test]
    fn test_tools_macro_creates_collection() {
        let tools = tools![get_weather, calculate_distance];
        
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[1].name, "calculate_distance");
    }
}