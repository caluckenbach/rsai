// Mock the ai_rs types for compile testing
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
        }
        
        pub trait ToolFunction: Send + Sync {
            fn schema(&self) -> types::Tool;
            fn execute<'a>(&'a self, params: serde_json::Value) -> types::BoxFuture<'a, Result<serde_json::Value, error::LlmError>>;
        }
    }
}

use ai_macros::tool;

#[tool]
/// Function with missing parameter description
/// param1: First parameter description
/// (param2 description is missing)
fn function_with_missing_desc(param1: String, param2: i32) -> String {
    format!("{} {}", param1, param2)
}

fn main() {}