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
/// Function with extra parameter in docstring
/// param1: Valid parameter description
/// nonexistent: This parameter doesn't exist in the function
fn function_with_extra_param(param1: String) -> String {
    param1
}

fn main() {}