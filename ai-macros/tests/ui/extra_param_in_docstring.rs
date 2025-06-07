// Mock the ai_rs types for compile testing
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
/// Function with extra parameter in docstring
/// param1: Valid parameter description
/// nonexistent: This parameter doesn't exist in the function
fn function_with_extra_param(param1: String) -> String {
    param1
}

fn main() {}