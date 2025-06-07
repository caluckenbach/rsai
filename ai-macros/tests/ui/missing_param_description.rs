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
/// Function with missing parameter description
/// param1: First parameter description
/// (param2 description is missing)
fn function_with_missing_desc(param1: String, param2: i32) -> String {
    format!("{} {}", param1, param2)
}

fn main() {}