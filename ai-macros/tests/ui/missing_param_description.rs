// Use the actual types from the main crate
use rsai::core::{ToolFunction, types::Tool, error::LlmError};
use rsai_macros::tool;

#[tool]
/// Function with missing parameter description
/// param1: First parameter description
/// (param2 description is missing)
fn function_with_missing_desc(param1: String, param2: i32) -> String {
    format!("{} {}", param1, param2)
}

fn main() {}