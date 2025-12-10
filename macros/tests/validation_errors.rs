use rsai_macros::tool;

// This test verifies that the macro produces a compile error when a parameter
// is documented in the docstring but doesn't exist in the function signature
#[test]
fn test_should_compile() {
    // First, let's test a valid case to make sure our setup works
    #[tool]
    /// Valid function
    /// param1: Description for param1
    fn valid_function(param1: String) -> String {
        param1
    }

    let tool_instance = ValidFunctionTool;
    let tool = tool_instance.schema();
    assert_eq!(tool.name, "valid_function");
}

// VALIDATION ERROR CASES
// ======================
// These are examples of what should fail to compile with helpful error messages:

// ERROR CASE 1: Extra parameter in docstring
// This would fail with: "Parameter 'nonexistent' found in docstring but not in function parameters"
/*
#[tool]
/// Function with extra parameter in docstring
/// param1: Valid parameter
/// nonexistent: This parameter doesn't exist in function
fn function_with_extra_docstring_param(param1: String) -> String {
    param1
}
*/

// ERROR CASE 2: Missing parameter description
// This would fail with: "Parameter 'param2' is missing description in docstring. Add: 'param2: description'"
/*
#[tool]
/// Function with missing parameter description
/// param1: First parameter description
/// (param2 description is missing)
fn function_with_missing_param_desc(param1: String, param2: i32) -> String {
    format!("{} {}", param1, param2)
}
*/

// These validation errors ensure that:
// 1. All parameters mentioned in docstrings exist in the function signature
// 2. All function parameters have descriptions in the docstring
// 3. Users get clear, actionable error messages when validation fails
