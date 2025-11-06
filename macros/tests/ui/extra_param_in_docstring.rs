use rsai::tool;

#[tool]
/// Function with extra parameter in docstring
/// param1: Valid parameter description
/// nonexistent: This parameter doesn't exist in the function
fn function_with_extra_param(param1: String) -> String {
    param1
}

fn main() {}
