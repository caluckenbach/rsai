#[test]
fn test_extra_param_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/extra_param_in_docstring.rs");
}

#[test]
fn test_missing_param_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/missing_param_description.rs");
}