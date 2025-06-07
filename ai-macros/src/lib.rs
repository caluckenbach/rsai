use proc_macro::TokenStream;
use quote::quote;

mod tool;

/// Attribute macro for types used with the `complete::<T>()` method.
///
/// This macro automatically adds the necessary derives and attributes:
/// - `#[derive(serde::Deserialize, schemars::JsonSchema)]`
/// - `#[schemars(deny_unknown_fields)]`
///
/// Usage:
/// ```rust
/// #[completion_schema]
/// struct Response {
///     answer: String,
/// }
///
/// let result = llm.complete::<Response>().await?;
/// ```
///
/// NOTE: This macro implementation was generated via Claude Code
/// and still needs to be double-checked for correctness and edge cases
#[proc_macro_attribute]
pub fn completion_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_tokens: proc_macro2::TokenStream = item.into();

    let expanded = quote! {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #[schemars(deny_unknown_fields)]
        #item_tokens
    };

    TokenStream::from(expanded)
}

/// Attribute macro for marking functions as tools that can be called by LLMs.
///
/// This macro generates the necessary boilerplate to make a function callable
/// as a tool, including JSON schema generation and parameter parsing.
///
/// Usage:
/// ```rust
/// #[tool]
/// async fn get_weather(
///     /// The city to get weather for
///     city: String,
///     /// Temperature unit (celsius or fahrenheit)
///     unit: Option<String>,
/// ) -> Result<WeatherData, ToolError> {
///     // Implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    match tool::tool_impl(attr.into(), item.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
