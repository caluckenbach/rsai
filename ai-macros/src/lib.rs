use proc_macro::TokenStream;
use quote::quote;

mod tool;
mod tools;

/// Attribute macro for types used with the `complete::<T>()` method.
///
/// This macro automatically adds the necessary derives and attributes:
/// - `#[derive(serde::Deserialize, schemars::JsonSchema)]`
/// - `#[schemars(deny_unknown_fields)]`
///
/// Usage:
/// ```rust,ignore
/// use ai_macros::completion_schema;
///
/// #[completion_schema]
/// struct Response {
///     answer: String,
/// }
///
/// // The macro expands to:
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// #[schemars(deny_unknown_fields)]
/// struct Response {
///     answer: String,
/// }
/// ```
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
/// use ai_macros::tool;
///
/// #[tool]
/// /// Get current weather for a city
/// /// city: The city to get weather for
/// /// unit: Temperature unit (celsius or fahrenheit)
/// fn get_weather(city: String, unit: Option<String>) -> String {
///     format!("Weather for {}: 22°C", city)
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    match tool::tool_impl(attr.into(), item.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Macro for creating a collection of tools from annotated functions.
///
/// This macro takes a comma-separated list of function names that have been
/// annotated with `#[tool]` and creates a `Box<[Tool]>` containing their schemas.
///
/// Usage:
/// ```rust
/// use ai_macros::{tool, tools};
///
/// #[tool]
/// /// Get current weather for a city
/// /// city: The city to get weather for
/// fn get_weather(city: String) -> String {
///     format!("Weather for {}: 22°C", city)
/// }
///
/// #[tool]
/// /// Calculate distance between two locations
/// /// from: Starting location
/// /// to: Destination location
/// fn calculate_distance(from: String, to: String) -> f64 {
///     42.5
/// }
///
/// let toolset = tools![get_weather, calculate_distance];
/// assert_eq!(toolset.tools.len(), 2);
/// ```
#[proc_macro]
pub fn tools(input: TokenStream) -> TokenStream {
    match tools::tools_impl(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
