use proc_macro::TokenStream;
use quote::quote;

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
