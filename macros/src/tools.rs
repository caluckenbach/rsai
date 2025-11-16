use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Result, Token, parse::Parse, parse::ParseStream};

/// Parses a comma-separated list of identifiers for the tools! macro
struct ToolsList {
    tools: Vec<Ident>,
}

impl Parse for ToolsList {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut tools = Vec::new();

        while !input.is_empty() {
            tools.push(input.parse::<Ident>()?);

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }

        Ok(ToolsList { tools })
    }
}

pub fn tools_impl(input: TokenStream) -> Result<TokenStream> {
    let tools_list = syn::parse2::<ToolsList>(input)?;

    if tools_list.tools.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "tools! macro requires at least one tool function",
        ));
    }

    // Use the same naming logic as the #[tool] macro to reference existing wrapper structs
    let wrapper_names: Vec<_> = tools_list
        .tools
        .iter()
        .map(|tool_name| {
            quote::format_ident!(
                "{}Tool",
                tool_name
                    .to_string()
                    .split('_')
                    .map(|s| {
                        let mut c = s.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    })
                    .collect::<String>()
            )
        })
        .collect();

    // Generate the complete code with tools array and type-safe choice enum
    let expanded = quote! {
        {
            use rsai::{Tool, ToolChoice, ToolFunction, ToolRegistry, ToolSet};

            let registry = ToolRegistry::new();
            #(
                registry.register(std::sync::Arc::new(#wrapper_names))
                .expect(&format!("Failed to register tool: {}", stringify!(#wrapper_names)));
            )*

            ToolSet {
                registry,
            }
        }
    };

    Ok(expanded)
}
