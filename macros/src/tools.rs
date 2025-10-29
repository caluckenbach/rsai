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

    // Generate enum variants for type-safe tool choice
    let enum_variants: Vec<_> = tools_list
        .tools
        .iter()
        .map(|tool_name| {
            let variant_name = quote::format_ident!(
                "{}",
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
            );
            quote! {
                #variant_name,
            }
        })
        .collect();

    let choice_impl_variants: Vec<_> = tools_list
        .tools
        .iter()
        .map(|tool_name| {
            let variant_name = quote::format_ident!(
                "{}",
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
            );
            let tool_name_str = tool_name.to_string();
            quote! {
                Choice::#variant_name => ToolChoice::Function {
                    name: #tool_name_str.to_string()
                },
            }
        })
        .collect();

    // Generate the complete code with tools array and type-safe choice enum
    let expanded = quote! {
        {
            use rsai::core::{ToolFunction, types::{Tool, ToolChoice}};

            // Type-safe tool choice enum for this specific tool set
            #[derive(Debug, Clone, PartialEq)]
            pub enum Choice {
                None,
                Auto,
                Required,
                #(#enum_variants)*
            }

            impl From<Choice> for ToolChoice {
                fn from(choice: Choice) -> Self {
                    match choice {
                        Choice::None => ToolChoice::None,
                        Choice::Auto => ToolChoice::Auto,
                        Choice::Required => ToolChoice::Required,
                        #(#choice_impl_variants)*
                    }
                }
            }

            // Use the ToolSet type from core::types

            let mut registry = rsai::core::types::ToolRegistry::new();
            #(
                registry.register(std::sync::Arc::new(#wrapper_names));
            )*

            rsai::core::types::ToolSet {
                registry,
            }
        }
    };

    Ok(expanded)
}
