use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse::Parse, parse::ParseStream, Result, Ident, Token};

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
            "tools! macro requires at least one tool function"
        ));
    }
    
    // Use the same naming logic as the #[tool] macro to reference existing wrapper structs
    let tool_instances: Vec<_> = tools_list.tools.iter().map(|tool_name| {
        let wrapper_name = quote::format_ident!("{}Tool", 
            tool_name.to_string()
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
        quote! { #wrapper_name.schema() }
    }).collect();
    
    // Generate the Box<[Tool]> creation code
    let expanded = quote! {
        {
            use ai_rs::core::ToolFunction;
            let tools: Vec<ai_rs::core::types::Tool> = vec![
                #(#tool_instances,)*
            ];
            tools.into_boxed_slice()
        }
    };
    
    Ok(expanded)
}