use proc_macro2::TokenStream;
use quote::quote;
use syn::{ItemFn, Result, FnArg, Pat, Type, Attribute};

pub fn tool_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let _ = attr; // Currently unused, could be used for tool configuration
    
    let input = syn::parse2::<ItemFn>(item)?;
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();
    
    // Extract function description and parameter descriptions from doc comments
    let (description, param_descriptions) = extract_doc_comment_and_params(&input.attrs);
    
    // Parse function parameters
    let params = parse_parameters(&input.sig.inputs, &param_descriptions)?;
    
    // Generate JSON schema for parameters
    let schema = generate_parameter_schema(&params)?;
    
    // Generate a function that returns the Tool struct
    let tool_fn_name = quote::format_ident!("{}_tool", fn_name);
    
    // Generate the complete implementation
    let expanded = quote! {
        #input
        
        pub fn #tool_fn_name() -> ai_rs::core::types::Tool {
            ai_rs::core::types::Tool {
                name: #fn_name_str.to_string(),
                description: #description,
                parameters: #schema,
                strict: Some(true),
            }
        }
    };
    
    Ok(expanded)
}

fn extract_doc_comment_and_params(attrs: &[Attribute]) -> (TokenStream, std::collections::HashMap<String, String>) {
    let doc_strings: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(meta) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &meta.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            return Some(lit_str.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .collect();
    
    let mut description_lines = Vec::new();
    let mut param_descriptions = std::collections::HashMap::new();
    
    for line in doc_strings {
        // Check if this line describes a parameter (format: "param_name: description")
        if let Some(colon_pos) = line.find(':') {
            let param_name = line[..colon_pos].trim();
            let param_desc = line[colon_pos + 1..].trim();
            
            // Only treat it as a parameter description if the param name looks like an identifier
            if param_name.chars().all(|c| c.is_alphanumeric() || c == '_') && !param_name.is_empty() {
                param_descriptions.insert(param_name.to_string(), param_desc.to_string());
                continue;
            }
        }
        
        // Otherwise, it's part of the function description
        description_lines.push(line);
    }
    
    let description = if description_lines.is_empty() {
        quote! { None }
    } else {
        let description = description_lines.join(" ");
        quote! { Some(#description.to_string()) }
    };
    
    (description, param_descriptions)
}

struct Parameter {
    name: String,
    ty: Type,
    description: Option<String>,
    required: bool,
}

fn parse_parameters(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    param_descriptions: &std::collections::HashMap<String, String>,
) -> Result<Vec<Parameter>> {
    let mut params = Vec::new();
    
    for arg in inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(arg, "tool functions cannot have self parameter"));
            }
            FnArg::Typed(pat_type) => {
                let name = match &*pat_type.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => return Err(syn::Error::new_spanned(&pat_type.pat, "only simple identifiers are supported for parameters")),
                };
                
                // Get parameter description from docstring parsing
                let description = param_descriptions.get(&name).cloned();
                
                // Check if type is Option<T>
                let (ty, required) = match &*pat_type.ty {
                    Type::Path(type_path) => {
                        let segments = &type_path.path.segments;
                        if segments.len() == 1 && segments[0].ident == "Option" {
                            // Extract inner type from Option<T>
                            if let syn::PathArguments::AngleBracketed(args) = &segments[0].arguments {
                                if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                                    (inner_ty.clone(), false)
                                } else {
                                    ((*pat_type.ty).clone(), true)
                                }
                            } else {
                                ((*pat_type.ty).clone(), true)
                            }
                        } else {
                            ((*pat_type.ty).clone(), true)
                        }
                    }
                    _ => ((*pat_type.ty).clone(), true),
                };
                
                params.push(Parameter {
                    name,
                    ty,
                    description,
                    required,
                });
            }
        }
    }
    
    Ok(params)
}


fn generate_parameter_schema(params: &[Parameter]) -> Result<TokenStream> {
    if params.is_empty() {
        return Ok(quote! {
            ::serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            })
        });
    }

    let properties: Vec<_> = params.iter().map(|param| {
        let name = &param.name;
        let type_str = type_to_json_type(&param.ty).unwrap_or("string");
        
        if let Some(desc) = &param.description {
            quote! {
                (#name, ::serde_json::json!({
                    "type": #type_str,
                    "description": #desc
                }))
            }
        } else {
            quote! {
                (#name, ::serde_json::json!({
                    "type": #type_str
                }))
            }
        }
    }).collect();

    let required_params: Vec<_> = params.iter()
        .filter(|p| p.required)
        .map(|p| &p.name)
        .collect();
    
    Ok(quote! {
        {
            let mut properties = ::serde_json::Map::new();
            #(
                properties.insert(#properties.0.to_string(), #properties.1);
            )*
            
            ::serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": [#(#required_params),*],
                "additionalProperties": false
            })
        }
    })
}

fn type_to_json_type(ty: &Type) -> Result<&'static str> {
    match ty {
        Type::Path(type_path) => {
            let ident = &type_path.path.segments.last()
                .ok_or_else(|| syn::Error::new_spanned(ty, "empty type path"))?
                .ident;
            
            match ident.to_string().as_str() {
                "String" | "str" => Ok("string"),
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => Ok("integer"),
                "f32" | "f64" => Ok("number"),
                "bool" => Ok("boolean"),
                "Vec" => Ok("array"),
                _ => Ok("object"), // Default to object for complex types
            }
        }
        _ => Ok("object"),
    }
}