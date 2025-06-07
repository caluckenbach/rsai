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
    
    // Validate that all docstring parameters exist as actual parameters
    validate_parameter_descriptions(&params, &param_descriptions, &input.sig)?;
    
    // Generate JSON schema for parameters
    let schema = generate_parameter_schema(&params)?;
    
    // Generate the wrapper struct name
    let wrapper_name = quote::format_ident!("{}Tool", 
        fn_name.to_string()
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
    
    // Check if function is async
    let is_async = input.sig.asyncness.is_some();
    
    // Generate the execution code
    let execute_impl = generate_execute_impl(&fn_name, &params, is_async)?;
    
    // Generate the complete implementation
    let expanded = quote! {
        #input
        
        #[derive(Clone)]
        pub struct #wrapper_name;
        
        impl ai::core::ToolFunction for #wrapper_name {
            fn schema(&self) -> ai::core::types::Tool {
                use ai::core::types::Tool;
                Tool {
                    name: #fn_name_str.to_string(),
                    description: #description,
                    parameters: #schema,
                    strict: Some(true),
                }
            }
            
            fn execute<'a>(&'a self, params: ::serde_json::Value) -> ai::core::types::BoxFuture<'a, Result<::serde_json::Value, ai::core::error::LlmError>> {
                use ai::core::{types::BoxFuture, error::LlmError};
                Box::pin(async move {
                    #execute_impl
                })
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

fn validate_parameter_descriptions(
    params: &[Parameter],
    param_descriptions: &std::collections::HashMap<String, String>,
    sig: &syn::Signature,
) -> Result<()> {
    let actual_param_names: std::collections::HashSet<String> = 
        params.iter().map(|p| p.name.clone()).collect();
    
    // Check for docstring parameters that don't exist in the function
    for docstring_param in param_descriptions.keys() {
        if !actual_param_names.contains(docstring_param) {
            return Err(syn::Error::new_spanned(
                sig,
                format!("Parameter '{}' found in docstring but not in function parameters", docstring_param)
            ));
        }
    }
    
    // Check for missing parameter descriptions in docstring
    for param in params {
        if !param_descriptions.contains_key(&param.name) {
            return Err(syn::Error::new_spanned(
                sig,
                format!("Parameter '{}' is missing description in docstring. Add: '{}: description'", param.name, param.name)
            ));
        }
    }
    
    Ok(())
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

fn generate_execute_impl(fn_name: &syn::Ident, params: &[Parameter], is_async: bool) -> Result<TokenStream> {
    let param_extractions = params.iter().map(|param| {
        let name = &param.name;
        let name_ident = quote::format_ident!("{}", name);
        let ty = &param.ty;
        
        if param.required {
            quote! {
                let #name_ident: #ty = params.get(#name)
                    .ok_or_else(|| LlmError::ToolExecution {
                        message: format!("Missing required parameter: {}", #name),
                        source: None,
                    })
                    .and_then(|v| ::serde_json::from_value(v.clone())
                        .map_err(|e| LlmError::ToolExecution {
                            message: format!("Invalid parameter '{}': {:?}", #name, e),
                            source: Some(Box::new(e)),
                        }))?;
            }
        } else {
            quote! {
                let #name_ident: Option<#ty> = params.get(#name)
                    .map(|v| ::serde_json::from_value(v.clone()))
                    .transpose()
                    .map_err(|e| LlmError::ToolExecution {
                        message: format!("Invalid parameter '{}': {:?}", #name, e),
                        source: Some(Box::new(e)),
                    })?;
            }
        }
    });
    
    let param_names: Vec<_> = params.iter().map(|p| quote::format_ident!("{}", p.name)).collect();
    
    let function_call = if is_async {
        quote! { #fn_name(#(#param_names),*).await }
    } else {
        quote! { #fn_name(#(#param_names),*) }
    };
    
    Ok(quote! {
        // Parse parameters from JSON
        let params = params.as_object()
            .ok_or_else(|| LlmError::ToolExecution {
                message: "Parameters must be an object".to_string(),
                source: None,
            })?;
        
        #(#param_extractions)*
        
        // Call the function
        let result = #function_call;
        
        // Convert result to JSON
        ::serde_json::to_value(result)
            .map_err(|e| LlmError::ToolExecution {
                message: "Failed to serialize result".to_string(),
                source: Some(Box::new(e)),
            })
    })
}