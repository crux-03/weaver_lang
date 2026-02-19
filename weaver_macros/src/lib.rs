use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, Pat, Type, parse_macro_input};

/// Derive a `WeaverProcessor` implementation from a function.
///
/// The function parameters become named properties. Type validation
/// is generated automatically from the parameter types.
///
/// # Attribute syntax
///
/// ```ignore
/// #[weaver_processor(namespace = "core.weaver", name = "wildcard")]
/// ```
///
/// # Supported parameter types
/// - `Value` — accepts any value, no validation
/// - `String` — validates the property is a string, passes the inner String
/// - `f64` — validates the property is a number, passes the inner f64
/// - `bool` — validates the property is a bool, passes the inner bool
/// - `Vec<Value>` — validates the property is an array, passes the inner Vec
///
/// # Example
/// ```ignore
/// #[weaver_processor(namespace = "core.weaver", name = "wildcard")]
/// fn wildcard(items: Vec<Value>) -> Result<Value, EvalError> {
///     // `items` is a Vec<Value> — the macro validated it was an array
///     let idx = rand::random::<usize>() % items.len();
///     Ok(items[idx].clone())
/// }
/// ```
#[proc_macro_attribute]
pub fn weaver_processor(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ProcessorArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let struct_name = format_ident!("{}Processor", to_pascal_case(&fn_name.to_string()));
    let proc_namespace = &args.namespace;
    let proc_name = &args.name;

    let mut param_extractions = Vec::new();
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut property_defs = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg
            && let Pat::Ident(ident) = &*pat_type.pat
        {
            let param_name = &ident.ident;
            let param_name_str = param_name.to_string();
            let ty = &*pat_type.ty;

            let (extraction, value_type, rust_type) = generate_extraction(&param_name_str, ty);
            param_extractions.push(extraction);
            param_names.push(param_name.clone());
            param_types.push(rust_type);
            property_defs.push(quote! {
                weaver_lang::registry::PropertyDef {
                    key: #param_name_str.to_string(),
                    expected_type: Some(#value_type),
                    required: true,
                }
            });
        }
    }

    let fn_body = &input_fn.block;

    let output = quote! {
        pub struct #struct_name;

        impl #struct_name {
            fn execute(#(#param_names: #param_types),*) -> Result<weaver_lang::Value, weaver_lang::EvalError> {
                #fn_body
            }
        }

        impl weaver_lang::registry::WeaverProcessor for #struct_name {
            fn call(
                &self,
                mut properties: std::collections::HashMap<String, weaver_lang::Value>,
            ) -> Result<weaver_lang::Value, weaver_lang::EvalError> {
                #(#param_extractions)*
                Self::execute(#(#param_names),*)
            }

            fn signature(&self) -> weaver_lang::registry::ProcessorSignature {
                weaver_lang::registry::ProcessorSignature {
                    namespace: #proc_namespace.to_string(),
                    name: #proc_name.to_string(),
                    properties: vec![#(#property_defs),*],
                }
            }
        }
    };

    output.into()
}

/// Derive a `WeaverCommand` implementation from a function.
///
/// The function parameters become positional arguments, except for two
/// reserved parameter names that receive system values:
///
/// - `ctx` — receives `&mut dyn EvalContext` (mutable access to state)
/// - `registry` — receives `&Registry` (access to other callables)
///
/// All other parameters are extracted from the positional argument list
/// in order, with automatic type validation.
///
/// # Attribute syntax
///
/// ```ignore
/// #[weaver_command(name = "set_var")]
/// ```
///
/// # Supported parameter types (for positional args)
/// - `Value` — accepts any value, no validation
/// - `String` — validates the arg is a string, passes the inner String
/// - `f64` — validates the arg is a number, passes the inner f64
/// - `bool` — validates the arg is a bool, passes the inner bool
/// - `Vec<Value>` — validates the arg is an array, passes the inner Vec
///
/// # Examples
///
/// Command that mutates context:
/// ```ignore
/// #[weaver_command(name = "set_var")]
/// fn set_var(key: String, value: Value, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
///     if let Some(pos) = key.find(':') {
///         ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
///     }
///     Ok(None)
/// }
/// ```
///
/// Pure command without context access:
/// ```ignore
/// #[weaver_command(name = "greet")]
/// fn greet(name: String) -> Result<Option<Value>, EvalError> {
///     Ok(Some(Value::String(format!("Hello, {name}!"))))
/// }
/// ```
#[proc_macro_attribute]
pub fn weaver_command(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as CommandArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let struct_name = format_ident!("{}Command", to_pascal_case(&fn_name.to_string()));
    let cmd_name = &args.name;

    let mut param_extractions = Vec::new();
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut param_defs = Vec::new();
    let mut has_ctx = false;
    let mut has_registry = false;
    let mut arg_index: usize = 0;

    for fn_arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = fn_arg
            && let Pat::Ident(ident) = &*pat_type.pat
        {
            let param_name = &ident.ident;
            let param_name_str = param_name.to_string();

            if param_name_str == "ctx" {
                has_ctx = true;
                param_names.push(param_name.clone());
                param_types.push(quote! { &mut dyn weaver_lang::EvalContext });
                continue;
            }

            if param_name_str == "registry" {
                has_registry = true;
                param_names.push(param_name.clone());
                param_types.push(quote! { &weaver_lang::Registry });
                continue;
            }

            let ty = &*pat_type.ty;
            let idx = arg_index;
            let (extraction, value_type, rust_type) =
                generate_arg_extraction(&param_name_str, ty, idx);
            param_extractions.push(extraction);
            param_names.push(param_name.clone());
            param_types.push(rust_type);
            param_defs.push(quote! {
                weaver_lang::registry::ParamDef {
                    name: #param_name_str.to_string(),
                    expected_type: Some(#value_type),
                    required: true,
                }
            });
            arg_index += 1;
        }
    }

    let fn_body = &input_fn.block;

    // Build the argument list for the execute call, inserting ctx/registry
    // in the right positions
    let mut call_args = Vec::new();
    for fn_arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = fn_arg
            && let Pat::Ident(ident) = &*pat_type.pat
        {
            let name = &ident.ident;
            let name_str = name.to_string();
            if name_str == "ctx" {
                call_args.push(quote! { ctx });
            } else if name_str == "registry" {
                call_args.push(quote! { registry });
            } else {
                call_args.push(quote! { #name });
            }
        }
    }

    // Suppress unused variable warnings when ctx/registry aren't requested
    let ctx_binding = if has_ctx {
        quote! {}
    } else {
        quote! { let _ = ctx; }
    };
    let registry_binding = if has_registry {
        quote! {}
    } else {
        quote! { let _ = registry; }
    };

    let output = quote! {
        pub struct #struct_name;

        impl #struct_name {
            fn execute(#(#param_names: #param_types),*) -> Result<Option<weaver_lang::Value>, weaver_lang::EvalError> {
                #fn_body
            }
        }

        impl weaver_lang::registry::WeaverCommand for #struct_name {
            fn call(
                &self,
                args: Vec<weaver_lang::Value>,
                ctx: &mut dyn weaver_lang::EvalContext,
                registry: &weaver_lang::Registry,
            ) -> Result<Option<weaver_lang::Value>, weaver_lang::EvalError> {
                #ctx_binding
                #registry_binding
                #(#param_extractions)*
                Self::execute(#(#call_args),*)
            }

            fn signature(&self) -> weaver_lang::registry::CommandSignature {
                weaver_lang::registry::CommandSignature {
                    name: #cmd_name.to_string(),
                    params: vec![#(#param_defs),*],
                }
            }
        }
    };

    output.into()
}

/// Generate extraction code for a positional command argument.
///
/// Returns (extraction_code, value_type_token, rust_type_token)
fn generate_arg_extraction(
    name: &str,
    ty: &Type,
    index: usize,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let ident = format_ident!("{}", name);
    let type_str = quote!(#ty).to_string().replace(' ', "");

    match type_str.as_str() {
        "Value" => (
            quote! {
                let #ident = args.get(#index).cloned().unwrap_or(weaver_lang::Value::None);
            },
            quote! { weaver_lang::registry::ValueType::Any },
            quote! { weaver_lang::Value },
        ),
        "String" => (
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::String(s)) => s.clone(),
                    Some(other) => return Err(weaver_lang::EvalError::type_error("string", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required argument at position {}: {}", #index, #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::String },
            quote! { String },
        ),
        "f64" => (
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::Number(n)) => *n,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("number", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required argument at position {}: {}", #index, #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Number },
            quote! { f64 },
        ),
        "bool" => (
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::Bool(b)) => *b,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("bool", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required argument at position {}: {}", #index, #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Bool },
            quote! { bool },
        ),
        "Vec<Value>" => (
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::Array(arr)) => arr.clone(),
                    Some(other) => return Err(weaver_lang::EvalError::type_error("array", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required argument at position {}: {}", #index, #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Array },
            quote! { Vec<weaver_lang::Value> },
        ),
        _ => (
            quote! {
                let #ident = args.get(#index).cloned().unwrap_or(weaver_lang::Value::None);
            },
            quote! { weaver_lang::registry::ValueType::Any },
            quote! { weaver_lang::Value },
        ),
    }
}

/// Returns (extraction_code, value_type_token, rust_type_token)
fn generate_extraction(
    name: &str,
    ty: &Type,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let ident = format_ident!("{}", name);
    let type_str = quote!(#ty).to_string().replace(' ', "");

    match type_str.as_str() {
        "Value" => (
            quote! {
                let #ident = properties.remove(#name).unwrap_or(weaver_lang::Value::None);
            },
            quote! { weaver_lang::registry::ValueType::Any },
            quote! { weaver_lang::Value },
        ),
        "String" => (
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::String(s)) => s,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("string", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required property: {}", #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::String },
            quote! { String },
        ),
        "f64" => (
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::Number(n)) => n,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("number", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required property: {}", #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Number },
            quote! { f64 },
        ),
        "bool" => (
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::Bool(b)) => b,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("bool", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required property: {}", #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Bool },
            quote! { bool },
        ),
        "Vec<Value>" => (
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::Array(arr)) => arr,
                    Some(other) => return Err(weaver_lang::EvalError::type_error("array", other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required property: {}", #name),
                    )),
                };
            },
            quote! { weaver_lang::registry::ValueType::Array },
            quote! { Vec<weaver_lang::Value> },
        ),
        _ => (
            quote! {
                let #ident = properties.remove(#name).unwrap_or(weaver_lang::Value::None);
            },
            quote! { weaver_lang::registry::ValueType::Any },
            quote! { weaver_lang::Value },
        ),
    }
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

// -- Attribute arg parsing -----------------------------------------------

struct ProcessorArgs {
    namespace: String,
    name: String,
}

impl syn::parse::Parse for ProcessorArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut namespace = None;
        let mut name = None;

        loop {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            let lit: syn::LitStr = input.parse()?;

            match ident.to_string().as_str() {
                "namespace" => namespace = Some(lit.value()),
                "name" => name = Some(lit.value()),
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unexpected key `{other}`, expected `namespace` or `name`"),
                    ));
                }
            }

            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }

        let namespace = namespace.ok_or_else(|| input.error("missing `namespace` attribute"))?;
        let name = name.ok_or_else(|| input.error("missing `name` attribute"))?;

        Ok(ProcessorArgs { namespace, name })
    }
}

struct CommandArgs {
    name: String,
}

impl syn::parse::Parse for CommandArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        if ident != "name" {
            return Err(syn::Error::new(ident.span(), "expected `name`"));
        }
        input.parse::<syn::Token![=]>()?;
        let lit: syn::LitStr = input.parse()?;
        Ok(CommandArgs { name: lit.value() })
    }
}
