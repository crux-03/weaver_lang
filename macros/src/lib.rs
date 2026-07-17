use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{FnArg, ItemFn, Pat, Type, parse_macro_input};

/// Derive a `WeaverProcessor` implementation from a function.
///
/// The function parameters become named properties. Both the type-validation
/// code and the signature an editor reads are generated from those same
/// parameters, so the two cannot disagree.
///
/// # Attribute syntax
///
/// ```ignore
/// #[weaver_processor(
///     namespace = "core.weaver",
///     name = "wildcard",
///     returns = "any",
///     params(items = "Candidates to choose from"),
/// )]
/// ```
///
/// `namespace`, `name` and `returns` are required. `returns` cannot be
/// inferred — every processor body returns `Result<Value, _>` — so it must be
/// declared: one of `string`, `number`, `bool`, `array`, `none`, `any`.
///
/// The function's `///` doc comment becomes the processor's description.
/// `params(..)` supplies per-property descriptions and is optional; a key
/// that matches no parameter is a compile error.
///
/// # Supported parameter types
/// - `Value` — accepts any value, no validation
/// - `String` — validates the property is a string, passes the inner String
/// - `f64` — validates the property is a number, passes the inner f64
/// - `bool` — validates the property is a bool, passes the inner bool
/// - `Vec<Value>` — validates the property is an array, passes the inner Vec
/// - `Option<T>` — any of the above, but optional: an absent property (or an
///   explicit `none`) passes `None` instead of erroring. This is the only way
///   to declare an optional property, and it is what makes `required: false`
///   in the generated signature true by construction.
///
/// # Example
/// ```ignore
/// /// Pick one item at random.
/// #[weaver_processor(namespace = "core.weaver", name = "wildcard", returns = "any")]
/// fn wildcard(items: Vec<Value>) -> Result<Value, EvalError> {
///     // `items` is a Vec<Value> — the macro validated it was an array
///     let idx = rand::random::<usize>() % items.len();
///     Ok(items[idx].clone())
/// }
/// ```
#[proc_macro_attribute]
pub fn weaver_processor(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MacroArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let call_site = fn_name.span();
    let struct_name = format_ident!("{}Processor", to_pascal_case(&fn_name.to_string()));

    let Some((proc_namespace, _)) = args.namespace.clone() else {
        return syn::Error::new(call_site, "missing `namespace` attribute")
            .to_compile_error()
            .into();
    };
    let Some(proc_name) = args.name.clone() else {
        return syn::Error::new(call_site, "missing `name` attribute")
            .to_compile_error()
            .into();
    };
    let returns = match returns_token(args.returns.clone(), call_site) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };
    let description = extract_doc(&input_fn.attrs);

    let mut param_extractions = Vec::new();
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut property_defs = Vec::new();
    let mut actual_params = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg
            && let Pat::Ident(ident) = &*pat_type.pat
        {
            let param_name = &ident.ident;
            let param_name_str = param_name.to_string();
            let ty = &*pat_type.ty;

            let (extraction, value_type, rust_type, required) =
                generate_extraction(&param_name_str, ty);
            let param_doc = args
                .params
                .get(&param_name_str)
                .cloned()
                .unwrap_or_default();
            actual_params.push(param_name_str.clone());
            param_extractions.push(extraction);
            param_names.push(param_name.clone());
            param_types.push(rust_type);
            property_defs.push(quote! {
                weaver_lang::registry::PropertyDef {
                    key: #param_name_str.to_string(),
                    description: #param_doc.to_string(),
                    expected_type: Some(#value_type),
                    required: #required,
                }
            });
        }
    }

    if let Err(e) = check_unknown_params(&args.params, &actual_params, call_site) {
        return e.to_compile_error().into();
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
                    description: #description.to_string(),
                    returns: #returns,
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
/// Taking `ctx` also sets `mutates_context` on the generated signature, so
/// tooling can mark which commands can change evaluation state.
///
/// # Attribute syntax
///
/// ```ignore
/// #[weaver_command(name = "set_var", returns = "none", params(key = "scope:name"))]
/// ```
///
/// `name` and `returns` are required; commands have no `namespace`. Use
/// `returns = "none"` for a command that exists only for its effect. The
/// function's `///` doc comment becomes the command's description.
///
/// # Supported parameter types (for positional args)
/// - `Value` — accepts any value, no validation
/// - `String` — validates the arg is a string, passes the inner String
/// - `f64` — validates the arg is a number, passes the inner f64
/// - `bool` — validates the arg is a bool, passes the inner bool
/// - `Vec<Value>` — validates the arg is an array, passes the inner Vec
/// - `Option<T>` — any of the above, but optional: a missing trailing arg (or
///   an explicit `none`) passes `None` instead of erroring, and the generated
///   signature reports `required: false`.
///
/// # Examples
///
/// Command that mutates context:
/// ```ignore
/// /// Set a variable in a writable scope.
/// #[weaver_command(name = "set_var", returns = "none")]
/// fn set_var(key: String, value: Value, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
///     if let Some(pos) = key.find(':') {
///         ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
///     }
///     Ok(None)
/// }
/// ```
///
/// Pure command with an optional argument:
/// ```ignore
/// /// Add `amount` to a numeric variable.
/// #[weaver_command(name = "inc_var", returns = "none")]
/// fn inc_var(key: String, amount: Option<f64>, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
///     let amount = amount.unwrap_or(1.0);
///     // ...
///     Ok(None)
/// }
/// ```
#[proc_macro_attribute]
pub fn weaver_command(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MacroArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let call_site = fn_name.span();
    let struct_name = format_ident!("{}Command", to_pascal_case(&fn_name.to_string()));

    if let Some((_, span)) = &args.namespace {
        return syn::Error::new(
            *span,
            "commands have no namespace — remove `namespace`, or use \
             `#[weaver_processor]` if you meant to declare a processor",
        )
        .to_compile_error()
        .into();
    }
    let Some(cmd_name) = args.name.clone() else {
        return syn::Error::new(call_site, "missing `name` attribute")
            .to_compile_error()
            .into();
    };
    let returns = match returns_token(args.returns.clone(), call_site) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };
    let description = extract_doc(&input_fn.attrs);

    let mut param_extractions = Vec::new();
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut param_defs = Vec::new();
    let mut actual_params = Vec::new();
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
            let (extraction, value_type, rust_type, required) =
                generate_arg_extraction(&param_name_str, ty, idx);
            let param_doc = args
                .params
                .get(&param_name_str)
                .cloned()
                .unwrap_or_default();
            actual_params.push(param_name_str.clone());
            param_extractions.push(extraction);
            param_names.push(param_name.clone());
            param_types.push(rust_type);
            param_defs.push(quote! {
                weaver_lang::registry::ParamDef {
                    name: #param_name_str.to_string(),
                    description: #param_doc.to_string(),
                    expected_type: Some(#value_type),
                    required: #required,
                }
            });
            arg_index += 1;
        }
    }

    if let Err(e) = check_unknown_params(&args.params, &actual_params, call_site) {
        return e.to_compile_error().into();
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
                    description: #description.to_string(),
                    returns: #returns,
                    params: vec![#(#param_defs),*],
                    mutates_context: #has_ctx,
                }
            }
        }
    };

    output.into()
}

/// The value kind a parameter accepts, independent of its optionality.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Any,
    String,
    Number,
    Bool,
    Array,
}

impl Kind {
    /// The `Value` variant this kind unwraps, and the error label used when
    /// the runtime value is something else.
    fn variant(self) -> (proc_macro2::TokenStream, &'static str) {
        match self {
            Kind::String => (quote! { String }, "string"),
            Kind::Number => (quote! { Number }, "number"),
            Kind::Bool => (quote! { Bool }, "bool"),
            Kind::Array => (quote! { Array }, "array"),
            Kind::Any => (quote! { None }, "any"), // unused; Any never unwraps
        }
    }

    fn value_type(self) -> proc_macro2::TokenStream {
        match self {
            Kind::Any => quote! { weaver_lang::registry::ValueType::Any },
            Kind::String => quote! { weaver_lang::registry::ValueType::String },
            Kind::Number => quote! { weaver_lang::registry::ValueType::Number },
            Kind::Bool => quote! { weaver_lang::registry::ValueType::Bool },
            Kind::Array => quote! { weaver_lang::registry::ValueType::Array },
        }
    }

    fn rust_type(self) -> proc_macro2::TokenStream {
        match self {
            Kind::Any => quote! { weaver_lang::Value },
            Kind::String => quote! { String },
            Kind::Number => quote! { f64 },
            Kind::Bool => quote! { bool },
            Kind::Array => quote! { Vec<weaver_lang::Value> },
        }
    }
}

/// Classify a parameter type into its kind and whether it is optional.
///
/// `Option<T>` is the only way to declare an optional parameter, which is
/// what keeps `required` in the generated signature honest: it is read off
/// the same type that drives the extraction code below, so the two cannot
/// disagree. An unrecognized type degrades to `Any`, matching the previous
/// behavior.
fn classify(ty: &Type) -> (Kind, bool) {
    let type_str = quote!(#ty).to_string().replace(' ', "");
    let (inner, optional) = match type_str
        .strip_prefix("Option<")
        .and_then(|s| s.strip_suffix('>'))
    {
        Some(inner) => (inner.to_string(), true),
        None => (type_str, false),
    };

    let kind = match inner.as_str() {
        "String" => Kind::String,
        "f64" => Kind::Number,
        "bool" => Kind::Bool,
        "Vec<Value>" => Kind::Array,
        _ => Kind::Any,
    };

    (kind, optional)
}

/// Generate extraction code for a positional command argument.
///
/// Returns (extraction_code, value_type_token, rust_type_token, required).
fn generate_arg_extraction(
    name: &str,
    ty: &Type,
    index: usize,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    bool,
) {
    let ident = format_ident!("{}", name);
    let (kind, optional) = classify(ty);
    let value_type = kind.value_type();
    let rust_type = kind.rust_type();

    let extraction = match (kind, optional) {
        // Bare `Value`: anything, absent included.
        (Kind::Any, false) => quote! {
            let #ident = args.get(#index).cloned().unwrap_or(weaver_lang::Value::None);
        },
        // `Option<Value>`: absent and explicit `none` both collapse to None.
        (Kind::Any, true) => quote! {
            let #ident = args
                .get(#index)
                .cloned()
                .filter(|v| !matches!(v, weaver_lang::Value::None));
        },
        (kind, false) => {
            let (variant, label) = kind.variant();
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::#variant(v)) => v.clone(),
                    Some(other) => return Err(weaver_lang::EvalError::type_error(#label, other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required argument at position {}: {}", #index, #name),
                    )),
                };
            }
        }
        (kind, true) => {
            let (variant, label) = kind.variant();
            quote! {
                let #ident = match args.get(#index) {
                    Some(weaver_lang::Value::#variant(v)) => Some(v.clone()),
                    Some(weaver_lang::Value::None) | None => None,
                    Some(other) => return Err(weaver_lang::EvalError::type_error(#label, other.type_name())),
                };
            }
        }
    };

    let rust_type = if optional {
        quote! { Option<#rust_type> }
    } else {
        rust_type
    };

    (extraction, value_type, rust_type, !optional)
}

/// Generate extraction code for a named processor property.
///
/// Returns (extraction_code, value_type_token, rust_type_token, required).
fn generate_extraction(
    name: &str,
    ty: &Type,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    bool,
) {
    let ident = format_ident!("{}", name);
    let (kind, optional) = classify(ty);
    let value_type = kind.value_type();
    let rust_type = kind.rust_type();

    let extraction = match (kind, optional) {
        (Kind::Any, false) => quote! {
            let #ident = properties.remove(#name).unwrap_or(weaver_lang::Value::None);
        },
        (Kind::Any, true) => quote! {
            let #ident = properties
                .remove(#name)
                .filter(|v| !matches!(v, weaver_lang::Value::None));
        },
        (kind, false) => {
            let (variant, label) = kind.variant();
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::#variant(v)) => v,
                    Some(other) => return Err(weaver_lang::EvalError::type_error(#label, other.type_name())),
                    None => return Err(weaver_lang::EvalError::new(
                        weaver_lang::EvalErrorKind::TypeError,
                        format!("missing required property: {}", #name),
                    )),
                };
            }
        }
        (kind, true) => {
            let (variant, label) = kind.variant();
            quote! {
                let #ident = match properties.remove(#name) {
                    Some(weaver_lang::Value::#variant(v)) => Some(v),
                    Some(weaver_lang::Value::None) | None => None,
                    Some(other) => return Err(weaver_lang::EvalError::type_error(#label, other.type_name())),
                };
            }
        }
    };

    let rust_type = if optional {
        quote! { Option<#rust_type> }
    } else {
        rust_type
    };

    (extraction, value_type, rust_type, !optional)
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

/// Parsed `#[weaver_command]` / `#[weaver_processor]` arguments.
///
/// One parser serves both; each macro validates `namespace` itself, since
/// processors require it and commands reject it.
struct MacroArgs {
    namespace: Option<(String, proc_macro2::Span)>,
    name: Option<String>,
    returns: Option<(String, proc_macro2::Span)>,
    /// Per-parameter descriptions from `params(key = "...")`.
    params: HashMap<String, String>,
}

impl syn::parse::Parse for MacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut args = MacroArgs {
            namespace: None,
            name: None,
            returns: None,
            params: HashMap::new(),
        };

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;

            // `params(a = "...", b = "...")` is the one nested form.
            if ident == "params" {
                let content;
                syn::parenthesized!(content in input);
                while !content.is_empty() {
                    let key: syn::Ident = content.parse()?;
                    content.parse::<syn::Token![=]>()?;
                    let lit: syn::LitStr = content.parse()?;
                    args.params.insert(key.to_string(), lit.value());
                    if content.is_empty() {
                        break;
                    }
                    content.parse::<syn::Token![,]>()?;
                }
            } else {
                input.parse::<syn::Token![=]>()?;
                let lit: syn::LitStr = input.parse()?;
                match ident.to_string().as_str() {
                    "namespace" => args.namespace = Some((lit.value(), ident.span())),
                    "name" => args.name = Some(lit.value()),
                    "returns" => args.returns = Some((lit.value(), lit.span())),
                    other => {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "unexpected key `{other}`, expected `namespace`, `name`, \
                                 `returns` or `params(..)`"
                            ),
                        ));
                    }
                }
            }

            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }

        Ok(args)
    }
}

/// Map a `returns = "..."` string onto a `ValueType` variant.
///
/// Deliberately has no default: a return type cannot be inferred from the
/// function signature (every processor body returns `Result<Value, _>`), so
/// omitting it would silently produce a wrong `Any`.
fn returns_token(
    returns: Option<(String, proc_macro2::Span)>,
    call_site: proc_macro2::Span,
) -> syn::Result<proc_macro2::TokenStream> {
    let (value, span) = returns.ok_or_else(|| {
        syn::Error::new(
            call_site,
            "missing `returns` — declare what this callable evaluates to, e.g. \
             `returns = \"string\"`. Valid: string, number, bool, array, none, any",
        )
    })?;

    let variant = match value.as_str() {
        "string" => quote! { String },
        "number" => quote! { Number },
        "bool" => quote! { Bool },
        "array" => quote! { Array },
        "none" => quote! { None },
        "any" => quote! { Any },
        other => {
            return Err(syn::Error::new(
                span,
                format!(
                    "unknown return type `{other}`. Valid: string, number, bool, \
                     array, none, any"
                ),
            ));
        }
    };

    Ok(quote! { weaver_lang::registry::ValueType::#variant })
}

/// Concatenate a function's `///` doc comment into a description string.
///
/// The whole comment is kept, not just the summary line — editors render it
/// as markdown on hover, and the first line is the summary by convention.
fn extract_doc(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    // Doc comments carry a leading space from `/// `; strip exactly that one
    // so indented code blocks inside the comment survive.
    lines
        .iter()
        .map(|l| l.strip_prefix(' ').unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Error out on `params(..)` keys that match no actual parameter — almost
/// always a typo or a renamed argument, and silently ignoring it would leave
/// the editor showing a description for a parameter that no longer exists.
fn check_unknown_params(
    declared: &HashMap<String, String>,
    actual: &[String],
    call_site: proc_macro2::Span,
) -> syn::Result<()> {
    for key in declared.keys() {
        if !actual.iter().any(|p| p == key) {
            return Err(syn::Error::new(
                call_site,
                format!(
                    "`params({key} = ..)` does not match any parameter of this \
                     function. Found: {}",
                    actual.join(", ")
                ),
            ));
        }
    }
    Ok(())
}
