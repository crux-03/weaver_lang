//! Signature metadata as seen by tooling.
//!
//! These tests exist because signature metadata used to be decorative — it
//! was read at registration for the name and never again, so nothing caught
//! it drifting from real behavior. Editor autocompletion reads it now, so
//! each fact it reports has to be pinned against the code that implements it.

use macros::{weaver_command, weaver_processor};
use weaver_lang::registry::ValueType;
// The macros rewrite the annotated fn's own types, so some of these are only
// referenced in source the macro discards.
#[allow(unused_imports)]
use weaver_lang::{EvalContext, EvalError, Registry, Value};

/// Slice a string.
///
/// `length` runs to the end of the string when omitted.
#[weaver_processor(
    namespace = "text",
    name = "substr",
    returns = "string",
    params(text = "The source text", start = "Zero-based start index")
)]
fn text_substr(text: String, start: f64, length: Option<f64>) -> Result<Value, EvalError> {
    let start = start as usize;
    let result: String = match length {
        Some(len) => text.chars().skip(start).take(len as usize).collect(),
        None => text.chars().skip(start).collect(),
    };
    Ok(Value::String(result))
}

/// Set a variable in a writable scope.
#[weaver_command(name = "set_var", returns = "none")]
fn set_var(
    key: String,
    value: Value,
    ctx: &mut dyn EvalContext,
) -> Result<Option<Value>, EvalError> {
    if let Some(pos) = key.find(':') {
        ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
    }
    Ok(None)
}

/// Uppercase a string.
#[weaver_command(name = "shout", returns = "string")]
fn shout(text: String) -> Result<Option<Value>, EvalError> {
    Ok(Some(Value::String(text.to_uppercase())))
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_processor(TextSubstrProcessor);
    registry.register_command(SetVarCommand);
    registry.register_command(ShoutCommand);
    registry
}

/// The drift this feature exists to close: `length` is optional and numeric,
/// and the signature has to say so. The old macro hardcoded `required: true`
/// and typed it `Any`, so an editor would have demanded a third argument of
/// the wrong type.
#[test]
fn optional_param_is_reported_optional_and_typed() {
    let sig = registry().processor_signature("text", "substr").unwrap();

    let length = sig.properties.iter().find(|p| p.key == "length").unwrap();
    assert!(!length.required);
    assert_eq!(length.expected_type, Some(ValueType::Number));

    let text = sig.properties.iter().find(|p| p.key == "text").unwrap();
    assert!(text.required);
    assert_eq!(text.expected_type, Some(ValueType::String));
}

/// An optional param the signature calls optional must actually be omittable
/// at runtime — the claim and the extraction code agree or the metadata lies.
#[test]
fn optional_param_can_be_omitted_at_runtime() {
    use std::collections::HashMap;

    let registry = registry();

    let mut props = HashMap::new();
    props.insert("text".to_string(), Value::String("hello world".to_string()));
    props.insert("start".to_string(), Value::Number(6.0));
    let out = registry.call_processor("text", "substr", props).unwrap();
    assert_eq!(out, Value::String("world".to_string()));

    // ...and still honors the type when supplied.
    let mut props = HashMap::new();
    props.insert("text".to_string(), Value::String("hello world".to_string()));
    props.insert("start".to_string(), Value::Number(0.0));
    props.insert("length".to_string(), Value::Number(5.0));
    let out = registry.call_processor("text", "substr", props).unwrap();
    assert_eq!(out, Value::String("hello".to_string()));

    // A wrong-typed optional is still an error, not a silent None.
    let mut props = HashMap::new();
    props.insert("text".to_string(), Value::String("hello".to_string()));
    props.insert("start".to_string(), Value::Number(0.0));
    props.insert("length".to_string(), Value::Bool(true));
    assert!(registry.call_processor("text", "substr", props).is_err());
}

#[test]
fn doc_comment_becomes_the_description() {
    let registry = registry();

    let sig = registry.processor_signature("text", "substr").unwrap();
    assert_eq!(
        sig.description,
        "Slice a string.\n\n`length` runs to the end of the string when omitted."
    );

    let sig = registry.command_signature("set_var").unwrap();
    assert_eq!(sig.description, "Set a variable in a writable scope.");
}

#[test]
fn param_descriptions_come_from_the_params_attribute() {
    let sig = registry().processor_signature("text", "substr").unwrap();

    let text = sig.properties.iter().find(|p| p.key == "text").unwrap();
    assert_eq!(text.description, "The source text");

    // Undescribed params are empty, not absent.
    let length = sig.properties.iter().find(|p| p.key == "length").unwrap();
    assert_eq!(length.description, "");
}

/// `mutates_context` is derived from the presence of a `ctx` parameter, so an
/// editor can badge state-changing commands without the author declaring it.
#[test]
fn mutates_context_is_derived_from_the_ctx_param() {
    let registry = registry();

    assert!(
        registry
            .command_signature("set_var")
            .unwrap()
            .mutates_context
    );
    assert!(!registry.command_signature("shout").unwrap().mutates_context);
}

#[test]
fn return_types_are_reported() {
    let registry = registry();

    assert_eq!(
        registry
            .processor_signature("text", "substr")
            .unwrap()
            .returns,
        ValueType::String
    );
    assert_eq!(
        registry.command_signature("set_var").unwrap().returns,
        ValueType::None
    );
    assert_eq!(
        registry.command_signature("shout").unwrap().returns,
        ValueType::String
    );
}

#[test]
fn registry_enumerates_everything_registered() {
    let registry = registry();

    let mut commands: Vec<String> = registry
        .command_signatures()
        .into_iter()
        .map(|s| s.name)
        .collect();
    commands.sort();
    assert_eq!(commands, vec!["set_var", "shout"]);

    let processors: Vec<String> = registry
        .processor_signatures()
        .into_iter()
        .map(|s| format!("{}.{}", s.namespace, s.name))
        .collect();
    assert_eq!(processors, vec!["text.substr"]);

    assert!(registry.command_signature("nonexistent").is_none());
    assert!(
        registry
            .processor_signature("text", "nonexistent")
            .is_none()
    );
}

/// The host ships these to a frontend, so the wire shape is part of the API.
#[cfg(feature = "serde")]
#[test]
fn signatures_serialize_for_the_editor() {
    let sig = registry().processor_signature("text", "substr").unwrap();
    let json = serde_json::to_value(&sig).unwrap();

    assert_eq!(json["namespace"], "text");
    assert_eq!(json["name"], "substr");
    assert_eq!(json["returns"], "string");

    let length = json["properties"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["key"] == "length")
        .unwrap();
    assert_eq!(length["expected_type"], "number");
    assert_eq!(length["required"], false);
}

#[cfg(feature = "serde")]
#[test]
fn value_type_none_survives_a_round_trip() {
    let json = serde_json::to_value(ValueType::None).unwrap();
    assert_eq!(json, serde_json::json!("none"));
    assert_eq!(
        serde_json::from_value::<ValueType>(json).unwrap(),
        ValueType::None
    );
}
