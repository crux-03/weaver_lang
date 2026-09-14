//! Instantiating a WTN document.
//!
//! Expansion happens once, at instantiation: inputs are bound and checked,
//! the document is evaluated, and what comes out is a plain [`Value`] the
//! host deserializes. Structural validation of that value is deliberately
//! not the language's job — `serde` already describes those types.

use std::collections::BTreeMap;

use super::{EvalContext, EvalOptions, Evaluator};
use crate::ast::doc::{InputDecl, InputType, ValueDoc};
use crate::ast::value::Value;
use crate::error::{EvalError, EvalErrorKind};
use crate::registry::Registry;

/// Parse and expand a WTN document in one step.
pub fn expand_value_doc(
    source: &str,
    inputs: &BTreeMap<String, Value>,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<Value, crate::RenderError> {
    let doc = crate::parse_value_doc(source).map_err(crate::RenderError::Parse)?;
    evaluate_value_doc(&doc, inputs, ctx, registry).map_err(crate::RenderError::Eval)
}

/// Expand a parsed document against a set of inputs.
///
/// Every declared input is bound before evaluation: the supplied value if
/// there is one, the declared default otherwise, and an error if neither.
/// The bound set is reachable from the document as the `input` scope —
/// `{{input:characters}}` — which the evaluator owns rather than the host,
/// so declared defaults are applied in one place.
///
/// ```rust
/// use std::collections::BTreeMap;
/// use weaver_lang::{Registry, SimpleContext, Value, expand_value_doc};
///
/// let mut ctx = SimpleContext::new();
/// let value = expand_value_doc(
///     r#"
/// #inputs
/// difficulty: enum("easy", "brutal") = "easy"
///
/// { mode: {{input:difficulty}} }
/// "#,
///     &BTreeMap::new(),
///     &mut ctx,
///     &Registry::new(),
/// )
/// .unwrap();
/// assert_eq!(value.to_json(), r#"{"mode":"easy"}"#);
/// ```
pub fn evaluate_value_doc(
    doc: &ValueDoc,
    inputs: &BTreeMap<String, Value>,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<Value, EvalError> {
    evaluate_value_doc_with_options(doc, inputs, ctx, registry, EvalOptions::new())
}

pub fn evaluate_value_doc_with_options(
    doc: &ValueDoc,
    inputs: &BTreeMap<String, Value>,
    ctx: &mut impl EvalContext,
    registry: &Registry,
    options: EvalOptions,
) -> Result<Value, EvalError> {
    let mut evaluator = Evaluator::new(options);
    let bound = bind_inputs(doc, inputs, &mut evaluator, ctx, registry)?;
    evaluator.inputs = Some(bound);
    evaluator.eval_doc_value(&doc.value, ctx, registry)
}

/// Bind every declared input, applying defaults and checking values.
fn bind_inputs(
    doc: &ValueDoc,
    supplied: &BTreeMap<String, Value>,
    evaluator: &mut Evaluator,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<BTreeMap<String, Value>, EvalError> {
    // A value for something the document never declared is a mistake worth
    // reporting: it is either a typo or a stale caller.
    for name in supplied.keys() {
        if !doc.inputs.iter().any(|decl| &decl.name == name) {
            return Err(EvalError::new(
                EvalErrorKind::UndefinedVariable,
                format!("no such input: {name}"),
            ));
        }
    }

    let mut bound = BTreeMap::new();
    for decl in &doc.inputs {
        let value = match supplied.get(&decl.name) {
            Some(value) => value.clone(),
            None => match &decl.default {
                Some(expr) => evaluator.eval_doc_value(expr, ctx, registry)?,
                None => {
                    return Err(EvalError::new(
                        EvalErrorKind::UndefinedVariable,
                        format!("missing required input: {} ({})", decl.name, decl.ty),
                    )
                    .with_span(decl.span));
                }
            },
        };

        check_value(&value, &decl.ty, decl, ctx, registry)?;
        bound.insert(decl.name.clone(), value);
    }
    Ok(bound)
}

/// Check one value against its declared type.
///
/// The language checks the shapes it named — string, number, bool, enum
/// membership, list-of — and hands `Ref<Kind>` to the host, which is the
/// only party that can say whether an id resolves to a live entity.
fn check_value(
    value: &Value,
    ty: &InputType,
    decl: &InputDecl,
    ctx: &impl EvalContext,
    registry: &Registry,
) -> Result<(), EvalError> {
    let mismatch =
        || Err(EvalError::type_error(&ty.to_string(), value.type_name()).with_span(decl.span));

    match ty {
        InputType::String if !matches!(value, Value::String(_)) => mismatch(),
        InputType::Number if !matches!(value, Value::Number(_)) => mismatch(),
        InputType::Bool if !matches!(value, Value::Bool(_)) => mismatch(),
        InputType::Enum(variants) => match value.as_string() {
            Some(got) if variants.iter().any(|v| v == got) => Ok(()),
            Some(got) => Err(EvalError::new(
                EvalErrorKind::TypeError,
                format!(
                    "input {}: {got:?} is not one of {}",
                    decl.name,
                    variants
                        .iter()
                        .map(|v| format!("{v:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .with_span(decl.span)),
            None => mismatch(),
        },
        InputType::List(inner) => match value.as_array() {
            Some(items) => {
                for item in items {
                    check_value(item, inner, decl, ctx, registry)?;
                }
                Ok(())
            }
            None => mismatch(),
        },
        InputType::Ref(kind) => {
            if !registry.has_kind(kind) {
                return Err(EvalError::new(
                    EvalErrorKind::UndefinedCallable,
                    format!("unknown entity kind: {kind}"),
                )
                .with_span(decl.span));
            }
            ctx.validate_input(kind, value).map_err(|e| {
                if e.span.is_none() {
                    e.with_span(decl.span)
                } else {
                    e
                }
            })
        }
        _ => Ok(()),
    }
}
