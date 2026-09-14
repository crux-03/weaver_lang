//! Property markers are optional on both callables.
//!
//! A processor's `call` takes named properties and a command's takes a
//! positional list, but that is an implementation detail of the trait, not
//! something a template should have to know. Either may be written either
//! way; the registry maps one onto the other using the declared signature.

use macros::{weaver_command, weaver_processor};
// The macro rewrites the annotated fn, so the error type reads as unused
// here even though the signatures below name it — as in `macro_tests.rs`.
#[allow(unused_imports)]
use weaver_lang::EvalError;
use weaver_lang::{
    ClosureCommand, ClosureProcessor, EvalOptions, Registry, SimpleContext, Value, parse, render,
    render_with_options,
};

#[weaver_processor(namespace = "t", name = "repeat", returns = "string")]
fn repeat_text(text: String, count: f64) -> Result<Value, EvalError> {
    Ok(Value::String(text.repeat(count.round() as usize)))
}

#[weaver_command(name = "join3", returns = "string")]
fn join3(a: Value, b: Value, c: Value) -> Result<Option<Value>, EvalError> {
    Ok(Some(Value::String(format!(
        "{}|{}|{}",
        a.to_output_string(),
        b.to_output_string(),
        c.to_output_string()
    ))))
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_processor(RepeatTextProcessor);
    registry.register_command(Join3Command);
    registry
}

fn eval(source: &str) -> String {
    let mut ctx = SimpleContext::new();
    render(source, &mut ctx, &registry()).unwrap()
}

fn eval_err(source: &str) -> String {
    let mut ctx = SimpleContext::new();
    match render(source, &mut ctx, &registry()) {
        Err(e) => format!("{e}"),
        Ok(out) => panic!("expected an error, got {out:?}"),
    }
}

fn parse_err(source: &str) -> String {
    parse(source).unwrap_err()[0].message.clone()
}

// ── Either form, either callable ────────────────────────────────────────

#[test]
fn a_processor_takes_positional_arguments() {
    assert_eq!(eval(r#"@[t.repeat(text: "ab", count: 2)]"#), "abab");
    assert_eq!(eval(r#"@[t.repeat("ab", 2)]"#), "abab");
}

#[test]
fn a_command_takes_named_arguments() {
    assert_eq!(eval(r#"$[join3(1, 2, 3)]"#), "1|2|3");
    assert_eq!(eval(r#"$[join3(a: 1, b: 2, c: 3)]"#), "1|2|3");
}

#[test]
fn named_arguments_may_be_written_in_any_order() {
    assert_eq!(eval(r#"@[t.repeat(count: 2, text: "ab")]"#), "abab");
    assert_eq!(eval(r#"$[join3(c: 3, a: 1, b: 2)]"#), "1|2|3");
}

#[test]
fn the_two_forms_may_be_mixed_positional_first() {
    assert_eq!(eval(r#"@[t.repeat("ab", count: 3)]"#), "ababab");
    assert_eq!(eval(r#"$[join3(1, c: 3, b: 2)]"#), "1|2|3");
}

#[test]
fn a_named_argument_may_skip_an_earlier_one() {
    // `b` was never supplied, so the command sees it as absent — the same
    // thing it sees for a trailing argument nobody passed.
    assert_eq!(eval(r#"$[join3(a: 1, c: 3)]"#), "1||3");
    assert_eq!(eval(r#"$[join3(c: 3)]"#), "||3");
}

// ── What the parser rejects ─────────────────────────────────────────────

#[test]
fn a_positional_argument_after_a_named_one_is_a_parse_error() {
    for source in [r#"@[t.repeat(text: "ab", 2)]"#, r#"$[join3(a: 1, 2)]"#] {
        assert!(
            parse_err(source).contains("positional argument after a named one"),
            "for {source}"
        );
    }
}

#[test]
fn a_name_written_twice_is_a_parse_error() {
    assert!(
        parse_err(r#"@[t.repeat(text: "a", text: "b")]"#).contains("argument given twice: text")
    );
    assert!(parse_err(r#"$[join3(a: 1, a: 2)]"#).contains("argument given twice: a"));
}

// ── What the registry rejects ───────────────────────────────────────────

#[test]
fn a_positional_argument_colliding_with_a_named_one_is_an_error() {
    // Position 1 is `text`, so naming it again fills the same slot twice.
    assert!(eval_err(r#"@[t.repeat("ab", text: "cd")]"#).contains("given text twice"));
    assert!(eval_err(r#"$[join3(1, a: 2)]"#).contains("given a twice"));
}

#[test]
fn too_many_positional_arguments_for_a_processor_are_reported() {
    let err = eval_err(r#"@[t.repeat("ab", 2, 3)]"#);
    assert!(err.contains("declares 2 properties"), "{err}");
    assert!(err.contains("position 3"), "{err}");
}

#[test]
fn a_name_the_command_does_not_declare_is_reported() {
    let err = eval_err(r#"$[join3(d: 1)]"#);
    assert!(err.contains("has no parameter named d"), "{err}");
}

// ── Callables that describe nothing ─────────────────────────────────────

#[test]
fn an_undeclared_callable_still_works_the_way_it_always_did() {
    // A closure registered without `.property()`/`.param()` has no
    // signature to map through, so only its own form is available.
    let mut registry = Registry::new();
    registry.register_processor(ClosureProcessor::new(
        "c",
        "echo",
        |props: std::collections::HashMap<String, Value>| {
            Ok(props.get("v").cloned().unwrap_or(Value::None))
        },
    ));
    registry.register_command(ClosureCommand::new("first", |args: Vec<Value>| {
        Ok(Some(args.first().cloned().unwrap_or(Value::None)))
    }));

    let mut ctx = SimpleContext::new();
    assert_eq!(render("@[c.echo(v: 1)]", &mut ctx, &registry).unwrap(), "1");
    assert_eq!(render("$[first(1, 2)]", &mut ctx, &registry).unwrap(), "1");

    // The other form has nothing to match against, and says so.
    let err = render("@[c.echo(1)]", &mut ctx, &registry).unwrap_err();
    assert!(format!("{err}").contains("declares no properties"), "{err}");
    let err = render("$[first(v: 1)]", &mut ctx, &registry).unwrap_err();
    assert!(format!("{err}").contains("declares no parameters"), "{err}");
}

// ── The colon rule ──────────────────────────────────────────────────────

#[test]
fn a_leading_name_colon_is_always_a_marker() {
    // `local:` at the top level of an argument list names an argument; it
    // is not the reference `local:text`. Interpolate or parenthesise to
    // pass one.
    let mut ctx = SimpleContext::new();
    ctx.set("local", "text", "ab");

    assert_eq!(
        render(r#"@[t.repeat({{local:text}}, 2)]"#, &mut ctx, &registry()).unwrap(),
        "abab"
    );
    assert_eq!(
        render(r#"@[t.repeat((local:text), 2)]"#, &mut ctx, &registry()).unwrap(),
        "abab"
    );

    // Nested inside a literal the colon is a reference, not a marker.
    let mut registry = registry();
    registry.register_processor(
        ClosureProcessor::new(
            "c",
            "count",
            |p: std::collections::HashMap<String, Value>| {
                Ok(Value::Number(
                    p.get("of")
                        .and_then(|v| v.as_array())
                        .map_or(0, <[Value]>::len) as f64,
                ))
            },
        )
        .property("of", weaver_lang::registry::ValueType::Array, true),
    );
    assert_eq!(
        render("@[c.count([local:text, local:text])]", &mut ctx, &registry).unwrap(),
        "2"
    );
}

// ── Everything else still holds ─────────────────────────────────────────

#[test]
fn separators_work_the_same_in_both_forms() {
    assert_eq!(eval("@[t.repeat(\n  \"ab\"\n  2\n)]"), "abab");
    assert_eq!(
        eval("@[t.repeat(\n  text: \"ab\",\n  count: 2,\n)]"),
        "abab"
    );
    assert_eq!(eval("$[join3(1, 2, 3,)]"), "1|2|3");
}

#[test]
fn lenient_output_keeps_the_form_that_was_written() {
    let mut ctx = SimpleContext::new();
    let opts = EvalOptions::new().lenient(true);
    let out = render_with_options(
        r#"@[gone.missing("a", k: 1)] $[nope(1, k: 2)]"#,
        &mut ctx,
        &Registry::new(),
        opts,
    )
    .unwrap();
    assert_eq!(out, "@[gone.missing(..., k: ...)] $[nope(..., k: ...)]");
}
