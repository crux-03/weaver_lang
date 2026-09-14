//! Object literals, indexing, and expressions in `{{ }}`.
//!
//! Three changes that only make sense together: an object literal is a
//! value the language can now *write* rather than only read, indexing
//! reaches into both halves of a structured value, and `{{ }}` holds the
//! expression that combines them instead of a bare name.

use std::cell::RefCell;
use std::collections::BTreeMap;

use weaver_lang::{
    EvalContext, EvalError, EvalErrorKind, EvalOptions, PathSegment, Registry, SimpleContext,
    Value, parse, render, render_with_options,
};

fn eval(source: &str) -> String {
    let mut ctx = SimpleContext::new();
    render(source, &mut ctx, &Registry::new()).unwrap()
}

fn eval_in(source: &str, ctx: &mut SimpleContext) -> String {
    render(source, ctx, &Registry::new()).unwrap()
}

fn eval_err(source: &str) -> EvalError {
    let mut ctx = SimpleContext::new();
    match render(source, &mut ctx, &Registry::new()) {
        Err(weaver_lang::RenderError::Eval(e)) => e,
        other => panic!("expected an eval error, got {other:?}"),
    }
}

fn alice() -> Value {
    Value::object([
        ("name", Value::String("Alice".into())),
        (
            "gear",
            Value::Array(vec!["sword".into(), "shield".into(), "torch".into()]),
        ),
        (
            "stats",
            Value::object([("hp", Value::Number(10.0)), ("mp", Value::Number(3.0))]),
        ),
    ])
}

// ── Object literals ─────────────────────────────────────────────────────

#[test]
fn an_object_literal_renders_as_compact_json() {
    assert_eq!(
        eval(r#"{{ {name: "Alice", hp: 10} }}"#),
        r#"{"hp":10,"name":"Alice"}"#
    );
}

#[test]
fn keys_may_be_quoted_so_json_parses_as_written() {
    assert_eq!(
        eval(r#"{{ {"needs quoting": 1, plain: 2} }}"#),
        r#"{"needs quoting":1,"plain":2}"#
    );
}

#[test]
fn an_empty_object_is_falsy_and_a_populated_one_is_truthy() {
    assert_eq!(eval("{# if {} #}yes{# else #}no{# endif #}"), "no");
    assert_eq!(eval("{# if {a: 1} #}yes{# else #}no{# endif #}"), "yes");
}

#[test]
fn entries_hold_arbitrary_expressions() {
    let mut ctx = SimpleContext::new();
    ctx.set("global", "base", 5i64);
    assert_eq!(
        eval_in(
            r#"{{ {total: global:base * 2, tag: "a" + "b"} }}"#,
            &mut ctx
        ),
        r#"{"tag":"ab","total":10}"#
    );
}

#[test]
fn objects_nest_in_arrays_and_arrays_in_objects() {
    assert_eq!(
        eval(r#"{{ {party: [{name: "Alice"}, {name: "Bob"}]} }}"#),
        r#"{"party":[{"name":"Alice"},{"name":"Bob"}]}"#
    );
}

#[test]
fn a_duplicate_key_is_a_parse_error() {
    let errors = parse(r#"{{ {hp: 1, hp: 2} }}"#).unwrap_err();
    assert!(
        errors[0].message.contains("duplicate object key: hp"),
        "got {:?}",
        errors[0].message
    );
}

// ── Separators ──────────────────────────────────────────────────────────

#[test]
fn a_newline_separates_entries_and_elements() {
    let source = "{{ {\n  name: \"Alice\"\n  hp: 10\n} }}";
    assert_eq!(eval(source), r#"{"hp":10,"name":"Alice"}"#);
    assert_eq!(eval("{{ [\n  1\n  2\n  3\n] }}"), "1, 2, 3");
}

#[test]
fn a_trailing_separator_is_allowed() {
    assert_eq!(eval(r#"{{ {a: 1, b: 2,} }}"#), r#"{"a":1,"b":2}"#);
    assert_eq!(eval("{{ [1, 2, 3,] }}"), "1, 2, 3");
    assert_eq!(eval("{{ [1,\n 2,\n] }}"), "1, 2");
}

#[test]
fn a_hand_written_list_may_be_one_element_per_line() {
    let source = r#"{{ {
  name: "Rags to Riches"
  agents: [
    {name: "Alice", role: "thief"}
    {name: "Bob", role: "fence"}
  ]
} }}"#;
    assert_eq!(
        eval(source),
        r#"{"agents":[{"name":"Alice","role":"thief"},{"name":"Bob","role":"fence"}],"name":"Rags to Riches"}"#
    );
}

#[test]
fn an_operator_must_open_on_the_line_its_operand_closed() {
    // Otherwise `[1\n-2]` would be the single element -1 rather than the
    // two elements written.
    assert_eq!(eval("{{ [1\n-2] }}"), "1, -2");
    assert_eq!(eval("{{ [1 -2] }}"), "-1");
    assert_eq!(eval("{{ 1 +\n 2 }}"), "3");
}

// ── Expressions in `{{ }}` ──────────────────────────────────────────────

#[test]
fn arithmetic_no_longer_needs_a_processor() {
    assert_eq!(eval("{{ 1 + 2 }}"), "3");
    assert_eq!(eval("{{ (2 + 3) * 4 }}"), "20");
    assert_eq!(eval("{{ 7 % 3 }}"), "1");
}

#[test]
fn a_reference_is_an_ordinary_atom() {
    let mut ctx = SimpleContext::new();
    ctx.set("global", "gold", 100i64);
    assert_eq!(eval_in("{{ global:gold - 10 }}", &mut ctx), "90");
    assert_eq!(
        eval_in("{# if global:gold > 50 #}rich{# endif #}", &mut ctx),
        "rich"
    );
}

#[test]
fn a_bare_reference_still_means_a_loop_binding_only() {
    assert_eq!(
        eval("{# foreach n in [1, 2, 3] #}{{ n * 2 }} {# endforeach #}"),
        "2 4 6 "
    );

    let mut ctx = SimpleContext::new();
    ctx.set("local", "n", 5i64);
    let err = match render("{{ n }}", &mut ctx, &Registry::new()) {
        Err(weaver_lang::RenderError::Eval(e)) => e,
        other => panic!("expected an eval error, got {other:?}"),
    };
    assert_eq!(err.kind, EvalErrorKind::UndefinedVariable);
}

#[test]
fn a_keyword_is_not_swallowed_by_a_reference() {
    // `true` is a literal, `truename` a reference — `kw_end` keeps the
    // first from claiming a prefix of the second.
    assert_eq!(
        eval("{# foreach truename in [\"x\"] #}{{ truename }}{# endforeach #}"),
        "x"
    );
    assert_eq!(eval("{{ true }}"), "true");
}

#[test]
fn a_glued_minus_still_trims_and_a_spaced_one_negates() {
    assert_eq!(eval("a   {{- 1 + 1 }}"), "a2");
    assert_eq!(eval("{{-1}}"), "-1");
    assert_eq!(eval("{{ -1 }}"), "-1");
}

// ── Indexing ────────────────────────────────────────────────────────────

#[test]
fn an_array_is_indexed_by_position() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());
    assert_eq!(eval_in("{{char:alice.gear[0]}}", &mut ctx), "sword");
    assert_eq!(eval_in("{{char:alice.gear[2]}}", &mut ctx), "torch");
}

#[test]
fn a_subscript_chains_with_dotted_access_in_both_directions() {
    let mut ctx = SimpleContext::new();
    ctx.set(
        "char",
        "party",
        Value::Array(vec![alice(), Value::object([("name", "Bob")])]),
    );
    assert_eq!(eval_in("{{char:party[0].name}}", &mut ctx), "Alice");
    assert_eq!(eval_in("{{char:party[0].stats.hp}}", &mut ctx), "10");
    assert_eq!(eval_in("{{char:party[1].name}}", &mut ctx), "Bob");
}

#[test]
fn a_string_subscript_is_the_same_as_a_dotted_key() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());
    assert_eq!(eval_in(r#"{{char:alice["name"]}}"#, &mut ctx), "Alice");
    assert_eq!(
        eval_in(r#"{{char:alice["stats"]["hp"]}}"#, &mut ctx),
        eval_in("{{char:alice.stats.hp}}", &mut ctx)
    );
}

#[test]
fn a_subscript_may_be_computed() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());
    ctx.set("global", "slot", 1i64);
    assert_eq!(
        eval_in("{{char:alice.gear[global:slot]}}", &mut ctx),
        "shield"
    );
    assert_eq!(
        eval_in("{{char:alice.gear[global:slot + 1]}}", &mut ctx),
        "torch"
    );
    assert_eq!(
        eval_in(
            "{# foreach i in [0, 2] #}{{char:alice.gear[i]}} {# endforeach #}",
            &mut ctx
        ),
        "sword torch "
    );
}

#[test]
fn a_literal_may_be_indexed_directly() {
    assert_eq!(eval(r#"{{ ["a", "b"][1] }}"#), "b");
    assert_eq!(eval(r#"{{ {hp: 10}.hp }}"#), "10");
}

#[test]
fn indexing_past_the_end_is_absence_not_zero() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());

    let err = match render("{{char:alice.gear[9]}}", &mut ctx, &Registry::new()) {
        Err(weaver_lang::RenderError::Eval(e)) => e,
        other => panic!("expected an eval error, got {other:?}"),
    };
    assert_eq!(err.kind, EvalErrorKind::UndefinedVariable);

    // Lenient mode passes the reference through unevaluated, exactly as it
    // does for a missing object key.
    let out = render_with_options(
        "{{char:alice.gear[9]}}",
        &mut ctx,
        &Registry::new(),
        EvalOptions::new().lenient(true),
    )
    .unwrap();
    assert_eq!(out, "{{char:alice.gear[9]}}");
}

#[test]
fn a_negative_or_fractional_subscript_is_an_error_not_a_wrap() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());

    for source in ["{{char:alice.gear[-1]}}", "{{char:alice.gear[1.5]}}"] {
        let err = match render(source, &mut ctx, &Registry::new()) {
            Err(weaver_lang::RenderError::Eval(e)) => e,
            other => panic!("expected an eval error for {source}, got {other:?}"),
        };
        assert_eq!(err.kind, EvalErrorKind::TypeError, "for {source}");
    }
}

#[test]
fn indexing_the_wrong_shape_is_a_type_error() {
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());

    // An object has no positions and an array has no keys.
    assert_eq!(eval_err("{{ {hp: 1}[0] }}").kind, EvalErrorKind::TypeError);
    assert_eq!(
        eval_err(r#"{{ ["a"]["name"] }}"#).kind,
        EvalErrorKind::TypeError
    );
    // A subscript is not coerced: "0" is a key, 0 is a position.
    assert_eq!(
        eval_err(r#"{{ ["a"]["0"] }}"#).kind,
        EvalErrorKind::TypeError
    );
    assert_eq!(eval_err("{{ [1][true] }}").kind, EvalErrorKind::TypeError);
}

// ── What the host is asked for ──────────────────────────────────────────

/// Records the paths pushed down to `resolve_variable_path`.
#[derive(Default)]
struct RecordingHost {
    requests: RefCell<Vec<String>>,
    values: BTreeMap<String, Value>,
}

impl EvalContext for RecordingHost {
    fn resolve_variable(&self, _scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        Ok(self.values.get(name).cloned())
    }

    fn resolve_variable_path(
        &self,
        scope: &str,
        name: &str,
        path: &[String],
    ) -> Result<Option<Value>, EvalError> {
        self.requests
            .borrow_mut()
            .push(format!("{scope}:{name}/{}", path.join(".")));
        let Some(root) = self.resolve_variable(scope, name)? else {
            return Ok(None);
        };
        Ok(root.get_path(path).unwrap().cloned())
    }

    fn set_variable(&mut self, _scope: &str, _name: &str, _v: Value) -> Result<(), EvalError> {
        Ok(())
    }

    fn fire_trigger(&mut self, _id: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no triggers"))
    }

    fn resolve_document(&mut self, _id: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no documents"))
    }
}

#[test]
fn the_host_is_handed_the_named_path_up_to_the_first_subscript() {
    let mut host = RecordingHost::default();
    host.values.insert("alice".into(), alice());

    let out = render("{{char:alice.gear[1]}}", &mut host, &Registry::new()).unwrap();

    assert_eq!(out, "shield");
    // `gear` is a field the host could have resolved; `[1]` is a position
    // in the value it returned, and the evaluator walks that itself.
    assert_eq!(*host.requests.borrow(), ["char:alice/gear"]);
}

#[test]
fn a_dotted_path_after_a_subscript_stays_with_the_evaluator() {
    let mut host = RecordingHost::default();
    host.values
        .insert("party".into(), Value::Array(vec![alice()]));

    let out = render("{{char:party[0].stats.hp}}", &mut host, &Registry::new()).unwrap();

    assert_eq!(out, "10");
    assert_eq!(*host.requests.borrow(), ["char:party/"]);
}

// ── Paths as data ───────────────────────────────────────────────────────

#[test]
fn get_segments_walks_keys_and_positions_alike() {
    let path = [PathSegment::Key("gear".into()), PathSegment::Index(1)];
    assert_eq!(
        alice().get_segments(&path).unwrap(),
        Some(&Value::String("shield".into()))
    );

    let past_end = [PathSegment::Key("gear".into()), PathSegment::Index(9)];
    assert_eq!(alice().get_segments(&past_end).unwrap(), None);

    let wrong_shape = [PathSegment::Key("name".into()), PathSegment::Index(0)];
    assert!(alice().get_segments(&wrong_shape).is_err());
}
