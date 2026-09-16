//! Weaver Template Notation: a document that is one value with holes.
//!
//! The premise is that this is not a second language. Below the entry rule
//! it is the same expressions, the same literals and the same loops text
//! mode uses — so these tests mostly pin down the three things that *are*
//! different: what a document is, what a string means inside one, and what
//! `#inputs` buys.

#![cfg(feature = "wtn")]

use std::cell::RefCell;
use std::collections::BTreeMap;

use weaver_lang::{
    EvalContext, EvalError, EvalErrorKind, EvalOptions, InputType, Registry, SimpleContext, Value,
    evaluate_value_doc_with_options, expand_value_doc, parse_value_doc,
};

fn expand(source: &str) -> Value {
    let mut ctx = SimpleContext::new();
    expand_value_doc(source, &BTreeMap::new(), &mut ctx, &Registry::new()).unwrap()
}

fn expand_with(source: &str, inputs: BTreeMap<String, Value>) -> Value {
    let mut ctx = SimpleContext::new();
    expand_value_doc(source, &inputs, &mut ctx, &Registry::new()).unwrap()
}

fn expand_err(source: &str, inputs: BTreeMap<String, Value>) -> EvalError {
    let mut ctx = SimpleContext::new();
    match expand_value_doc(source, &inputs, &mut ctx, &Registry::new()) {
        Err(weaver_lang::RenderError::Eval(e)) => e,
        other => panic!("expected an eval error, got {other:?}"),
    }
}

fn inputs(pairs: [(&str, Value); 1]) -> BTreeMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Rewrite every number as an f64, so a comparison is about values rather
/// than about how serde_json happened to tag them.
fn as_floats(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Number(n) => serde_json::json!(n.as_f64().unwrap()),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(as_floats).collect())
        }
        serde_json::Value::Object(map) => {
            serde_json::Value::Object(map.iter().map(|(k, v)| (k.clone(), as_floats(v))).collect())
        }
        other => other.clone(),
    }
}

// ── A document is a value ───────────────────────────────────────────────

#[test]
fn existing_json_parses_and_means_what_it_meant() {
    let json = r#"{"name": "Alice", "hp": 10, "gear": ["sword", "shield"], "alive": true}"#;
    assert_eq!(
        expand(json).to_json(),
        r#"{"alive":true,"gear":["sword","shield"],"hp":10,"name":"Alice"}"#
    );
}

#[test]
fn json_that_is_not_weaver_flavoured_still_parses() {
    // `null`, exponents and JSON's full escape set are the three places a
    // real file would otherwise need editing before it became a document.
    let source = r#"{
      "nothing": null,
      "big": 1e5,
      "small": 1.5E-3,
      "slash": "a\/b",
      "accented": "caf\u00e9",
      "astral": "\ud83d\ude00",
      "control": "bell\bform\ffeed"
    }"#;

    let expanded = expand(source);
    let expected: serde_json::Value = serde_json::from_str(source).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&expanded.to_json()).unwrap();
    // Every number is an f64 either way; only the spelling of a whole one
    // differs, since weaver writes `1e5` back out as `100000`.
    assert_eq!(as_floats(&actual), as_floats(&expected));

    assert_eq!(
        expanded.get_path(&["accented".to_string()]).unwrap(),
        Some(&Value::String("café".into()))
    );
    assert_eq!(
        expanded.get_path(&["astral".to_string()]).unwrap(),
        Some(&Value::String("😀".into()))
    );
}

#[test]
fn a_document_may_be_any_value_not_only_an_object() {
    assert_eq!(
        expand("[1, 2, 3]"),
        Value::Array(vec![1.0.into(), 2.0.into(), 3.0.into()])
    );
    assert_eq!(
        expand(r#""just a string""#),
        Value::String("just a string".into())
    );
    assert_eq!(expand("1 + 2"), Value::Number(3.0));
}

#[test]
fn comments_are_ignored() {
    let doc = r#"
// the realm this template builds
{
  name: "Rags to Riches",  // a title
  // hp: 10
  hp: 20
}"#;
    assert_eq!(
        expand(doc).to_json(),
        r#"{"hp":20,"name":"Rags to Riches"}"#
    );
}

// ── Loops and conditionals in value position ────────────────────────────

#[test]
fn a_loop_in_array_position_yields_elements() {
    let doc = r#"
{
  agents: [
    {# foreach c in [{name: "Alice"}, {name: "Bob"}] #}
    { name: c.name, greeting: "I am {{c.name}}." }
    {# endforeach #}
  ]
}"#;
    assert_eq!(
        expand(doc).to_json(),
        r#"{"agents":[{"greeting":"I am Alice.","name":"Alice"},{"greeting":"I am Bob.","name":"Bob"}]}"#
    );
}

#[test]
fn a_loop_in_object_position_yields_entries() {
    let doc = r#"
{
  {# foreach k in ["hp", "mp"] #}
  "{{k}}": 0
  {# endforeach #}
}"#;
    assert_eq!(expand(doc).to_json(), r#"{"hp":0,"mp":0}"#);
}

#[test]
fn an_element_in_object_position_does_not_parse() {
    // The two contexts have their own item rules, so the mismatch is a
    // parse error rather than a strange value at evaluation time.
    let doc = r#"{ {# foreach k in ["a"] #} 1 {# endforeach #} }"#;
    assert!(parse_value_doc(doc).is_err());
}

#[test]
fn an_entry_in_array_position_does_not_parse() {
    let doc = r#"[ {# foreach k in ["a"] #} name: 1 {# endforeach #} ]"#;
    assert!(parse_value_doc(doc).is_err());
}

#[test]
fn a_conditional_selects_items() {
    let doc = r#"
{
  gear: [
    "sword"
    {# if true #}
    "shield"
    {# endif #}
    {# if false #}
    "torch"
    {# else #}
    "lantern"
    {# endif #}
  ]
}"#;
    assert_eq!(
        expand(doc).to_json(),
        r#"{"gear":["sword","shield","lantern"]}"#
    );
}

#[test]
fn a_conditional_chooses_among_elif_branches() {
    let doc = r#"
{
  {# if false #}
  a: 1
  {# elif true #}
  b: 2
  {# else #}
  c: 3
  {# endif #}
}"#;
    assert_eq!(expand(doc).to_json(), r#"{"b":2}"#);
}

#[test]
fn a_conditional_inside_a_loop_filters() {
    let doc = r#"
[
  {# foreach n in [1, 2, 3, 4] #}
  {# if n % 2 == 0 #}
  n
  {# endif #}
  {# endforeach #}
]"#;
    assert_eq!(expand(doc), Value::Array(vec![2.0.into(), 4.0.into()]));
}

#[test]
fn a_loop_over_a_non_array_is_an_error() {
    let err = expand_err("[{# foreach n in 5 #}n{# endforeach #}]", BTreeMap::new());
    assert_eq!(err.kind, EvalErrorKind::NotIterable);
}

#[test]
fn a_computed_key_produced_twice_is_an_error() {
    let doc = r#"{ {# foreach k in ["hp", "hp"] #} "{{k}}": 1 {# endforeach #} }"#;
    let err = expand_err(doc, BTreeMap::new());
    assert!(
        err.message.contains("duplicate object key: hp"),
        "{}",
        err.message
    );
}

#[test]
fn the_iteration_limit_still_applies() {
    let doc = parse_value_doc("[{# foreach n in [1, 2, 3] #}n{# endforeach #}]").unwrap();
    let mut ctx = SimpleContext::new();
    let err = evaluate_value_doc_with_options(
        &doc,
        &BTreeMap::new(),
        &mut ctx,
        &Registry::new(),
        EvalOptions::new().max_iterations(2),
    )
    .unwrap_err();
    assert_eq!(err.kind, EvalErrorKind::ResourceLimit);
}

// ── Strings nest text mode ──────────────────────────────────────────────

#[test]
fn a_string_is_a_template() {
    let doc = r#"
{
  prompt: "You are {{c:name}}, a {{c:role}}."
}"#;
    let mut ctx = SimpleContext::new();
    ctx.set("c", "name", "Alice");
    ctx.set("c", "role", "thief");
    let value = expand_value_doc(doc, &BTreeMap::new(), &mut ctx, &Registry::new()).unwrap();
    assert_eq!(value.to_json(), r#"{"prompt":"You are Alice, a thief."}"#);
}

#[test]
fn a_raw_string_is_not() {
    let doc = r#"{ prompt: r"Fill in {{placeholder}} yourself." }"#;
    assert_eq!(
        expand(doc).to_json(),
        r#"{"prompt":"Fill in {{placeholder}} yourself."}"#
    );
}

#[test]
fn a_raw_string_may_be_hash_delimited_to_hold_a_quote() {
    let doc = r##"{ prompt: r#"Say "hello" using {{x}}"# }"##;
    assert_eq!(
        expand(doc).to_json(),
        r#"{"prompt":"Say \"hello\" using {{x}}"}"#
    );
}

#[test]
fn a_quoted_value_is_a_string_and_an_unquoted_one_is_an_expression() {
    let mut ctx = SimpleContext::new();
    ctx.set("c", "hp", 10i64);
    let value = expand_value_doc(
        r#"{ text: "{{c:hp}}", number: {{c:hp}} }"#,
        &BTreeMap::new(),
        &mut ctx,
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(value.to_json(), r#"{"number":10,"text":"10"}"#);
}

#[test]
fn an_error_inside_a_string_points_into_the_string() {
    let source = "{ a: \"x {# if #} y\" }";
    let errors = parse_value_doc(source).unwrap_err();
    let span = errors[0].span;
    // Inside the quotes, not at the document's edges.
    assert!(
        span.start > source.find('"').unwrap() && span.end <= source.rfind('"').unwrap() + 1,
        "span {span:?} should fall inside the string literal"
    );
}

// ── Enum variants ───────────────────────────────────────────────────────

#[test]
fn a_newtype_variant_carries_its_value_directly() {
    assert_eq!(
        expand(r#"{ scheduler: Custom([{agent: "alice", turns: 2}]) }"#).to_json(),
        r#"{"scheduler":{"Custom":[{"agent":"alice","turns":2}]}}"#
    );
}

#[test]
fn a_tuple_variant_carries_a_list() {
    assert_eq!(
        expand(r#"{ span: Range(1, 10) }"#).to_json(),
        r#"{"span":{"Range":[1,10]}}"#
    );
}

#[test]
fn a_unit_variant_is_the_string_serde_reads_it_from() {
    assert_eq!(
        expand(r#"{ mode: "RoundRobin" }"#).to_json(),
        r#"{"mode":"RoundRobin"}"#
    );
}

#[test]
fn a_reference_on_its_own_line_is_not_a_variant() {
    // The parenthesis must be glued, so a newline separates two items.
    let doc = "[\n  1\n  (2 + 3)\n]";
    assert_eq!(expand(doc), Value::Array(vec![1.0.into(), 5.0.into()]));
}

// ── Inputs ──────────────────────────────────────────────────────────────

const WITH_INPUTS: &str = r#"
#inputs
characters: [Ref<Character>]
difficulty: enum("easy", "normal", "brutal") = "normal"
rounds: number = 3

{
  difficulty: {{input:difficulty}},
  rounds: {{input:rounds}},
  agents: [
    {# foreach c in {{input:characters}} #}
    { character: c }
    {# endforeach #}
  ]
}"#;

/// A host that knows which Snowflakes are live characters.
#[derive(Default)]
struct RealmHost {
    live: Vec<String>,
    validated: RefCell<Vec<String>>,
}

impl EvalContext for RealmHost {
    fn resolve_variable(&self, _scope: &str, _name: &str) -> Result<Option<Value>, EvalError> {
        Ok(None)
    }
    fn set_variable(&mut self, _s: &str, _n: &str, _v: Value) -> Result<(), EvalError> {
        Ok(())
    }
    fn fire_trigger(&mut self, _i: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no triggers"))
    }
    fn resolve_document(&mut self, _i: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no documents"))
    }
    fn validate_input(&self, kind: &str, value: &Value) -> Result<(), EvalError> {
        self.validated
            .borrow_mut()
            .push(format!("{kind}:{}", value.to_output_string()));
        match value.as_string() {
            Some(id) if self.live.iter().any(|live| live == id) => Ok(()),
            _ => Err(EvalError::host_error(format!(
                "no live {kind}: {}",
                value.to_output_string()
            ))),
        }
    }
}

fn realm_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_kind("Character");
    registry
}

#[test]
fn declarations_are_readable_without_evaluating_anything() {
    // This is what generates the instantiation form.
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    assert_eq!(doc.inputs.len(), 3);

    assert_eq!(doc.inputs[0].name, "characters");
    assert_eq!(
        doc.inputs[0].ty,
        InputType::List(Box::new(InputType::Ref("Character".into())))
    );
    assert!(doc.inputs[0].is_required());

    assert_eq!(
        doc.inputs[1].ty,
        InputType::Enum(vec!["easy".into(), "normal".into(), "brutal".into()])
    );
    assert!(!doc.inputs[1].is_required());
    assert_eq!(
        doc.inputs[1].ty.to_string(),
        r#"enum("easy", "normal", "brutal")"#
    );
    assert_eq!(doc.inputs[0].ty.to_string(), "[Ref<Character>]");
}

#[test]
fn defaults_are_applied_by_the_evaluator() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let supplied = inputs([("characters", Value::Array(vec!["snowflake-1".into()]))]);

    let value =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap();

    assert_eq!(
        value.to_json(),
        r#"{"agents":[{"character":"snowflake-1"}],"difficulty":"normal","rounds":3}"#
    );
    // A bare `c` serialized to the id, not a snapshot of the character.
    assert_eq!(*host.validated.borrow(), ["Character:snowflake-1"]);
}

#[test]
fn a_missing_required_input_is_reported_against_its_declaration() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost::default();
    let err = weaver_lang::evaluate_value_doc(&doc, &BTreeMap::new(), &mut host, &realm_registry())
        .unwrap_err();

    assert!(
        err.message.contains("missing required input: characters"),
        "{}",
        err.message
    );
    let span = err.span.expect("the declaration's span");
    assert_eq!(
        &WITH_INPUTS[span.start..span.end],
        "characters: [Ref<Character>]"
    );
}

#[test]
fn a_value_of_the_wrong_shape_is_rejected() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost::default();
    let supplied = inputs([("characters", Value::String("not-a-list".into()))]);
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert_eq!(err.kind, EvalErrorKind::TypeError);
}

#[test]
fn a_value_outside_an_enum_is_rejected() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let mut supplied = inputs([("characters", Value::Array(vec!["snowflake-1".into()]))]);
    supplied.insert("difficulty".into(), Value::String("impossible".into()));

    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message
            .contains(r#""impossible" is not one of "easy", "normal", "brutal""#),
        "{}",
        err.message
    );
}

#[test]
fn a_ref_that_does_not_resolve_is_the_hosts_verdict() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let supplied = inputs([("characters", Value::Array(vec!["deleted".into()]))]);
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message.contains("no live Character: deleted"),
        "{}",
        err.message
    );
    assert!(err.span.is_some(), "reported against the declaration");
}

#[test]
fn an_unregistered_kind_is_an_error() {
    let doc = parse_value_doc(WITH_INPUTS).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let supplied = inputs([("characters", Value::Array(vec!["snowflake-1".into()]))]);
    // An empty registry knows no kinds.
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &Registry::new()).unwrap_err();
    assert!(
        err.message.contains("unknown entity kind: Character"),
        "{}",
        err.message
    );
}

#[test]
fn supplying_an_input_the_document_never_declared_is_an_error() {
    let err = expand_err(
        "#inputs\nname: string = \"x\"\n\n{ a: 1 }",
        inputs([("nmae", Value::String("typo".into()))]),
    );
    assert!(
        err.message.contains("no such input: nmae"),
        "{}",
        err.message
    );
}

#[test]
fn an_input_may_be_indexed_like_any_other_value() {
    let value = expand_with(
        "#inputs\nparty: [string]\n\n{ first: {{input:party[0]}}, all: {{input:party}} }",
        inputs([("party", Value::Array(vec!["Alice".into(), "Bob".into()]))]),
    );
    assert_eq!(
        value.to_json(),
        r#"{"all":["Alice","Bob"],"first":"Alice"}"#
    );
}

#[test]
fn a_document_without_an_inputs_block_is_still_a_document() {
    assert_eq!(expand("{ a: 1 }").to_json(), r#"{"a":1}"#);
    assert!(parse_value_doc("{ a: 1 }").unwrap().inputs.is_empty());
}

#[test]
fn in_text_mode_input_is_an_ordinary_host_scope() {
    // The evaluator only owns `input` inside a data document. A text-mode
    // template resolves it through the host like any other scope.
    let mut ctx = SimpleContext::new();
    ctx.set("input", "name", "from the host");
    let out = weaver_lang::render("{{input:name}}", &mut ctx, &Registry::new()).unwrap();
    assert_eq!(out, "from the host");
}

#[test]
fn the_value_grammar_is_the_same_one_text_mode_uses() {
    // Loops in item position, comments and raw strings are not WTN-only
    // features bolted on above text mode — they are the shared grammar.
    let mut ctx = SimpleContext::new();
    let out = weaver_lang::render(
        "{{ [{# foreach n in [1, 2, 3] #} n * 10 {# endforeach #}] }}",
        &mut ctx,
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(out, "10, 20, 30");

    // A string in text mode stays a string, though: the prose around it is
    // already the template.
    let out =
        weaver_lang::render(r#"{{ "literal {{braces}}" }}"#, &mut ctx, &Registry::new()).unwrap();
    assert_eq!(out, "literal {{braces}}");
}

// ── The whole thing ─────────────────────────────────────────────────────

#[test]
fn the_realm_template_from_the_design_note_expands() {
    let source = r#"
#inputs
characters: [Ref<Character>]
difficulty: enum("easy", "normal", "brutal") = "normal"

{
  name: "Rags to Riches"
  scheduler: Custom(["alice", "bob"])
  difficulty: {{input:difficulty}}
  agents: [
    {# foreach c in {{input:characters}} #}
    {
      character: c                                  // stores the id
      prompt: "You are character {{c}}. Play {{input:difficulty}}."
    }
    {# endforeach #}
  ]
}"#;

    let doc = parse_value_doc(source).unwrap();
    let mut host = RealmHost {
        live: vec!["c-1".into(), "c-2".into()],
        ..Default::default()
    };
    let supplied = inputs([("characters", Value::Array(vec!["c-1".into(), "c-2".into()]))]);

    let value =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap();

    assert_eq!(
        value.to_json(),
        concat!(
            r#"{"agents":[{"character":"c-1","prompt":"You are character c-1. Play normal."},"#,
            r#"{"character":"c-2","prompt":"You are character c-2. Play normal."}],"#,
            r#""difficulty":"normal","name":"Rags to Riches","scheduler":{"Custom":["alice","bob"]}}"#
        )
    );
}

// ── Extended declarations: objects and spaces ───────────────────────────
//
// Two things a form generator could not express before: a field-by-field
// shape (`{char: Ref<Character>, talkativeness: number}`) and a bounded
// space (`range`, `span`), which is a slider with one handle or two.

const GROUP_CHAT: &str = r#"
#inputs
participants: [{char: Ref<Character>, talkativeness: number}]
temperature: range(0, 2) = 0.8
turns: span(1, 50) = {from: 4, to: 8}

{
  temperature: {{input:temperature}}
  turns: {{input:turns}}
  agents: [
    {# foreach p in {{input:participants}} #}
    { character: p.char, weight: p.talkativeness }
    {# endforeach #}
  ]
}"#;

fn group_chat_inputs() -> BTreeMap<String, Value> {
    let mut supplied = BTreeMap::new();
    supplied.insert(
        "participants".to_string(),
        Value::Array(vec![Value::object([
            ("char", Value::from("snowflake-1")),
            ("talkativeness", Value::from(0.5)),
        ])]),
    );
    supplied
}

#[test]
fn an_object_type_declares_the_fields_a_form_renders() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    assert_eq!(doc.inputs.len(), 3);

    assert_eq!(
        doc.inputs[0].ty,
        InputType::List(Box::new(InputType::Object(vec![
            ("char".into(), InputType::Ref("Character".into())),
            ("talkativeness".into(), InputType::Number),
        ])))
    );
    assert_eq!(
        doc.inputs[0].ty.to_string(),
        "[{char: Ref<Character>, talkativeness: number}]"
    );

    assert_eq!(doc.inputs[1].ty, InputType::Range(0.0, 2.0));
    assert_eq!(doc.inputs[1].ty.to_string(), "range(0, 2)");
    assert_eq!(doc.inputs[2].ty, InputType::Span(1.0, 50.0));
    assert_eq!(doc.inputs[2].ty.to_string(), "span(1, 50)");
}

#[test]
fn an_object_input_expands_with_its_fields_reachable() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };

    let value =
        weaver_lang::evaluate_value_doc(&doc, &group_chat_inputs(), &mut host, &realm_registry())
            .unwrap();

    assert_eq!(
        value.to_json(),
        concat!(
            r#"{"agents":[{"character":"snowflake-1","weight":0.5}],"#,
            r#""temperature":0.8,"turns":{"from":4,"to":8}}"#
        )
    );
    // The `Ref` nested inside the object still went to the host.
    assert_eq!(*host.validated.borrow(), ["Character:snowflake-1"]);
}

#[test]
fn a_missing_field_is_reported_against_the_declaration() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let mut supplied = BTreeMap::new();
    supplied.insert(
        "participants".to_string(),
        Value::Array(vec![Value::object([("char", "snowflake-1")])]),
    );

    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message
            .contains("input participants: missing field talkativeness (number)"),
        "{}",
        err.message
    );
    let span = err.span.expect("the declaration's span");
    assert_eq!(
        &GROUP_CHAT[span.start..span.end],
        "participants: [{char: Ref<Character>, talkativeness: number}]"
    );
}

#[test]
fn a_field_the_type_never_declared_is_rejected() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let mut supplied = BTreeMap::new();
    supplied.insert(
        "participants".to_string(),
        Value::Array(vec![Value::object([
            ("char", Value::from("snowflake-1")),
            ("talkativeness", Value::from(0.5)),
            ("mood", Value::from("smug")),
        ])]),
    );

    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message
            .contains("input participants: no such field: mood"),
        "{}",
        err.message
    );
}

#[test]
fn a_range_holds_one_number_inside_its_space() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };

    let mut supplied = group_chat_inputs();
    supplied.insert("temperature".into(), Value::from(2.0));
    assert!(
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).is_ok(),
        "the bounds are inclusive"
    );

    supplied.insert("temperature".into(), Value::from(2.5));
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message
            .contains("input temperature: 2.5 is outside range(0, 2)"),
        "{}",
        err.message
    );

    supplied.insert("temperature".into(), Value::from("warm"));
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message.contains("expected range(0, 2), got string"),
        "{}",
        err.message
    );
}

#[test]
fn a_span_holds_a_from_to_pair_inside_its_space() {
    let doc = parse_value_doc(GROUP_CHAT).unwrap();
    let mut host = RealmHost {
        live: vec!["snowflake-1".into()],
        ..Default::default()
    };
    let mut supplied = group_chat_inputs();

    let span = |from: f64, to: f64| Value::object([("from", from), ("to", to)]);

    supplied.insert("turns".into(), span(10.0, 20.0));
    assert!(weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).is_ok());

    supplied.insert("turns".into(), span(20.0, 10.0));
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message.contains("input turns: from 20 is above to 10"),
        "{}",
        err.message
    );

    supplied.insert("turns".into(), span(0.0, 20.0));
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message
            .contains("input turns: 0 to 20 is outside span(1, 50)"),
        "{}",
        err.message
    );

    // An array is the other plausible spelling, and is not this one.
    supplied.insert("turns".into(), Value::Array(vec![4i64.into(), 8i64.into()]));
    let err =
        weaver_lang::evaluate_value_doc(&doc, &supplied, &mut host, &realm_registry()).unwrap_err();
    assert!(
        err.message.contains("expected span(1, 50), got array"),
        "{}",
        err.message
    );
}

#[test]
fn a_space_written_backwards_is_a_parse_error() {
    let errors = parse_value_doc("#inputs\nheat: range(10, 0)\n\n{}").unwrap_err();
    assert!(
        errors[0].message.contains("empty space: 10 is above 0"),
        "{}",
        errors[0].message
    );
}
