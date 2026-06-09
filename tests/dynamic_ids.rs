use weaver_lang::{
    ClosureProcessor, EvalOptions, Registry, SimpleContext, Value, evaluate, evaluate_with_options,
    parse,
};

fn ctx() -> SimpleContext {
    let mut c = SimpleContext::new();
    c.set_trigger("dark_forest", "TRIGGER:dark_forest");
    c.set_trigger("enemy_goblin", "TRIGGER:goblin");
    c.set_document("LORE", "DOC:lore");
    c.set_document("new_encounter", "DOC:new_encounter");
    c
}

// 1. Dynamic trigger id from a variable.
#[test]
fn trigger_id_from_variable() {
    let mut c = ctx();
    c.set("enemy", "active_id", "dark_forest");
    let t = parse(r#"<trigger id={{enemy:active_id}}>"#).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert_eq!(out, "TRIGGER:dark_forest");
}

// 2. Literal trigger id still works (regression).
#[test]
fn trigger_id_literal_still_works() {
    let mut c = ctx();
    let t = parse(r#"<trigger id="dark_forest">"#).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert_eq!(out, "TRIGGER:dark_forest");
}

// 3. Dynamic document id from a variable.
#[test]
fn document_id_from_variable() {
    let mut c = ctx();
    c.set("state", "doc", "LORE");
    let t = parse(r#"[[{{state:doc}}]]"#).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert_eq!(out, "DOC:lore");
}

// 4. Bare-identifier document id still works (regression).
#[test]
fn document_id_bare_identifier_still_works() {
    let mut c = ctx();
    let t = parse(r#"[[LORE]]"#).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert_eq!(out, "DOC:lore");
}

// 5. The user's exact template, both branches.
#[test]
fn user_encounter_template_active_branch() {
    let src = r#"<Encounter>
{# if {{enemy:has_active}} #}
{# if {{enemy:active_id}} != none #}
<trigger id={{enemy:active_id}}>
{# endif #}
{# else #}
[[new_encounter]]
{# endif #}
</Encounter>"#;
    let mut c = ctx();
    c.set("enemy", "has_active", Value::Bool(true));
    c.set("enemy", "active_id", "enemy_goblin");
    let t = parse(src).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert!(out.contains("TRIGGER:goblin"), "got: {out:?}");
}

#[test]
fn user_encounter_template_else_branch() {
    let src = r#"<Encounter>
{# if {{enemy:has_active}} #}
<trigger id={{enemy:active_id}}>
{# else #}
[[new_encounter]]
{# endif #}
</Encounter>"#;
    let mut c = ctx();
    c.set("enemy", "has_active", Value::Bool(false));
    let t = parse(src).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert!(out.contains("DOC:new_encounter"), "got: {out:?}");
}

// 6. A non-string id is a type error, not a silent empty fire.
#[test]
fn trigger_id_none_is_type_error() {
    let mut c = ctx();
    c.set("enemy", "active_id", Value::None);
    let t = parse(r#"<trigger id={{enemy:active_id}}>"#).unwrap();
    let result = evaluate(&t, &mut c, &Registry::new());
    assert!(result.is_err(), "none id should be a type error");
}

#[test]
fn trigger_id_number_is_type_error() {
    let mut c = ctx();
    c.set("enemy", "active_id", Value::Number(42.0));
    let t = parse(r#"<trigger id={{enemy:active_id}}>"#).unwrap();
    assert!(evaluate(&t, &mut c, &Registry::new()).is_err());
}

// 7. Parenthesized expression escape hatch (string concat).
#[test]
fn trigger_id_parenthesized_expr() {
    let mut c = ctx();
    c.set_trigger("enemy_goblin", "TRIGGER:goblin");
    c.set("enemy", "kind", "goblin");
    let t = parse(r#"<trigger id=({{enemy:kind}} + "")>"#).unwrap();
    // ("goblin" + "") == "goblin" — not a registered trigger, should error,
    // proving the expression actually evaluated (concat happened).
    let result = evaluate(&t, &mut c, &Registry::new());
    assert!(result.is_err()); // "goblin" trigger not registered
}

#[test]
fn trigger_id_parenthesized_concat_resolves() {
    let mut c = ctx();
    c.set_trigger("enemy_goblin", "TRIGGER:goblin");
    c.set("enemy", "kind", "goblin");
    let t = parse(r#"<trigger id=("enemy_" + {{enemy:kind}})>"#).unwrap();
    let out = evaluate(&t, &mut c, &Registry::new()).unwrap();
    assert_eq!(out, "TRIGGER:goblin");
}

// 8. Lenient mode falls back to the *evaluated* id, not raw syntax.
#[test]
fn lenient_dynamic_trigger_fallback_uses_value() {
    let mut c = ctx();
    c.set("enemy", "active_id", "missing_entry");
    let t = parse(r#"<trigger id={{enemy:active_id}}>"#).unwrap();
    let opts = EvalOptions::new().lenient(true);
    let out = evaluate_with_options(&t, &mut c, &Registry::new(), opts).unwrap();
    assert_eq!(out, r#"<trigger id="missing_entry">"#);
}

// 9. Processor-call as a trigger id.
#[test]
fn trigger_id_from_processor() {
    use std::collections::HashMap;
    let mut c = ctx();
    c.set_trigger("dark_forest", "TRIGGER:dark_forest");
    let mut reg = Registry::new();
    reg.register_processor(weaver_lang_proc(
        "pick",
        "forest",
        |_props: HashMap<String, Value>| Ok(Value::String("dark_forest".into())),
    ));
    let t = parse(r#"<trigger id=@[pick.forest()]>"#).unwrap();
    let out = evaluate(&t, &mut c, &reg).unwrap();
    assert_eq!(out, "TRIGGER:dark_forest");
}

// Tiny local helper mirroring the crate's ClosureProcessor constructor.
fn weaver_lang_proc<F>(ns: &str, name: &str, f: F) -> ClosureProcessor<F>
where
    F: Fn(std::collections::HashMap<String, Value>) -> Result<Value, weaver_lang::EvalError>
        + Send
        + Sync,
{
    ClosureProcessor::new(ns, name, f)
}
