//! Documents as value-producing units.
//!
//! `resolve_document_value` is what lets `{# return #}` compose across
//! entries. Without it a returned array collapses to a joined string at
//! the first nesting boundary and only the top-level caller ever sees a
//! typed value.

use std::collections::HashMap;

use weaver_lang::{
    EvalContext, EvalError, Registry, Value, evaluate_value, parse, render, render_value,
};

/// A host that stores entry *sources* and evaluates them on demand, so a
/// document's `{# return #}` reaches the including template.
struct EntryHost {
    entries: HashMap<String, String>,
    vars: HashMap<String, Value>,
}

impl EntryHost {
    fn new(entries: &[(&str, &str)]) -> Self {
        Self {
            entries: entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            vars: HashMap::new(),
        }
    }
}

impl EvalContext for EntryHost {
    fn resolve_variable(&self, _scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        Ok(self.vars.get(name).cloned())
    }

    fn set_variable(&mut self, _scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        self.vars.insert(name.to_string(), value);
        Ok(())
    }

    fn fire_trigger(&mut self, entry_id: &str, _registry: &Registry) -> Result<String, EvalError> {
        Ok(format!("<fired {entry_id}>"))
    }

    fn resolve_document(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        Ok(self
            .resolve_document_value(document_id, registry)?
            .to_output_string())
    }

    fn resolve_document_value(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<Value, EvalError> {
        let source = self
            .entries
            .get(document_id)
            .cloned()
            .ok_or_else(|| EvalError::host_error(format!("unknown document: {document_id}")))?;
        let template = parse(&source).map_err(|e| {
            EvalError::host_error(format!("failed to parse {document_id}: {:?}", e[0].message))
        })?;
        evaluate_value(&template, self, registry)
    }
}

fn host() -> EntryHost {
    EntryHost::new(&[
        ("LOOT", r#"{# return ["sword", "shield", "potion"] #}"#),
        ("COUNT", "{# return 3 #}"),
        ("PRICES", r#"{# return ["10", "20"] #}"#),
        ("PROSE", "a plain document"),
        ("STATS", "{# return 7 #}"),
    ])
}

// ── the composability win ───────────────────────────────────────────────

#[test]
fn a_returned_array_is_iterable_from_the_including_template() {
    let out = render(
        "{# foreach item in [[LOOT]] #}- {{item}}\n{# endforeach #}",
        &mut host(),
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(out, "- sword\n- shield\n- potion\n");
}

#[test]
fn a_returned_number_keeps_its_type_in_arithmetic() {
    let value = render_value("{# return [[COUNT]] + 1 #}", &mut host(), &Registry::new()).unwrap();
    assert_eq!(value, Value::Number(4.0));
}

#[test]
fn a_returned_value_works_in_a_condition() {
    let out = render(
        "{# if [[COUNT]] > 2 #}many{# else #}few{# endif #}",
        &mut host(),
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(out, "many");
}

#[test]
fn documents_compose_into_a_returned_value() {
    // An entry that assembles its result from other entries.
    let mut h = host();
    h.entries
        .insert("BUNDLE".into(), "{# return [[LOOT]] #}".into());
    let value = render_value("{# return [[BUNDLE]] #}", &mut h, &Registry::new()).unwrap();
    assert_eq!(
        value,
        Value::Array(vec!["sword".into(), "shield".into(), "potion".into()])
    );
}

#[test]
fn a_returned_array_still_renders_joined_in_template_position() {
    // Template position is expression position plus to_output_string, so
    // the two can't diverge.
    let out = render("loot: [[LOOT]]", &mut host(), &Registry::new()).unwrap();
    assert_eq!(out, "loot: sword, shield, potion");
}

#[test]
fn a_document_without_a_return_is_still_a_string() {
    let value = render_value("{# return [[PROSE]] #}", &mut host(), &Registry::new()).unwrap();
    assert_eq!(value, Value::String("a plain document".into()));
}

// ── the default implementation is behaviour-preserving ──────────────────

/// A host that implements only the String method — i.e. every host that
/// existed before `resolve_document_value` was added.
struct LegacyHost;

impl EvalContext for LegacyHost {
    fn resolve_variable(&self, _scope: &str, _name: &str) -> Result<Option<Value>, EvalError> {
        Ok(None)
    }
    fn set_variable(&mut self, _s: &str, _n: &str, _v: Value) -> Result<(), EvalError> {
        Ok(())
    }
    fn fire_trigger(&mut self, _id: &str, _r: &Registry) -> Result<String, EvalError> {
        Ok(String::new())
    }
    fn resolve_document(&mut self, id: &str, _r: &Registry) -> Result<String, EvalError> {
        Ok(format!("<{id}>"))
    }
}

#[test]
fn a_host_that_never_overrides_the_value_method_is_unaffected() {
    assert_eq!(
        render("doc: [[FOO]]", &mut LegacyHost, &Registry::new()).unwrap(),
        "doc: <FOO>"
    );
    assert_eq!(
        render_value("{# return [[FOO]] #}", &mut LegacyHost, &Registry::new()).unwrap(),
        Value::String("<FOO>".into())
    );
}

// ── flow does not cross the entry boundary ──────────────────────────────

#[test]
fn a_return_inside_a_document_does_not_terminate_the_including_template() {
    let out = render("before [[STATS]] after", &mut host(), &Registry::new()).unwrap();
    assert_eq!(out, "before 7 after");
}

#[test]
fn a_break_inside_a_document_cannot_reach_the_including_loop() {
    // The document's own parse rejects a bare break, and even a legal one
    // is confined to the document's evaluator. Here the including loop
    // must run all three iterations.
    let mut h = host();
    h.entries.insert(
        "INNER".into(),
        "{# foreach y in [1, 2] #}{{y}}{# break #}{# endforeach #}".into(),
    );
    let out = render(
        "{# foreach x in [1, 2, 3] #}[[INNER]]{# endforeach #}",
        &mut h,
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(out, "111");
}

#[test]
fn a_stop_inside_a_document_does_not_terminate_the_including_template() {
    // Same entry-boundary guarantee as break and return: the document's
    // stop ends the document, and the including template carries on.
    let mut h = host();
    h.entries
        .insert("PARTIAL".into(), "head{# stop #}tail".into());
    let out = render("before [[PARTIAL]] after", &mut h, &Registry::new()).unwrap();
    assert_eq!(out, "before head after");
}

// ── triggers deliberately have no value path ────────────────────────────

#[test]
fn triggers_remain_string_valued() {
    // A trigger activates an entry; it does not produce content, so there
    // is no fire_trigger_value counterpart to route through.
    let value = render_value(
        r#"{# return <trigger id="somewhere"> #}"#,
        &mut host(),
        &Registry::new(),
    )
    .unwrap();
    assert_eq!(value, Value::String("<fired somewhere>".into()));
}
