//! Host-facing behavior of dotted variable paths.
//!
//! The split these tests pin down: the host resolves a single-segment name,
//! and the language indexes the rest of the path into the value it gets
//! back. A host that can do better is allowed to intercept the whole path.

use std::collections::BTreeMap;

use weaver_lang::eval::EvalContext;
use weaver_lang::{EvalError, Registry, SimpleContext, Value, render};

/// Records what the host was asked for, so the tests can assert on the
/// shape of the request rather than only on the answer.
#[derive(Default)]
struct RecordingHost {
    /// Every `(scope, name)` passed to `resolve_variable`.
    roots_requested: std::cell::RefCell<Vec<String>>,
    values: BTreeMap<String, Value>,
}

impl EvalContext for RecordingHost {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        self.roots_requested
            .borrow_mut()
            .push(format!("{scope}:{name}"));
        Ok(self.values.get(name).cloned())
    }

    fn set_variable(&mut self, _scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        self.values.insert(name.to_string(), value);
        Ok(())
    }

    fn fire_trigger(&mut self, _id: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no triggers"))
    }

    fn resolve_document(&mut self, _id: &str, _r: &Registry) -> Result<String, EvalError> {
        Err(EvalError::host_error("no documents"))
    }
}

fn alice() -> Value {
    Value::object([(
        "stats",
        Value::object([("hp", Value::Number(10.0)), ("mp", Value::Number(3.0))]),
    )])
}

#[test]
fn host_is_asked_for_the_root_only() {
    let mut host = RecordingHost::default();
    host.values.insert("alice".to_string(), alice());
    let registry = Registry::new();

    let out = render("{{char:alice.stats.hp}}", &mut host, &registry).unwrap();

    assert_eq!(out, "10");
    // Not "char:alice.stats.hp" — the dotted tail never reaches the host.
    assert_eq!(*host.roots_requested.borrow(), ["char:alice"]);
}

/// A host that resolves the path itself instead of materializing the root.
#[derive(Default)]
struct PushdownHost {
    /// Set when the override is used, so the test can prove the default
    /// walk was bypassed.
    path_requests: std::cell::RefCell<Vec<String>>,
}

impl EvalContext for PushdownHost {
    fn resolve_variable(&self, _scope: &str, _name: &str) -> Result<Option<Value>, EvalError> {
        panic!("the override should have handled this without the root");
    }

    fn resolve_variable_path(
        &self,
        scope: &str,
        name: &str,
        path: &[String],
    ) -> Result<Option<Value>, EvalError> {
        self.path_requests
            .borrow_mut()
            .push(format!("{scope}:{name}/{}", path.join(".")));
        Ok(Some(Value::String("pushed down".to_string())))
    }

    fn set_variable(&mut self, _s: &str, _n: &str, _v: Value) -> Result<(), EvalError> {
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
fn host_can_intercept_the_whole_path() {
    let mut host = PushdownHost::default();
    let registry = Registry::new();

    let out = render("{{char:alice.stats.hp}}", &mut host, &registry).unwrap();

    assert_eq!(out, "pushed down");
    assert_eq!(*host.path_requests.borrow(), ["char:alice/stats.hp"]);
}

#[test]
fn set_path_round_trips_through_a_read() {
    // The write side stays the host's business, but `set_path` is what
    // makes a host's `set_var` agree with how reads walk. This is the
    // shape a host command is expected to use.
    let mut ctx = SimpleContext::new();
    let registry = Registry::new();

    let mut root = Value::None;
    root.set_path(
        &["gear".to_string(), "weapon".to_string()],
        Value::String("sword".to_string()),
    )
    .unwrap();
    ctx.set("char", "alice", root);

    let out = render("{{char:alice.gear.weapon}}", &mut ctx, &registry).unwrap();
    assert_eq!(out, "sword");
}

#[test]
fn objects_are_not_iterable() {
    // Deliberate: iterating an object would need a key/value convention the
    // language does not have. Hosts can expose one as a processor.
    let mut ctx = SimpleContext::new();
    ctx.set("char", "alice", alice());
    let registry = Registry::new();

    let err = render(
        "{# foreach s in {{char:alice.stats}} #}x{# endforeach #}",
        &mut ctx,
        &registry,
    )
    .unwrap_err();

    assert!(err.to_string().contains("array"), "{err}");
}
