use std::any::Any;
use std::collections::HashMap;

use crate::Registry;
use crate::ast::value::Value;
use crate::error::{EvalError, EvalErrorKind};

/// Trait implemented by the host application to provide state and side effects
/// to the weaver-lang evaluator.
///
/// The evaluator calls these methods when it encounters constructs that
/// require external data: variable lookups, trigger activation, and document
/// resolution. The host controls what scopes exist, what variables are
/// available, and how triggers and documents behave.
///
/// Temporary scopes (e.g. foreach loop bindings) are managed internally by
/// the evaluator and do not pass through this trait. Only named scopes like
/// `"global"` and `"local"` reach the host.
///
/// # Implementor's note
///
/// If `fire_trigger` or `resolve_document` can cause recursive evaluation
/// (i.e. the triggered entry itself contains weaver-lang templates), the
/// host is responsible for cycle detection and depth limiting. The
/// evaluator does not track cross-entry recursion.
pub trait EvalContext: Any {
    /// Look up a variable by scope and name.
    ///
    /// `name` is always a single segment — the dotted part of
    /// `{{char:alice.stats.hp}}` is handled by
    /// [`resolve_variable_path`](Self::resolve_variable_path), not here.
    ///
    /// Return `Ok(None)` if the variable does not exist. The evaluator
    /// will produce an "undefined variable" error in that case.
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError>;

    /// Look up a variable and index a dotted path into it.
    ///
    /// Called for every scoped variable reference; `path` is empty for a
    /// plain `{{scope:name}}`. The default implementation resolves the root
    /// through [`resolve_variable`](Self::resolve_variable) and walks the
    /// path with [`Value::get_path`], which is correct for any host.
    ///
    /// Override it when resolving the root is expensive and the path could
    /// be pushed down into storage — the default has to materialize the
    /// whole root value (an entire character sheet, say) to read one field.
    /// An override must keep the same contract:
    ///
    /// - `Ok(None)` — the root or some segment along the path is absent.
    ///   The evaluator treats this exactly like an undefined variable.
    /// - `Err(_)` with [`EvalErrorKind::TypeError`](crate::EvalErrorKind::TypeError)
    ///   — a non-leaf segment existed but was not an object.
    fn resolve_variable_path(
        &self,
        scope: &str,
        name: &str,
        path: &[String],
    ) -> Result<Option<Value>, EvalError> {
        let Some(root) = self.resolve_variable(scope, name)? else {
            return Ok(None);
        };
        match root.get_path(path) {
            Ok(found) => Ok(found.cloned()),
            Err(err) => Err(EvalError::new(EvalErrorKind::TypeError, err.to_string())),
        }
    }

    /// Store a variable in the given scope.
    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError>;

    /// Evaluate a triggered entry and return its output.
    ///
    /// Called when the evaluator encounters `<trigger id="...">`. The host
    /// should look up the entry, evaluate it, and return the resulting
    /// string. If the entry does not exist, return an appropriate error.
    fn fire_trigger(&mut self, entry_id: &str, registry: &Registry) -> Result<String, EvalError>;

    /// Resolve a document reference and return its content.
    ///
    /// Called when the evaluator encounters `[[DOCUMENT_ID]]`. The host
    /// can return either pre-evaluated content or a raw template string
    /// (which the evaluator will not parse further — the host should
    /// evaluate it before returning if needed).
    fn resolve_document(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<String, EvalError>;

    /// Check that a declared input's value is acceptable.
    ///
    /// Called once per `Ref<Kind>` value at instantiation, before the
    /// document is expanded. `Ref<Character>` means "a Snowflake that must
    /// resolve to a live Character", and only the host can answer that —
    /// the language checks the shape of everything else itself.
    ///
    /// The default accepts anything, so a host that does not use data mode
    /// (or does not need the check) is unaffected. Returning `Err` reports
    /// the failure against the declaration that asked for the value.
    #[cfg(feature = "data")]
    fn validate_input(&self, _kind: &str, _value: &Value) -> Result<(), EvalError> {
        Ok(())
    }

    /// Resolve a document reference and return its [`Value`].
    ///
    /// This is the path that lets one entry hand structured data to
    /// another. A document whose template ends in `{# return [...] #}`
    /// produces an array here, so the including template can iterate it:
    ///
    /// ```text
    /// // LOOT_TABLE:  {# return ["sword", "shield"] #}
    /// {# foreach item in [[LOOT_TABLE]] #} - {{item}}
    /// {# endforeach #}
    /// ```
    ///
    /// Note there is deliberately no trigger counterpart. A trigger marks
    /// another entry for activation rather than producing content, so it
    /// has no value to carry.
    fn resolve_document_value(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<Value, EvalError> {
        Ok(Value::String(self.resolve_document(document_id, registry)?))
    }
}

/// A minimal [`EvalContext`] implementation for testing and single-file use.
///
/// Stores variables in an in-memory map keyed by `(scope, name)`. Triggers
/// and documents are not supported and will return errors if invoked.
///
/// ```rust
/// use weaver_lang::{SimpleContext, Value};
///
/// let mut ctx = SimpleContext::new();
/// ctx.set("global", "hp", 100i64);
/// ctx.set("local", "name", "Alice");
/// ```
pub struct SimpleContext {
    variables: HashMap<String, HashMap<String, Value>>,
    triggers: HashMap<String, String>,
    documents: HashMap<String, String>,
}

impl SimpleContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            triggers: HashMap::new(),
            documents: HashMap::new(),
        }
    }

    /// Set a variable in the given scope. Accepts any type that implements
    /// `Into<Value>` (strings, numbers, booleans, vectors).
    pub fn set(&mut self, scope: &str, name: &str, value: impl Into<Value>) {
        self.variables
            .entry(scope.to_string())
            .or_default()
            .insert(name.to_string(), value.into());
    }

    // Harness-only helpers so dynamic trigger/document resolution can be
    // verified end-to-end.
    pub fn set_trigger(&mut self, entry_id: &str, content: &str) {
        self.triggers
            .insert(entry_id.to_string(), content.to_string());
    }

    pub fn set_document(&mut self, document_id: &str, content: &str) {
        self.documents
            .insert(document_id.to_string(), content.to_string());
    }
}

impl Default for SimpleContext {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalContext for SimpleContext {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        Ok(self
            .variables
            .get(scope)
            .and_then(|vars| vars.get(name))
            .cloned())
    }

    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        self.variables
            .entry(scope.to_string())
            .or_default()
            .insert(name.to_string(), value);
        Ok(())
    }

    fn fire_trigger(&mut self, entry_id: &str, _registry: &Registry) -> Result<String, EvalError> {
        self.triggers
            .get(entry_id)
            .cloned()
            .ok_or_else(|| EvalError::host_error(format!("unknown trigger entry: {entry_id}")))
    }

    fn resolve_document(
        &mut self,
        document_id: &str,
        _registry: &Registry,
    ) -> Result<String, EvalError> {
        self.documents
            .get(document_id)
            .cloned()
            .ok_or_else(|| EvalError::host_error(format!("unknown document: {document_id}")))
    }
}
