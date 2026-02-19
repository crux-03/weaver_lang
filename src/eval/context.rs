use crate::ast::value::Value;
use crate::error::EvalError;
use crate::Registry;

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
pub trait EvalContext {
    /// Look up a variable by scope and name.
    ///
    /// Return `Ok(None)` if the variable does not exist. The evaluator
    /// will produce an "undefined variable" error in that case.
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError>;

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
    fn resolve_document(&mut self, document_id: &str, registry: &Registry) -> Result<String, EvalError>;
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
    variables: std::collections::HashMap<String, std::collections::HashMap<String, Value>>,
}

impl SimpleContext {
    pub fn new() -> Self {
        Self {
            variables: std::collections::HashMap::new(),
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
        Err(EvalError::new(
            crate::error::EvalErrorKind::TriggerNotFound,
            format!("SimpleContext does not support triggers (tried: {entry_id})"),
        ))
    }

    fn resolve_document(&mut self, document_id: &str, _registry: &Registry) -> Result<String, EvalError> {
        Err(EvalError::new(
            crate::error::EvalErrorKind::DocumentNotFound,
            format!("SimpleContext does not support documents (tried: {document_id})"),
        ))
    }
}
