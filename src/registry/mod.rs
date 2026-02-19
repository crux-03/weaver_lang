//! Command and processor registration for weaver-lang.
//!
//! The [`Registry`] stores callable commands and processors that templates
//! can invoke. The host application populates the registry before evaluation.
//!
//! There are two ways to register callables:
//!
//! - **Closure-based**: Use [`ClosureCommand`] or [`ClosureProcessor`] for
//!   simple cases where a full trait implementation would be boilerplate.
//! - **Trait-based**: Implement [`WeaverCommand`] or [`WeaverProcessor`]
//!   directly for more control over signatures and validation. The
//!   `#[weaver_processor]` macro in the `weaver-macros` crate can generate
//!   processor implementations from a function signature.

use crate::ast::value::Value;
use crate::error::EvalError;
use crate::eval::EvalContext;
use std::collections::HashMap;

// ── Trait definitions ───────────────────────────────────────────────────

/// A callable command, invoked via `$[name(args)]` in templates.
///
/// Commands take positional arguments and can mutate the evaluation
/// context. Their return value is optional — returning `None` produces
/// no output in template position.
pub trait WeaverCommand: Send + Sync {
    /// Execute the command with pre-evaluated positional arguments.
    ///
    /// The command receives mutable access to the evaluation context and
    /// an immutable reference to the registry, allowing it to read/write
    /// variables and invoke other callables.
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        registry: &Registry,
    ) -> Result<Option<Value>, EvalError>;

    /// Declare this command's identity and parameter expectations.
    fn signature(&self) -> CommandSignature;
}

/// A callable processor, invoked via `@[namespace.name(key: value)]` in templates.
///
/// Processors take named properties (key-value pairs) and return a [`Value`].
/// They are pure computations — they do not receive mutable context access.
pub trait WeaverProcessor: Send + Sync {
    /// Execute the processor with pre-evaluated named properties.
    fn call(&self, properties: HashMap<String, Value>) -> Result<Value, EvalError>;

    /// Declare this processor's identity and property expectations.
    fn signature(&self) -> ProcessorSignature;
}

// ── Signatures ──────────────────────────────────────────────────────────

/// Describes a command's name and expected parameters.
#[derive(Debug, Clone)]
pub struct CommandSignature {
    pub name: String,
    pub params: Vec<ParamDef>,
}

/// Describes a processor's namespace, name, and expected properties.
#[derive(Debug, Clone)]
pub struct ProcessorSignature {
    pub namespace: String,
    pub name: String,
    pub properties: Vec<PropertyDef>,
}

/// A positional parameter definition for a command signature.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: String,
    pub expected_type: Option<ValueType>,
    pub required: bool,
}

/// A named property definition for a processor signature.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    pub key: String,
    pub expected_type: Option<ValueType>,
    pub required: bool,
}

/// Type tag used in signatures for runtime validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    String,
    Number,
    Bool,
    Array,
    /// Accepts any value type.
    Any,
}

impl ValueType {
    /// Check whether a runtime [`Value`] matches this type expectation.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            ValueType::Any => true,
            ValueType::String => matches!(value, Value::String(_)),
            ValueType::Number => matches!(value, Value::Number(_)),
            ValueType::Bool => matches!(value, Value::Bool(_)),
            ValueType::Array => matches!(value, Value::Array(_)),
        }
    }
}

// ── Registry ────────────────────────────────────────────────────────────

/// Stores registered commands and processors for use during evaluation.
///
/// The host creates and populates a `Registry`, then passes it to the
/// evaluator. Processors are keyed by `"namespace.name"`, commands by name.
///
/// ```rust
/// use weaver_lang::{Registry, ClosureCommand, ClosureProcessor, Value};
///
/// let mut registry = Registry::new();
///
/// registry.register_processor(ClosureProcessor::new("math", "add", |props| {
///     let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
///     let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
///     Ok(Value::Number(a + b))
/// }));
/// ```
pub struct Registry {
    commands: HashMap<String, Box<dyn WeaverCommand>>,
    processors: HashMap<String, Box<dyn WeaverProcessor>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
            processors: HashMap::new(),
        }
    }

    /// Register a command. If a command with the same name already exists,
    /// it is replaced.
    pub fn register_command(&mut self, cmd: impl WeaverCommand + 'static) {
        let sig = cmd.signature();
        self.commands.insert(sig.name.clone(), Box::new(cmd));
    }

    /// Register a processor. If a processor with the same namespace and
    /// name already exists, it is replaced.
    pub fn register_processor(&mut self, proc: impl WeaverProcessor + 'static) {
        let sig = proc.signature();
        let key = format!("{}.{}", sig.namespace, sig.name);
        self.processors.insert(key, Box::new(proc));
    }

    /// Dispatch a command call. Returns [`EvalError`] if the command
    /// is not registered.
    pub fn call_command(
        &self,
        name: &str,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
    ) -> Result<Option<Value>, EvalError> {
        match self.commands.get(name) {
            Some(cmd) => cmd.call(args, ctx, self),
            None => Err(EvalError::undefined_command(name)),
        }
    }

    /// Dispatch a processor call. Returns [`EvalError`] if the processor
    /// is not registered.
    pub fn call_processor(
        &self,
        namespace: &str,
        name: &str,
        properties: HashMap<String, Value>,
    ) -> Result<Value, EvalError> {
        let key = format!("{namespace}.{name}");
        match self.processors.get(&key) {
            Some(proc) => proc.call(properties),
            None => Err(EvalError::undefined_processor(namespace, name)),
        }
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Closure-based convenience wrappers ──────────────────────────────────

/// A [`WeaverCommand`] implementation backed by a closure.
///
/// Use this for simple commands where implementing the trait manually
/// would be boilerplate. Note: closure commands do not receive context
/// or registry access. For commands that need to mutate state, implement
/// [`WeaverCommand`] directly.
///
/// ```rust
/// use weaver_lang::{ClosureCommand, Value};
///
/// let cmd = ClosureCommand::new("greet", |args| {
///     let name = args.get(0).and_then(|v| v.as_string()).unwrap_or("world");
///     Ok(Some(Value::String(format!("Hello, {name}!"))))
/// });
/// ```
pub struct ClosureCommand<F>
where
    F: Fn(Vec<Value>) -> Result<Option<Value>, EvalError> + Send + Sync,
{
    sig: CommandSignature,
    func: F,
}

impl<F> ClosureCommand<F>
where
    F: Fn(Vec<Value>) -> Result<Option<Value>, EvalError> + Send + Sync,
{
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            sig: CommandSignature {
                name: name.into(),
                params: Vec::new(),
            },
            func,
        }
    }
}

impl<F> WeaverCommand for ClosureCommand<F>
where
    F: Fn(Vec<Value>) -> Result<Option<Value>, EvalError> + Send + Sync,
{
    fn call(
        &self,
        args: Vec<Value>,
        _ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        (self.func)(args)
    }

    fn signature(&self) -> CommandSignature {
        self.sig.clone()
    }
}

/// A [`WeaverProcessor`] implementation backed by a closure.
///
/// Use this for simple processors where implementing the trait manually
/// would be boilerplate.
///
/// ```rust
/// use weaver_lang::{ClosureProcessor, Value};
///
/// let proc = ClosureProcessor::new("core", "echo", |props| {
///     Ok(props.get("message").cloned().unwrap_or(Value::None))
/// });
/// ```
pub struct ClosureProcessor<F>
where
    F: Fn(HashMap<String, Value>) -> Result<Value, EvalError> + Send + Sync,
{
    sig: ProcessorSignature,
    func: F,
}

impl<F> ClosureProcessor<F>
where
    F: Fn(HashMap<String, Value>) -> Result<Value, EvalError> + Send + Sync,
{
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        func: F,
    ) -> Self {
        Self {
            sig: ProcessorSignature {
                namespace: namespace.into(),
                name: name.into(),
                properties: Vec::new(),
            },
            func,
        }
    }
}

impl<F> WeaverProcessor for ClosureProcessor<F>
where
    F: Fn(HashMap<String, Value>) -> Result<Value, EvalError> + Send + Sync,
{
    fn call(&self, properties: HashMap<String, Value>) -> Result<Value, EvalError> {
        (self.func)(properties)
    }

    fn signature(&self) -> ProcessorSignature {
        self.sig.clone()
    }
}
