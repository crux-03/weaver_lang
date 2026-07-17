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
//!
//! # Signatures describe callables to tooling
//!
//! Every callable declares a [`CommandSignature`] or [`ProcessorSignature`]:
//! a description, a return type, and its parameters with their types and
//! optionality. This is the only view an editor has of what a plugin
//! provides, so the fields are required rather than optional — a callable
//! that cannot describe itself cannot be registered.
//!
//! Hosts read this back with [`Registry::command_signatures`] and
//! [`Registry::processor_signatures`]. Enable the `serde` feature to ship
//! the result to a frontend.
//!
//! Prefer the macros over hand-written signatures. A hand-written signature
//! is an independent claim about code that may not match it; a generated one
//! is derived from the same function the extraction code is, so `required`
//! and `expected_type` cannot drift from what the body actually accepts.

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

/// Describes a command: its identity, what it does, and what it expects.
///
/// Every field is public and required. Signatures are the only description
/// of a callable that tooling — editor autocompletion, plugin browsers,
/// generated docs — can see, so a callable that cannot describe itself
/// cannot be registered.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CommandSignature {
    pub name: String,
    /// One line, imperative, no trailing period: "Set a variable in a
    /// writable scope."
    pub description: String,
    /// The type this command evaluates to in template position.
    /// [`ValueType::None`] for commands that exist only for their effect.
    pub returns: ValueType,
    pub params: Vec<ParamDef>,
    /// Whether the command takes `&mut dyn EvalContext` — that is, whether
    /// calling it can change evaluation state. Derived by the
    /// `#[weaver_command]` macro from the presence of a `ctx` parameter.
    pub mutates_context: bool,
}

/// Describes a processor: its identity, what it does, and what it expects.
///
/// See [`CommandSignature`] on why every field is required.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProcessorSignature {
    pub namespace: String,
    pub name: String,
    /// One line, imperative, no trailing period: "Uppercase a string."
    pub description: String,
    /// The type this processor evaluates to. Processors always produce a
    /// value, so this is rarely [`ValueType::None`].
    pub returns: ValueType,
    pub properties: Vec<PropertyDef>,
}

/// A positional parameter definition for a command signature.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamDef {
    pub name: String,
    /// May be empty — a parameter whose name and type say enough.
    pub description: String,
    pub expected_type: Option<ValueType>,
    pub required: bool,
}

/// A named property definition for a processor signature.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PropertyDef {
    pub key: String,
    /// May be empty — a property whose key and type say enough.
    pub description: String,
    pub expected_type: Option<ValueType>,
    pub required: bool,
}

/// Type tag used in signatures for runtime validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ValueType {
    String,
    Number,
    Bool,
    Array,
    /// The absence of a value. Mirrors [`Value::None`] — used as a return
    /// type for commands that produce no output in template position.
    None,
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
            ValueType::None => matches!(value, Value::None),
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

    /// Every registered command's signature, in unspecified order.
    ///
    /// Intended for tooling: an editor lists these to offer completions and
    /// hovers. Sort by [`CommandSignature::name`] if you need stable output.
    pub fn command_signatures(&self) -> Vec<CommandSignature> {
        self.commands.values().map(|c| c.signature()).collect()
    }

    /// Every registered processor's signature, in unspecified order.
    pub fn processor_signatures(&self) -> Vec<ProcessorSignature> {
        self.processors.values().map(|p| p.signature()).collect()
    }

    /// One command's signature, or `None` if it is not registered.
    pub fn command_signature(&self, name: &str) -> Option<CommandSignature> {
        self.commands.get(name).map(|c| c.signature())
    }

    /// One processor's signature, or `None` if it is not registered.
    pub fn processor_signature(&self, namespace: &str, name: &str) -> Option<ProcessorSignature> {
        self.processors
            .get(&format!("{namespace}.{name}"))
            .map(|p| p.signature())
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
    /// Create a closure command. The signature starts undescribed and
    /// untyped — `returns` is [`ValueType::Any`] and there are no declared
    /// params. Use [`describe`](Self::describe), [`returns`](Self::returns)
    /// and [`param`](Self::param) to fill it in; anything left blank shows
    /// up blank in editors.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            sig: CommandSignature {
                name: name.into(),
                description: String::new(),
                returns: ValueType::Any,
                params: Vec::new(),
                mutates_context: false,
            },
            func,
        }
    }

    /// Set the one-line description shown in editor hovers.
    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.sig.description = description.into();
        self
    }

    /// Declare what this command evaluates to in template position.
    pub fn returns(mut self, returns: ValueType) -> Self {
        self.sig.returns = returns;
        self
    }

    /// Append a positional parameter. Order of calls is argument order.
    pub fn param(
        mut self,
        name: impl Into<String>,
        expected_type: ValueType,
        required: bool,
    ) -> Self {
        self.sig.params.push(ParamDef {
            name: name.into(),
            description: String::new(),
            expected_type: Some(expected_type),
            required,
        });
        self
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
    /// Create a closure processor. As with [`ClosureCommand::new`], the
    /// signature starts undescribed — fill it in with
    /// [`describe`](Self::describe), [`returns`](Self::returns) and
    /// [`property`](Self::property).
    pub fn new(namespace: impl Into<String>, name: impl Into<String>, func: F) -> Self {
        Self {
            sig: ProcessorSignature {
                namespace: namespace.into(),
                name: name.into(),
                description: String::new(),
                returns: ValueType::Any,
                properties: Vec::new(),
            },
            func,
        }
    }

    /// Set the one-line description shown in editor hovers.
    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.sig.description = description.into();
        self
    }

    /// Declare what this processor evaluates to.
    pub fn returns(mut self, returns: ValueType) -> Self {
        self.sig.returns = returns;
        self
    }

    /// Declare a named property this processor reads.
    pub fn property(
        mut self,
        key: impl Into<String>,
        expected_type: ValueType,
        required: bool,
    ) -> Self {
        self.sig.properties.push(PropertyDef {
            key: key.into(),
            description: String::new(),
            expected_type: Some(expected_type),
            required,
        });
        self
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
