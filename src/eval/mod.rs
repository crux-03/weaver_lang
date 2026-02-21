//! Template evaluation engine.
//!
//! The evaluator walks a parsed [`Template`] AST and produces a string
//! output. It resolves variables and dispatches processor/command calls
//! through the provided [`EvalContext`] and [`Registry`].
//!
//! Temporary scopes (e.g. `foreach` loop bindings) are managed internally.
//! The host's `EvalContext` only sees named scope operations like `"global"`
//! and `"local"`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ast::expr::*;
use crate::ast::template::*;
use crate::ast::value::Value;
use crate::error::{EvalError, EvalErrorKind};
use crate::registry::Registry;

mod context;

pub use context::{EvalContext, SimpleContext};

/// Evaluate a template against a context and registry, producing the final
/// string output.
///
/// This creates a fresh evaluator with an empty scope stack on each call.
/// For repeated evaluation of the same template, use [`CompiledTemplate`](crate::CompiledTemplate).
pub fn evaluate(
    template: &Template,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<String, EvalError> {
    let mut evaluator = Evaluator::new(EvalOptions::default());
    evaluator.eval_template(template, ctx, registry)
}

/// Evaluate a parsed expression and return its [`Value`] directly.
///
/// Unlike [`evaluate`], which walks a full template AST and produces
/// a string, this evaluates a single expression and returns the typed
/// result. The value is not coerced to a string — you get the actual
/// [`Value::Bool`], [`Value::Number`], etc.
///
/// This is the evaluation counterpart to [`parse_expr`](crate::parse_expr).
/// Together they support use cases like lorebook activation conditions
/// where you need a boolean result, not rendered text.
///
/// # Examples
///
/// ```rust
/// use weaver_lang::{parse_expr, eval_expr_value, SimpleContext, Registry, Value};
///
/// let expr = parse_expr("1 + 2 == 3").unwrap();
/// let mut ctx = SimpleContext::new();
/// let registry = Registry::new();
///
/// let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
/// assert_eq!(result, Value::Bool(true));
/// ```
///
/// ```rust
/// use weaver_lang::{parse_expr, eval_expr_value, SimpleContext, Registry, Value};
///
/// let expr = parse_expr(r#"{{global:hp}} > 50"#).unwrap();
/// let mut ctx = SimpleContext::new();
/// ctx.set("global", "hp", 75i64);
/// let registry = Registry::new();
///
/// let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
/// assert_eq!(result, Value::Bool(true));
/// ```
pub fn eval_expr_value(
    expr: &Expr,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<Value, EvalError> {
    let mut evaluator = Evaluator::new(EvalOptions::default());
    evaluator.eval_expr(expr, ctx, registry)
}

/// Evaluate a template with custom options for resource limits, cancellation,
/// and lenient mode.
///
/// # Example: resource limits
///
/// ```rust
/// use weaver_lang::{evaluate_with_options, EvalOptions, SimpleContext, Registry};
///
/// let template = weaver_lang::parse("Hello!").unwrap();
/// let mut ctx = SimpleContext::new();
/// let registry = Registry::new();
///
/// let opts = EvalOptions::new().max_node_evaluations(1000).max_iterations(100);
/// let result = evaluate_with_options(&template, &mut ctx, &registry, opts);
/// assert_eq!(result.unwrap(), "Hello!");
/// ```
///
/// # Example: lenient mode
///
/// ```rust
/// use weaver_lang::{evaluate_with_options, EvalOptions, SimpleContext, Registry};
///
/// let template = weaver_lang::parse("Hi {{global:missing}}!").unwrap();
/// let mut ctx = SimpleContext::new();
/// let registry = Registry::new();
///
/// // Strict mode (default) would error. Lenient mode passes through raw syntax.
/// let opts = EvalOptions::new().lenient(true);
/// let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
/// assert_eq!(result, "Hi {{global:missing}}!");
/// ```
pub fn evaluate_with_options(
    template: &Template,
    ctx: &mut impl EvalContext,
    registry: &Registry,
    options: EvalOptions,
) -> Result<String, EvalError> {
    let mut evaluator = Evaluator::new(options);
    evaluator.eval_template(template, ctx, registry)
}

// ── Evaluation options ──────────────────────────────────────────────────

/// Configuration for resource limits, cancellation, and error handling
/// during template evaluation.
///
/// Create with [`EvalOptions::new()`] and chain builder methods:
///
/// ```rust
/// use weaver_lang::EvalOptions;
/// use std::sync::Arc;
/// use std::sync::atomic::AtomicBool;
///
/// let token = Arc::new(AtomicBool::new(false));
/// let opts = EvalOptions::new()
///     .max_node_evaluations(10_000)
///     .max_iterations(1_000)
///     .cancellation_token(token)
///     .lenient(true);
/// ```
#[derive(Clone, Default)]
pub struct EvalOptions {
    /// Maximum number of AST node evaluations before the evaluator
    /// returns a [`ResourceLimit`](EvalErrorKind::ResourceLimit) error.
    /// `None` means unlimited.
    pub max_node_evaluations: Option<u64>,

    /// Maximum number of loop iterations (across all foreach blocks)
    /// before the evaluator returns a
    /// [`ResourceLimit`](EvalErrorKind::ResourceLimit) error.
    /// `None` means unlimited.
    pub max_iterations: Option<u64>,

    /// An external flag that can be set to `true` to cancel an
    /// in-progress evaluation. Checked after each node evaluation.
    pub cancellation_token: Option<Arc<AtomicBool>>,

    /// When `true`, undefined variables, failed processor/command calls,
    /// and trigger/document errors render as their original raw syntax
    /// instead of producing hard errors.
    ///
    /// Useful for preview/authoring contexts where partial results are
    /// preferred over failure.
    pub lenient: bool,
}

impl EvalOptions {
    /// Create a new `EvalOptions` with all defaults (no limits, strict mode).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of AST node evaluations.
    pub fn max_node_evaluations(mut self, limit: u64) -> Self {
        self.max_node_evaluations = Some(limit);
        self
    }

    /// Set the maximum number of loop iterations.
    pub fn max_iterations(mut self, limit: u64) -> Self {
        self.max_iterations = Some(limit);
        self
    }

    /// Attach a cancellation token. Set the `AtomicBool` to `true` from
    /// another thread to abort evaluation.
    pub fn cancellation_token(mut self, token: Arc<AtomicBool>) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Enable or disable lenient evaluation mode.
    pub fn lenient(mut self, lenient: bool) -> Self {
        self.lenient = lenient;
        self
    }
}

impl std::fmt::Debug for EvalOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalOptions")
            .field("max_node_evaluations", &self.max_node_evaluations)
            .field("max_iterations", &self.max_iterations)
            .field("cancellation_token", &self.cancellation_token.is_some())
            .field("lenient", &self.lenient)
            .finish()
    }
}

// ── Lexical scope stack ─────────────────────────────────────────────────

struct ScopeFrame {
    bindings: HashMap<String, Value>,
}

impl ScopeFrame {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn set(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.bindings.get(name)
    }
}

struct Evaluator {
    scopes: Vec<ScopeFrame>,
    options: EvalOptions,
    node_count: u64,
    iteration_count: u64,
}

impl Evaluator {
    fn new(options: EvalOptions) -> Self {
        Self {
            scopes: Vec::new(),
            options,
            node_count: 0,
            iteration_count: 0,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: String, value: Value) {
        if let Some(frame) = self.scopes.last_mut() {
            frame.set(name, value);
        }
    }

    fn resolve_lexical(&self, name: &str) -> Option<Value> {
        for frame in self.scopes.iter().rev() {
            if let Some(val) = frame.get(name) {
                return Some(val.clone());
            }
        }
        None
    }

    /// Check resource limits and cancellation. Called once per node evaluation.
    fn check_limits(&mut self) -> Result<(), EvalError> {
        self.node_count += 1;

        if let Some(max) = self.options.max_node_evaluations
            && self.node_count > max
        {
            return Err(EvalError::new(
                EvalErrorKind::ResourceLimit,
                format!("evaluation exceeded maximum of {max} node evaluations"),
            ));
        }

        if let Some(ref token) = self.options.cancellation_token
            && token.load(Ordering::Relaxed)
        {
            return Err(EvalError::new(
                EvalErrorKind::Cancelled,
                "evaluation cancelled",
            ));
        }

        Ok(())
    }

    /// Check iteration limit. Called once per foreach loop iteration.
    fn check_iteration_limit(&mut self) -> Result<(), EvalError> {
        self.iteration_count += 1;

        if let Some(max) = self.options.max_iterations
            && self.iteration_count > max
        {
            return Err(EvalError::new(
                EvalErrorKind::ResourceLimit,
                format!("evaluation exceeded maximum of {max} loop iterations"),
            ));
        }

        Ok(())
    }

    // ── Template evaluation ─────────────────────────────────────────────

    fn eval_template(
        &mut self,
        template: &Template,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        let mut output = String::new();
        let nodes = &template.nodes;
        let len = nodes.len();

        let mut i = 0;
        while i < len {
            match &nodes[i].node {
                NodeKind::Command(_) => {
                    // Check if this command is standalone on its line.
                    let info = check_standalone(nodes, i);

                    if info.is_standalone {
                        // Trim trailing whitespace (indent) from output
                        if info.ws_only_trim > 0 && output.len() >= info.ws_only_trim {
                            let new_len = output.len() - info.ws_only_trim;
                            output.truncate(new_len);
                        }
                        // If at end of template, also trim the preceding newline
                        if info.trim_preceding_newline
                            && !output.is_empty()
                            && output.ends_with('\n')
                        {
                            output.pop();
                            // Handle \r\n
                            if output.ends_with('\r') {
                                output.pop();
                            }
                        }

                        // Evaluate the command (discard output)
                        self.eval_node(&nodes[i], ctx, registry)?;
                        // Skip leading newline in the following literal
                        if i + 1 < len
                            && let NodeKind::Literal(text) = &nodes[i + 1].node
                        {
                            let trimmed = strip_leading_newline(text);
                            output.push_str(trimmed);
                            i += 2;
                            continue;
                        }

                        // If no following literal (command is at end), just skip
                        i += 1;
                        continue;
                    }

                    // Not standalone — evaluate and use return value as output
                    let fragment = self.eval_node(&nodes[i], ctx, registry)?;
                    output.push_str(&fragment);
                }
                _ => {
                    let fragment = self.eval_node(&nodes[i], ctx, registry)?;
                    output.push_str(&fragment);
                }
            }
            i += 1;
        }

        Ok(output)
    }

    fn eval_node(
        &mut self,
        node: &crate::ast::span::Spanned<NodeKind>,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        self.check_limits()?;

        match &node.node {
            NodeKind::Literal(text) => Ok(text.clone()),
            NodeKind::Expression(expr_kind) => {
                let value = self.eval_expr_kind(expr_kind, node.span, ctx, registry)?;
                Ok(value.to_output_string())
            }
            NodeKind::Command(cmd) => {
                let result = self.eval_command(cmd, node.span, ctx, registry);
                if self.options.lenient {
                    match result {
                        Ok(Some(v)) => Ok(v.to_output_string()),
                        Ok(None) => Ok(String::new()),
                        Err(_) => Ok(reconstruct_command(cmd)),
                    }
                } else {
                    match result? {
                        Some(v) => Ok(v.to_output_string()),
                        None => Ok(String::new()),
                    }
                }
            }
            NodeKind::IfBlock(block) => self.eval_if_block(block, ctx, registry),
            NodeKind::ForEach(block) => self.eval_foreach(block, ctx, registry),
        }
    }

    // ── Expression evaluation ───────────────────────────────────────────

    fn eval_expr(
        &mut self,
        expr: &Expr,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<Value, EvalError> {
        self.eval_expr_kind(&expr.node, expr.span, ctx, registry)
    }

    fn eval_expr_kind(
        &mut self,
        kind: &ExprKind,
        span: crate::ast::span::Span,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<Value, EvalError> {
        match kind {
            ExprKind::Literal(val) => Ok(val.clone()),

            ExprKind::ArrayLiteral(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for elem in elements {
                    values.push(self.eval_expr(elem, ctx, registry)?);
                }
                Ok(Value::Array(values))
            }

            ExprKind::Variable(var) => self.resolve_variable(var, span, ctx),

            ExprKind::ProcessorCall(call) => {
                let mut props = HashMap::new();
                for prop in &call.properties {
                    let val = self.eval_expr(&prop.value, ctx, registry)?;
                    props.insert(prop.key.clone(), val);
                }
                let result = registry
                    .call_processor(&call.namespace, &call.name, props)
                    .map_err(|e| {
                        if e.span.is_none() {
                            e.with_span(span)
                        } else {
                            e
                        }
                    });

                if self.options.lenient {
                    result.or_else(|_| Ok(Value::String(reconstruct_processor(call))))
                } else {
                    result
                }
            }

            ExprKind::CommandCall(cmd) => {
                let result = self.eval_command(cmd, span, ctx, registry);

                if self.options.lenient {
                    result
                        .map(|v| v.unwrap_or(Value::None))
                        .or_else(|_| Ok(Value::String(reconstruct_command(cmd))))
                } else {
                    Ok(result?.unwrap_or(Value::None))
                }
            }

            ExprKind::Trigger(trig) => {
                let result = ctx.fire_trigger(&trig.entry_id, registry).map_err(|e| {
                    if e.span.is_none() {
                        e.with_span(span)
                    } else {
                        e
                    }
                });

                if self.options.lenient {
                    result
                        .map(Value::String)
                        .or_else(|_| Ok(Value::String(reconstruct_trigger(trig))))
                } else {
                    Ok(Value::String(result?))
                }
            }

            ExprKind::Document(doc) => {
                let result = ctx
                    .resolve_document(&doc.document_id, registry)
                    .map_err(|e| {
                        if e.span.is_none() {
                            e.with_span(span)
                        } else {
                            e
                        }
                    });

                if self.options.lenient {
                    result
                        .map(Value::String)
                        .or_else(|_| Ok(Value::String(reconstruct_document(doc))))
                } else {
                    Ok(Value::String(result?))
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                let left_val = self.eval_expr(left, ctx, registry)?;
                let right_val = self.eval_expr(right, ctx, registry)?;
                eval_binary_op(&left_val, *op, &right_val, span)
            }

            ExprKind::UnaryOp { op, operand } => {
                let val = self.eval_expr(operand, ctx, registry)?;
                eval_unary_op(*op, &val, span)
            }
        }
    }

    /// Evaluate a command call, forwarding context and registry.
    fn eval_command(
        &mut self,
        cmd: &CommandCall,
        span: crate::ast::span::Span,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let mut args = Vec::with_capacity(cmd.args.len());
        for arg in &cmd.args {
            args.push(self.eval_expr(arg, ctx, registry)?);
        }
        registry.call_command(&cmd.name, args, ctx).map_err(|e| {
            if e.span.is_none() {
                e.with_span(span)
            } else {
                e
            }
        })
    }

    /// Variable resolution with lexical scope shadowing.
    ///
    /// - **Bare variables** (`{{item}}`): resolve from the lexical scope
    ///   stack only. These are loop bindings and never reach the host.
    /// - **`local` scope**: check lexical stack first, then EvalContext.
    /// - **All other scopes** (global, custom): go directly to EvalContext.
    ///
    /// This means a foreach binding `item` is accessible as `{{item}}`
    /// (preferred) or `{{local:item}}` (backward-compatible). Bare syntax
    /// does NOT fall through to the host context.
    fn resolve_variable(
        &self,
        var: &VariableRef,
        span: crate::ast::span::Span,
        ctx: &impl EvalContext,
    ) -> Result<Value, EvalError> {
        match &var.scope {
            // Bare variable — lexical scope only
            None => match self.resolve_lexical(&var.name) {
                Some(val) => Ok(val),
                None => {
                    if self.options.lenient {
                        Ok(Value::String(format!("{{{{{}}}}}", var.name)))
                    } else {
                        Err(EvalError::new(
                            EvalErrorKind::UndefinedVariable,
                            format!("undefined loop variable: {}", var.name),
                        )
                        .with_span(span))
                    }
                }
            },
            // Scoped variable — check lexical for "local", then host
            Some(scope) => {
                if scope == "local"
                    && let Some(val) = self.resolve_lexical(&var.name)
                {
                    return Ok(val);
                }

                match ctx.resolve_variable(scope, &var.name)? {
                    Some(val) => Ok(val),
                    None => {
                        if self.options.lenient {
                            Ok(Value::String(format!("{{{{{0}:{1}}}}}", scope, var.name)))
                        } else {
                            Err(EvalError::undefined_variable(scope, &var.name).with_span(span))
                        }
                    }
                }
            }
        }
    }

    // ── Control flow ────────────────────────────────────────────────────

    fn eval_if_block(
        &mut self,
        block: &IfBlock,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        let cond_val = self.eval_expr(&block.condition, ctx, registry)?;
        if cond_val.is_truthy() {
            return self.eval_template(&block.body, ctx, registry);
        }

        for elif in &block.elif_branches {
            let elif_val = self.eval_expr(&elif.condition, ctx, registry)?;
            if elif_val.is_truthy() {
                return self.eval_template(&elif.body, ctx, registry);
            }
        }

        if let Some(else_body) = &block.else_body {
            return self.eval_template(else_body, ctx, registry);
        }

        Ok(String::new())
    }

    fn eval_foreach(
        &mut self,
        block: &ForEachBlock,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        let iterable = self.eval_expr(&block.iterable, ctx, registry)?;

        let type_name = iterable.type_name();
        let items = iterable
            .into_array()
            .ok_or_else(|| EvalError::not_iterable(type_name).with_span(block.iterable.span))?;

        let mut output = String::new();

        self.push_scope();

        for item in items {
            self.check_iteration_limit()?;
            self.bind(block.binding.clone(), item);
            let fragment = self.eval_template(&block.body, ctx, registry)?;
            output.push_str(&fragment);
        }

        self.pop_scope();

        Ok(output)
    }
}

// ── Lenient mode: raw syntax reconstruction ─────────────────────────────

fn reconstruct_processor(call: &ProcessorCall) -> String {
    let mut s = format!("@[{}.{}(", call.namespace, call.name);
    for (i, prop) in call.properties.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{}: ...", prop.key));
    }
    s.push_str(")]");
    s
}

fn reconstruct_command(cmd: &CommandCall) -> String {
    format!("$[{}(...)]", cmd.name)
}

fn reconstruct_trigger(trig: &TriggerRef) -> String {
    format!("<trigger id=\"{}\">", trig.entry_id)
}

fn reconstruct_document(doc: &DocumentRef) -> String {
    format!("[[{}]]", doc.document_id)
}

// ── Standalone command detection ────────────────────────────────────────

/// Result of standalone detection for a command node.
struct StandaloneInfo {
    /// Whether the command is standalone on its line.
    is_standalone: bool,
    /// Number of bytes of whitespace-only content to trim from the
    /// already-accumulated output (the indent before the command).
    /// Does NOT include the preceding newline.
    ws_only_trim: usize,
    /// Whether to also trim the preceding newline character.
    /// This is true only when there is no following literal (command at end).
    trim_preceding_newline: bool,
}

/// Check if the command at `idx` is standalone on its line.
fn check_standalone(nodes: &[Node], idx: usize) -> StandaloneInfo {
    let not_standalone = StandaloneInfo {
        is_standalone: false,
        ws_only_trim: 0,
        trim_preceding_newline: false,
    };

    // Check preceding context: the preceding literal must end with \n
    // followed by optional whitespace, OR the command must be the first node.
    let at_start = idx == 0;
    let (preceding_ok, ws_after_nl) = if at_start {
        (true, 0)
    } else {
        match &nodes[idx - 1].node {
            NodeKind::Literal(text) => {
                if text.is_empty() {
                    (true, 0)
                } else {
                    match trailing_ws_after_newline(text) {
                        Some(ws) => (true, ws),
                        None => (false, 0),
                    }
                }
            }
            _ => (false, 0),
        }
    };

    if !preceding_ok {
        return not_standalone;
    }

    // Check following context: the next node must be a literal starting
    // with a newline, OR the command must be the last node.
    let has_following = idx + 1 < nodes.len();
    let following_ok = if !has_following {
        true
    } else {
        match &nodes[idx + 1].node {
            NodeKind::Literal(text) => text.starts_with('\n') || text.starts_with("\r\n"),
            _ => false,
        }
    };

    if !following_ok {
        return not_standalone;
    }

    // A command that is the only node (no preceding AND no following)
    // is NOT standalone — it should render its return value normally.
    if at_start && !has_following {
        return not_standalone;
    }

    if !following_ok {
        return not_standalone;
    }

    // Command at end of template (no following literal): trim the newline
    // AND whitespace from the preceding literal.
    // Command in middle: only trim whitespace indent, strip \n from following.
    let trim_preceding_newline = !has_following && idx > 0;

    StandaloneInfo {
        is_standalone: true,
        ws_only_trim: ws_after_nl,
        trim_preceding_newline,
    }
}

/// If the string ends with `\n` (or `\r\n`) followed by only spaces/tabs,
/// return the count of whitespace-only bytes AFTER the newline.
/// Returns `Some(0)` if the string ends with a bare newline and no trailing ws.
fn trailing_ws_after_newline(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = bytes.len();

    // Skip trailing spaces/tabs
    while i > 0 && (bytes[i - 1] == b' ' || bytes[i - 1] == b'\t') {
        i -= 1;
    }

    // Must find a newline
    if i > 0 && bytes[i - 1] == b'\n' {
        // ws_count = bytes after the \n
        Some(bytes.len() - i)
    } else {
        None
    }
}

/// Strip the leading newline (\n or \r\n) from a string.
fn strip_leading_newline(s: &str) -> &str {
    if let Some(stripped) = s.strip_prefix("\r\n") {
        stripped
    } else if let Some(stripped) = s.strip_prefix('\n') {
        stripped
    } else {
        s
    }
}

// ── Pure operator evaluation ────────────────────────────────────────────

fn eval_binary_op(
    left: &Value,
    op: BinOp,
    right: &Value,
    span: crate::ast::span::Span,
) -> Result<Value, EvalError> {
    match op {
        BinOp::Eq => Ok(Value::Bool(values_equal(left, right))),
        BinOp::NotEq => Ok(Value::Bool(!values_equal(left, right))),

        BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
            let l = require_number(left, span)?;
            let r = require_number(right, span)?;
            let result = match op {
                BinOp::Lt => l < r,
                BinOp::Gt => l > r,
                BinOp::LtEq => l <= r,
                BinOp::GtEq => l >= r,
                _ => unreachable!(),
            };
            Ok(Value::Bool(result))
        }

        BinOp::And => Ok(Value::Bool(left.is_truthy() && right.is_truthy())),
        BinOp::Or => Ok(Value::Bool(left.is_truthy() || right.is_truthy())),

        BinOp::Add => eval_add(left, right, span),
        BinOp::Sub | BinOp::Mul | BinOp::Div => {
            let l = require_number(left, span)?;
            let r = require_number(right, span)?;
            let result = match op {
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => {
                    if r == 0.0 {
                        return Err(EvalError::new(
                            EvalErrorKind::ArithmeticError,
                            "division by zero",
                        )
                        .with_span(span));
                    }
                    l / r
                }
                _ => unreachable!(),
            };
            Ok(Value::Number(result))
        }
    }
}

fn eval_add(left: &Value, right: &Value, span: crate::ast::span::Span) -> Result<Value, EvalError> {
    if let (Some(l), Some(r)) = (left.as_number(), right.as_number()) {
        return Ok(Value::Number(l + r));
    }
    if matches!(left, Value::String(_)) || matches!(right, Value::String(_)) {
        return Ok(Value::String(format!(
            "{}{}",
            left.to_output_string(),
            right.to_output_string()
        )));
    }
    Err(EvalError::type_error(
        "number or string",
        &format!("{} + {}", left.type_name(), right.type_name()),
    )
    .with_span(span))
}

fn eval_unary_op(
    op: UnaryOp,
    val: &Value,
    span: crate::ast::span::Span,
) -> Result<Value, EvalError> {
    match op {
        UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
        UnaryOp::Neg => {
            let n = require_number(val, span)?;
            Ok(Value::Number(-n))
        }
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Number(a), Value::Number(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::None, Value::None) => true,
        _ => false,
    }
}

fn require_number(val: &Value, span: crate::ast::span::Span) -> Result<f64, EvalError> {
    val.as_number()
        .ok_or_else(|| EvalError::type_error("number", val.type_name()).with_span(span))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;
    use crate::registry::{ClosureCommand, ClosureProcessor};

    fn eval_simple(source: &str) -> String {
        let template = parser::parse(source).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        evaluate(&template, &mut ctx, &registry).expect("eval failed")
    }

    fn eval_with_ctx(source: &str, ctx: &mut SimpleContext) -> String {
        let template = parser::parse(source).expect("parse failed");
        let registry = Registry::new();
        evaluate(&template, ctx, &registry).expect("eval failed")
    }

    #[test]
    fn test_literal() {
        assert_eq!(eval_simple("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn test_variable_substitution() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "name", "Alice");
        assert_eq!(
            eval_with_ctx("Hello, {{global:name}}!", &mut ctx),
            "Hello, Alice!"
        );
    }

    #[test]
    fn test_if_true() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        assert_eq!(
            eval_with_ctx("{# if {{global:show}} #}yes{# endif #}", &mut ctx),
            "yes"
        );
    }

    #[test]
    fn test_if_false_with_else() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(false));
        assert_eq!(
            eval_with_ctx(
                "{# if {{global:show}} #}yes{# else #}no{# endif #}",
                &mut ctx
            ),
            "no"
        );
    }

    #[test]
    fn test_elif() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "x", Value::Number(2.0));
        assert_eq!(
            eval_with_ctx(
                "{# if {{global:x}} == 1 #}one{# elif {{global:x}} == 2 #}two{# else #}other{# endif #}",
                &mut ctx
            ),
            "two"
        );
    }

    #[test]
    fn test_foreach() {
        assert_eq!(
            eval_simple(r#"{# foreach item in ["a", "b", "c"] #}{{item}}{# endforeach #}"#),
            "abc"
        );
    }

    #[test]
    fn test_foreach_with_separator() {
        assert_eq!(
            eval_simple(r#"{# foreach item in ["a", "b", "c"] #}{{item}}, {# endforeach #}"#),
            "a, b, c, "
        );
    }

    #[test]
    fn test_processor_call() {
        let template = parser::parse(r#"@[math.add(a: 1, b: 2)]"#).expect("parse failed");
        let mut ctx = SimpleContext::new();

        let mut registry = Registry::new();
        registry.register_processor(ClosureProcessor::new("math", "add", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a + b))
        }));

        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn test_command_call() {
        let template = parser::parse(r#"$[greet("world")]"#).expect("parse failed");
        let mut ctx = SimpleContext::new();

        let mut registry = Registry::new();
        registry.register_command(ClosureCommand::new("greet", |args| {
            let name = args.first().and_then(|v| v.as_string()).unwrap_or("anon");
            Ok(Some(Value::String(format!("Hello, {name}!"))))
        }));

        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_arithmetic_operators() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "a", Value::Number(10.0));
        ctx.set("global", "b", Value::Number(3.0));

        // Arithmetic is tested via if conditions — weaver-lang doesn't have
        // a standalone expression-in-template syntax beyond variables/calls.
        assert_eq!(
            eval_with_ctx(
                r#"{# if {{global:a}} + {{global:b}} == 13 #}correct{# else #}wrong{# endif #}"#,
                &mut ctx
            ),
            "correct"
        );
    }

    #[test]
    fn test_undefined_variable_errors() {
        let template = parser::parse("{{global:missing}}").expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_foreach_does_not_shadow_global() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "item", "GLOBAL_VALUE");
        let src = r#"{# foreach item in ["local_val"] #}local={{item}} global={{global:item}}{# endforeach #}"#;
        assert_eq!(
            eval_with_ctx(src, &mut ctx),
            "local=local_val global=GLOBAL_VALUE"
        );
    }

    #[test]
    fn test_foreach_shadows_host_local() {
        let mut ctx = SimpleContext::new();
        ctx.set("local", "item", "HOST_VALUE");
        let src = r#"before={{local:item}} {# foreach item in ["LOOP"] #}during={{item}}{# endforeach #} after={{local:item}}"#;
        assert_eq!(
            eval_with_ctx(src, &mut ctx),
            "before=HOST_VALUE during=LOOP after=HOST_VALUE"
        );
    }

    #[test]
    fn test_foreach_bare_and_local_scope_compat() {
        // Both {{item}} and {{local:item}} resolve the loop binding
        assert_eq!(
            eval_simple(
                r#"{# foreach x in ["ok"] #}bare={{x}} scoped={{local:x}}{# endforeach #}"#
            ),
            "bare=ok scoped=ok"
        );
    }

    #[test]
    fn test_bare_variable_outside_loop_errors() {
        let template = parser::parse("{{oops}}").expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, EvalErrorKind::UndefinedVariable);
        assert!(err.message.contains("oops"));
    }

    #[test]
    fn test_bare_variable_does_not_resolve_host_locals() {
        // A bare variable should NOT fall through to the host context
        let template = parser::parse("{{name}}").expect("parse failed");
        let mut ctx = SimpleContext::new();
        ctx.set("local", "name", "Alice");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry);
        assert!(
            result.is_err(),
            "bare {{name}} should not resolve host local:name"
        );
    }
}

#[cfg(test)]
mod whitespace_normalization_tests {
    use super::*;
    use crate::parser;
    use crate::registry::ClosureProcessor;

    #[test]
    fn test_if_block_standalone_no_extra_newlines() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        let src = "Line 1\n{# if {{global:show}} #}\nvisible\n{# endif #}\nLine 3";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Line 1\nvisible\nLine 3");
    }

    #[test]
    fn test_if_else_standalone() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(false));
        let src = "Before\n{# if {{global:show}} #}\nyes\n{# else #}\nno\n{# endif #}\nAfter";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Before\nno\nAfter");
    }

    #[test]
    fn test_if_elif_standalone() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "x", Value::Number(2.0));
        let src = "Before\n{# if {{global:x}} == 1 #}\none\n{# elif {{global:x}} == 2 #}\ntwo\n{# else #}\nother\n{# endif #}\nAfter";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Before\ntwo\nAfter");
    }

    #[test]
    fn test_foreach_standalone_no_extra_newlines() {
        let src = "Before\n{# foreach item in [\"a\", \"b\", \"c\"] #}\n- {{item}}\n{# endforeach #}\nAfter";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Before\n- a\n- b\n- c\nAfter");
    }

    #[test]
    fn test_if_at_start() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        let src = "{# if {{global:show}} #}\ncontent\n{# endif #}\nAfter";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "content\nAfter");
    }

    #[test]
    fn test_if_at_end() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        let src = "Before\n{# if {{global:show}} #}\ncontent\n{# endif #}";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Before\ncontent\n");
    }

    #[test]
    fn test_inline_if_not_affected() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        let src = "Result: {# if {{global:show}} #}yes{# endif #}!";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Result: yes!");
    }

    #[test]
    fn test_blank_line_preserved_between_blocks() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "show", Value::Bool(true));
        let src = "{# if {{global:show}} #}\ncontent\n{# endif #}\n\nMore text";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "content\n\nMore text");
    }

    #[test]
    fn test_foreach_with_indent() {
        let src = "Items:\n{# foreach item in [\"sword\", \"shield\"] #}\n  - {{item}}\n{# endforeach #}\nDone";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "Items:\n  - sword\n  - shield\nDone");
    }

    #[test]
    fn test_nested_if_in_foreach() {
        let mut ctx = SimpleContext::new();
        ctx.set("global", "flag", Value::Bool(true));
        let src = "{# foreach item in [\"a\", \"b\"] #}\n{# if {{global:flag}} #}\n{{item}}!\n{# endif #}\n{# endforeach #}";
        let template = parser::parse(src).expect("parse failed");
        let registry = Registry::new();
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(result, "a!\nb!\n");
    }

    #[test]
    fn test_full_scenario() {
        // Simulates the user's actual template (simplified)
        let mut ctx = SimpleContext::new();
        ctx.set("global", "class", "Mage");
        ctx.set("global", "name", "Traveler");
        ctx.set("global", "hp", Value::Number(100.0));

        let mut registry = Registry::new();
        registry.register_processor(ClosureProcessor::new("math", "mul", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a * b))
        }));

        // A set_var command
        use crate::registry::WeaverCommand;
        struct SetVarCmd;
        impl WeaverCommand for SetVarCmd {
            fn call(
                &self,
                args: Vec<Value>,
                ctx: &mut dyn EvalContext,
                _registry: &Registry,
            ) -> Result<Option<Value>, EvalError> {
                let key = args
                    .get(0)
                    .and_then(|v| v.as_string())
                    .unwrap_or("")
                    .to_string();
                let val = args.get(1).cloned().unwrap_or(Value::None);
                if let Some(pos) = key.find(':') {
                    ctx.set_variable(&key[..pos], &key[pos + 1..], val)?;
                }
                Ok(None)
            }
            fn signature(&self) -> crate::registry::CommandSignature {
                crate::registry::CommandSignature {
                    name: "set_var".to_string(),
                    params: Vec::new(),
                }
            }
        }
        registry.register_command(SetVarCmd);

        let src = "\
{# if {{global:class}} == \"Mage\" #}
The mage known as {{global:name}} channels arcane energy.
Their power level: @[math.mul(a: {{global:hp}}, b: 1.5)]
$[set_var(\"local:title\", \"Archmage\")]
Title bestowed: {{local:title}}
{# else #}
{{global:name}} wanders the realm, class unknown.
{# endif #}
$[set_var(\"local:power\", @[math.mul(a: {{global:hp}}, b: 1.5)])]
{# if {{local:power}} > 100 #}
$[set_var(\"local:title\", \"Test\")]
Epic test title: {{local:title}}
{# endif #}";

        let template = parser::parse(src).expect("parse failed");
        let result = evaluate(&template, &mut ctx, &registry).unwrap();
        assert_eq!(
            result,
            "The mage known as Traveler channels arcane energy.\nTheir power level: 150\nTitle bestowed: Archmage\nEpic test title: Test\n"
        );
    }
}

// ── Resource limits and options tests ───────────────────────────────────

#[cfg(test)]
mod options_tests {
    use super::*;
    use crate::parser;

    #[test]
    fn test_node_evaluation_limit() {
        // A simple template that exceeds a very low limit
        let src = r#"{# foreach item in ["a", "b", "c", "d", "e"] #}{{item}}{# endforeach #}"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        // Limit to 3 node evaluations — should fail
        let opts = EvalOptions::new().max_node_evaluations(3);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, EvalErrorKind::ResourceLimit);
        assert!(err.message.contains("node evaluations"));
    }

    #[test]
    fn test_node_evaluation_limit_sufficient() {
        let src = "Hello, world!";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().max_node_evaluations(100);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_iteration_limit() {
        let src = r#"{# foreach item in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] #}{{item}}{# endforeach #}"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        // Allow only 5 iterations
        let opts = EvalOptions::new().max_iterations(5);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, EvalErrorKind::ResourceLimit);
        assert!(err.message.contains("loop iterations"));
    }

    #[test]
    fn test_iteration_limit_sufficient() {
        let src = r#"{# foreach item in ["a", "b", "c"] #}{{item}}{# endforeach #}"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().max_iterations(100);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_cancellation() {
        let src = r#"{# foreach item in ["a", "b", "c"] #}{{item}}{# endforeach #}"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        // Pre-cancelled token
        let token = Arc::new(AtomicBool::new(true));
        let opts = EvalOptions::new().cancellation_token(token);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, EvalErrorKind::Cancelled);
    }

    #[test]
    fn test_cancellation_not_triggered() {
        let src = "Hello!";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let token = Arc::new(AtomicBool::new(false));
        let opts = EvalOptions::new().cancellation_token(token);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Hello!");
    }

    #[test]
    fn test_lenient_undefined_variable() {
        let src = "Hello, {{global:missing}}!";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Hello, {{global:missing}}!");
    }

    #[test]
    fn test_lenient_undefined_bare_variable() {
        let src = "Hello, {{missing}}!";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Hello, {{missing}}!");
    }

    #[test]
    fn test_lenient_defined_variable_still_resolves() {
        let src = "Hello, {{global:name}}!";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        ctx.set("global", "name", "Alice");
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_lenient_undefined_processor() {
        let src = r#"Result: @[missing.proc(key: "val")]"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        // Should contain the reconstructed processor call
        assert!(result.starts_with("Result: @[missing.proc("));
    }

    #[test]
    fn test_lenient_undefined_command() {
        let src = r#"Result: $[unknown("arg")]"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert!(result.contains("$[unknown("));
    }

    #[test]
    fn test_lenient_trigger_fallback() {
        let src = r#"<trigger id="missing_entry">"#;
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, r#"<trigger id="missing_entry">"#);
    }

    #[test]
    fn test_lenient_document_fallback() {
        let src = "[[MISSING_DOC]]";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "[[MISSING_DOC]]");
    }

    #[test]
    fn test_lenient_mixed_defined_and_undefined() {
        let src = "Name: {{global:name}}, Score: {{global:score}}";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        ctx.set("global", "name", "Alice");
        // score is NOT set
        let registry = Registry::new();

        let opts = EvalOptions::new().lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "Name: Alice, Score: {{global:score}}");
    }

    #[test]
    fn test_strict_mode_is_default() {
        let src = "{{global:missing}}";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        // Default evaluate should still be strict
        let result = evaluate(&template, &mut ctx, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_combined_limits_and_lenient() {
        let src = "{{global:missing}} and more";
        let template = parser::parse(src).expect("parse failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let opts = EvalOptions::new()
            .max_node_evaluations(1000)
            .max_iterations(100)
            .lenient(true);
        let result = evaluate_with_options(&template, &mut ctx, &registry, opts).unwrap();
        assert_eq!(result, "{{global:missing}} and more");
    }

    #[test]
    fn test_error_chaining() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let eval_err = EvalError::host_error("failed to load").with_source(io_err);
        assert_eq!(eval_err.kind, EvalErrorKind::HostError);
        assert!(eval_err.source.is_some());

        // Verify the error chain works via std::error::Error::source()
        let source = std::error::Error::source(&eval_err);
        assert!(source.is_some());
    }

    #[test]
    fn test_error_cloneable_with_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let eval_err = EvalError::host_error("failed to load").with_source(io_err);

        // EvalError should remain Clone even with a source
        let cloned = eval_err.clone();
        assert_eq!(cloned.message, eval_err.message);
        assert!(cloned.source.is_some());
    }
}

#[cfg(test)]
mod expr_eval_tests {
    use super::*;
    use crate::parser;
    use crate::registry::{ClosureProcessor};

    fn eval_expr(source: &str) -> Value {
        let expr = parser::parse_expr(source).expect("parse_expr failed");
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        eval_expr_value(&expr, &mut ctx, &registry).expect("eval failed")
    }

    fn eval_expr_ctx(source: &str, ctx: &mut SimpleContext, registry: &Registry) -> Value {
        let expr = parser::parse_expr(source).expect("parse_expr failed");
        eval_expr_value(&expr, ctx, registry).expect("eval failed")
    }

    // ── Literals ────────────────────────────────────────────────────

    #[test]
    fn test_expr_string_literal() {
        assert_eq!(eval_expr(r#""hello""#), Value::String("hello".into()));
    }

    #[test]
    fn test_expr_number_literal() {
        assert_eq!(eval_expr("42"), Value::Number(42.0));
    }

    #[test]
    fn test_expr_float_literal() {
        assert_eq!(eval_expr("3.14"), Value::Number(3.14));
    }

    #[test]
    fn test_expr_negative_number() {
        assert_eq!(eval_expr("-7"), Value::Number(-7.0));
    }

    #[test]
    fn test_expr_bool_true() {
        assert_eq!(eval_expr("true"), Value::Bool(true));
    }

    #[test]
    fn test_expr_bool_false() {
        assert_eq!(eval_expr("false"), Value::Bool(false));
    }

    #[test]
    fn test_expr_none() {
        assert_eq!(eval_expr("none"), Value::None);
    }

    // ── Arrays ──────────────────────────────────────────────────────

    #[test]
    fn test_expr_empty_array() {
        assert_eq!(eval_expr("[]"), Value::Array(vec![]));
    }

    #[test]
    fn test_expr_array_of_strings() {
        assert_eq!(
            eval_expr(r#"["a", "b", "c"]"#),
            Value::Array(vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
            ])
        );
    }

    #[test]
    fn test_expr_mixed_array() {
        assert_eq!(
            eval_expr(r#"["hello", 42, true]"#),
            Value::Array(vec![
                Value::String("hello".into()),
                Value::Number(42.0),
                Value::Bool(true),
            ])
        );
    }

    // ── Arithmetic ──────────────────────────────────────────────────

    #[test]
    fn test_expr_addition() {
        assert_eq!(eval_expr("1 + 2"), Value::Number(3.0));
    }

    #[test]
    fn test_expr_subtraction() {
        assert_eq!(eval_expr("10 - 3"), Value::Number(7.0));
    }

    #[test]
    fn test_expr_multiplication() {
        assert_eq!(eval_expr("4 * 5"), Value::Number(20.0));
    }

    #[test]
    fn test_expr_division() {
        assert_eq!(eval_expr("15 / 3"), Value::Number(5.0));
    }

    #[test]
    fn test_expr_chained_arithmetic() {
        // Left-to-right: 1 + (2 * 3) = 7
        assert_eq!(eval_expr("1 + 2 * 3"), Value::Number(7.0));
    }

    #[test]
    fn test_expr_parenthesized() {
        assert_eq!(eval_expr("(1 + 2) * 3"), Value::Number(9.0));
    }

    #[test]
    fn test_expr_string_concatenation() {
        assert_eq!(
            eval_expr(r#""hello" + " " + "world""#),
            Value::String("hello world".into())
        );
    }

    // ── Comparison ──────────────────────────────────────────────────

    #[test]
    fn test_expr_eq_true() {
        assert_eq!(eval_expr("5 == 5"), Value::Bool(true));
    }

    #[test]
    fn test_expr_eq_false() {
        assert_eq!(eval_expr("5 == 6"), Value::Bool(false));
    }

    #[test]
    fn test_expr_neq() {
        assert_eq!(eval_expr("5 != 6"), Value::Bool(true));
    }

    #[test]
    fn test_expr_string_equality() {
        assert_eq!(eval_expr(r#""abc" == "abc""#), Value::Bool(true));
        assert_eq!(eval_expr(r#""abc" == "xyz""#), Value::Bool(false));
    }

    #[test]
    fn test_expr_lt() {
        assert_eq!(eval_expr("3 < 5"), Value::Bool(true));
        assert_eq!(eval_expr("5 < 3"), Value::Bool(false));
    }

    #[test]
    fn test_expr_gt() {
        assert_eq!(eval_expr("5 > 3"), Value::Bool(true));
    }

    #[test]
    fn test_expr_lteq() {
        assert_eq!(eval_expr("5 <= 5"), Value::Bool(true));
        assert_eq!(eval_expr("5 <= 6"), Value::Bool(true));
        assert_eq!(eval_expr("6 <= 5"), Value::Bool(false));
    }

    #[test]
    fn test_expr_gteq() {
        assert_eq!(eval_expr("5 >= 5"), Value::Bool(true));
        assert_eq!(eval_expr("5 >= 4"), Value::Bool(true));
        assert_eq!(eval_expr("4 >= 5"), Value::Bool(false));
    }

    // ── Logical operators ───────────────────────────────────────────

    #[test]
    fn test_expr_and() {
        assert_eq!(eval_expr("true && true"), Value::Bool(true));
        assert_eq!(eval_expr("true && false"), Value::Bool(false));
    }

    #[test]
    fn test_expr_or() {
        assert_eq!(eval_expr("false || true"), Value::Bool(true));
        assert_eq!(eval_expr("false || false"), Value::Bool(false));
    }

    #[test]
    fn test_expr_not() {
        assert_eq!(eval_expr("!true"), Value::Bool(false));
        assert_eq!(eval_expr("!false"), Value::Bool(true));
    }

    #[test]
    fn test_expr_not_truthy() {
        // 0 is falsy, non-zero is truthy
        assert_eq!(eval_expr("!0"), Value::Bool(true));
        assert_eq!(eval_expr("!1"), Value::Bool(false));
        // empty string is falsy
        assert_eq!(eval_expr(r#"!"""#), Value::Bool(true));
        assert_eq!(eval_expr(r#"!"hello""#), Value::Bool(false));
    }

    // ── Variables ───────────────────────────────────────────────────

    #[test]
    fn test_expr_scoped_variable() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "hp", 75i64);
        let registry = Registry::new();
        assert_eq!(
            eval_expr_ctx("{{state:hp}}", &mut ctx, &registry),
            Value::Number(75.0)
        );
    }

    #[test]
    fn test_expr_variable_in_comparison() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "level", 8i64);
        let registry = Registry::new();
        assert_eq!(
            eval_expr_ctx("{{state:level}} >= 5", &mut ctx, &registry),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_expr_string_variable_equality() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "location", "dark_forest");
        let registry = Registry::new();
        assert_eq!(
            eval_expr_ctx(
                r#"{{state:location}} == "dark_forest""#,
                &mut ctx,
                &registry,
            ),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_expr_variable_arithmetic() {
        let mut ctx = SimpleContext::new();
        ctx.set("char", "base_hp", 50i64);
        ctx.set("char", "level", 10i64);
        let registry = Registry::new();
        assert_eq!(
            eval_expr_ctx("{{char:base_hp}} + {{char:level}} * 5", &mut ctx, &registry,),
            // Precedence-climbing: 50 + (10 * 5) = 100
            Value::Number(100.0)
        );
    }

    #[test]
    fn test_expr_multiple_variables_logical() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "has_key", Value::Bool(true));
        ctx.set("state", "door_locked", Value::Bool(true));
        let registry = Registry::new();
        assert_eq!(
            eval_expr_ctx(
                "{{state:has_key}} && {{state:door_locked}}",
                &mut ctx,
                &registry,
            ),
            Value::Bool(true)
        );
    }

    // ── Processor calls ─────────────────────────────────────────────

    #[test]
    fn test_expr_processor_call() {
        let mut ctx = SimpleContext::new();
        let mut registry = Registry::new();
        registry.register_processor(ClosureProcessor::new("math", "add", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a + b))
        }));

        assert_eq!(
            eval_expr_ctx("@[math.add(a: 10, b: 20)]", &mut ctx, &registry),
            Value::Number(30.0)
        );
    }

    #[test]
    fn test_expr_processor_in_comparison() {
        let mut ctx = SimpleContext::new();
        ctx.set("char", "level", 10i64);
        let mut registry = Registry::new();
        registry.register_processor(ClosureProcessor::new("math", "mul", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a * b))
        }));

        assert_eq!(
            eval_expr_ctx(
                "@[math.mul(a: {{char:level}}, b: 1.5)] > 10",
                &mut ctx,
                &registry,
            ),
            Value::Bool(true) // 10 * 1.5 = 15 > 10
        );
    }

    #[test]
    fn test_expr_array_contains_processor() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "location", "dark_forest");
        let mut registry = Registry::new();
        registry.register_processor(ClosureProcessor::new("array", "contains", |props| {
            let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
            let value = props.get("value").cloned().unwrap_or(Value::None);
            let found = items.iter().any(|item| item == &value);
            Ok(Value::Bool(found))
        }));

        assert_eq!(
            eval_expr_ctx(
                r#"@[array.contains(items: ["dark_forest", "dark_forest_outskirts"], value: {{state:location}})]"#,
                &mut ctx,
                &registry,
            ),
            Value::Bool(true)
        );

        ctx.set("state", "location", "town_square");
        assert_eq!(
            eval_expr_ctx(
                r#"@[array.contains(items: ["dark_forest", "dark_forest_outskirts"], value: {{state:location}})]"#,
                &mut ctx,
                &registry,
            ),
            Value::Bool(false)
        );
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn test_expr_undefined_variable_errors() {
        let expr = parser::parse_expr("{{state:missing}}").unwrap();
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = eval_expr_value(&expr, &mut ctx, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_expr_type_error_on_comparison() {
        // Comparing string < number should fail
        let expr = parser::parse_expr(r#""hello" < 5"#).unwrap();
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = eval_expr_value(&expr, &mut ctx, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_expr_division_by_zero() {
        let expr = parser::parse_expr("10 / 0").unwrap();
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();
        let result = eval_expr_value(&expr, &mut ctx, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_expr_parse_error_on_garbage() {
        let result = parser::parse_expr("== == ==");
        assert!(result.is_err());
    }

    // ── Activation-style conditions ─────────────────────────────────
    // These mirror the patterns that lorebook entries will actually use.

    #[test]
    fn test_activation_location_check() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "location", "dark_forest");
        let registry = Registry::new();

        let expr = parser::parse_expr(r#"{{state:location}} == "dark_forest""#).unwrap();

        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_activation_level_gate() {
        let mut ctx = SimpleContext::new();
        ctx.set("char", "level", 12i64);
        let registry = Registry::new();

        let expr = parser::parse_expr("{{char:level}} >= 10 && {{char:level}} < 20").unwrap();

        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(true));

        ctx.set("char", "level", 5i64);
        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_activation_compound_condition() {
        let mut ctx = SimpleContext::new();
        ctx.set("state", "quest_active", Value::Bool(true));
        ctx.set("char", "class", "Mage");
        ctx.set("char", "level", 8i64);
        let registry = Registry::new();

        let expr = parser::parse_expr(
            r#"{{state:quest_active}} && {{char:class}} == "Mage" && {{char:level}} > 5"#,
        )
        .unwrap();

        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(true));

        // Flip one condition
        ctx.set("state", "quest_active", Value::Bool(false));
        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_activation_truthy_none_is_false() {
        // An unset state variable in lenient mode returns None,
        // which is falsy — useful as a "has this been initialized" check.
        let expr = parser::parse_expr("!none").unwrap();
        let mut ctx = SimpleContext::new();
        let registry = Registry::new();

        let result = eval_expr_value(&expr, &mut ctx, &registry).unwrap();
        assert_eq!(result, Value::Bool(true)); // !none == true
    }
}
