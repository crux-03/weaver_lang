//! # weaver-lang
//!
//! A text evaluation language for procedural content generation. Templates
//! contain embedded expressions, control flow, and external calls that are
//! evaluated at runtime into a final string output.
//!
//! The crate is split into two layers:
//!
//! - **The language** (parsing, AST, evaluation) lives here and has no
//!   knowledge of any specific application domain.
//! - **The host** implements [`EvalContext`] to provide state (variables,
//!   triggers, documents) and populates a [`Registry`] with callable
//!   processors and commands.
//!
//! ## Quick start
//!
//! ```rust
//! use weaver_lang::{render, SimpleContext, Registry};
//!
//! let mut ctx = SimpleContext::new();
//! ctx.set("global", "name", "Alice");
//!
//! let registry = Registry::new();
//! let output = render("Hello, {{global:name}}!", &mut ctx, &registry).unwrap();
//! assert_eq!(output, "Hello, Alice!");
//! ```
//!
//! ## Compiled templates
//!
//! For repeated evaluation, parse once with [`CompiledTemplate::compile`]
//! and call [`evaluate`] against different contexts:
//!
//! ```rust
//! use weaver_lang::{CompiledTemplate, SimpleContext, Registry};
//!
//! let template = CompiledTemplate::compile("Hello, {{global:name}}!").unwrap();
//! let registry = Registry::new();
//!
//! let mut ctx = SimpleContext::new();
//! ctx.set("global", "name", "Alice");
//! assert_eq!(template.evaluate(&mut ctx, &registry).unwrap(), "Hello, Alice!");
//! ```
//!
//! ## Evaluation options
//!
//! Use [`EvalOptions`] to configure resource limits, cancellation, and
//! lenient mode:
//!
//! ```rust
//! use weaver_lang::{render_with_options, EvalOptions, SimpleContext, Registry};
//!
//! let mut ctx = SimpleContext::new();
//! let registry = Registry::new();
//!
//! let opts = EvalOptions::new()
//!     .max_node_evaluations(10_000)
//!     .max_iterations(1_000)
//!     .lenient(true);
//!
//! let result = render_with_options(
//!     "Hello, {{global:missing}}!",
//!     &mut ctx,
//!     &registry,
//!     opts,
//! ).unwrap();
//! assert_eq!(result, "Hello, {{global:missing}}!");
//! ```

pub mod ast;
pub mod error;
pub mod eval;
mod parser;
pub mod registry;

pub use ast::span::{Span, Spanned};
pub use ast::template::Template;
pub use ast::value::Value;
pub use error::{EvalError, EvalErrorKind, ParseError};
pub use eval::{
    EvalContext, EvalOptions, SimpleContext, eval_expr_value, evaluate, evaluate_with_options,
};
pub use parser::{parse, parse_expr};
pub use registry::{ClosureCommand, ClosureProcessor, Registry, WeaverCommand, WeaverProcessor};

/// Parse source text and evaluate it in a single step.
///
/// For repeated evaluation of the same source, prefer [`CompiledTemplate`]
/// to avoid re-parsing.
pub fn render(
    source: &str,
    ctx: &mut impl EvalContext,
    registry: &Registry,
) -> Result<String, RenderError> {
    let template = parser::parse(source).map_err(RenderError::Parse)?;
    evaluate(&template, ctx, registry).map_err(RenderError::Eval)
}

/// Parse source text and evaluate it with custom options.
///
/// Combines parsing and [`evaluate_with_options`] in a single step.
pub fn render_with_options(
    source: &str,
    ctx: &mut impl EvalContext,
    registry: &Registry,
    options: EvalOptions,
) -> Result<String, RenderError> {
    let template = parser::parse(source).map_err(RenderError::Parse)?;
    evaluate_with_options(&template, ctx, registry, options).map_err(RenderError::Eval)
}

/// Combined error type returned by [`render`] and [`render_with_options`].
#[derive(Debug)]
pub enum RenderError {
    /// One or more errors occurred during parsing.
    Parse(Vec<ParseError>),
    /// An error occurred during evaluation.
    Eval(EvalError),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Parse(errors) => {
                for e in errors {
                    writeln!(f, "{e}")?;
                }
                Ok(())
            }
            RenderError::Eval(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// A parsed template that can be evaluated multiple times without re-parsing.
///
/// Use this when the same template source will be evaluated against different
/// contexts or at different points in time.
///
/// ```rust
/// use weaver_lang::{CompiledTemplate, SimpleContext, Registry};
///
/// let template = CompiledTemplate::compile("HP: {{global:hp}}").unwrap();
/// let registry = Registry::new();
///
/// let mut ctx = SimpleContext::new();
/// ctx.set("global", "hp", 100i64);
/// assert_eq!(template.evaluate(&mut ctx, &registry).unwrap(), "HP: 100");
///
/// ctx.set("global", "hp", 75i64);
/// assert_eq!(template.evaluate(&mut ctx, &registry).unwrap(), "HP: 75");
/// ```
pub struct CompiledTemplate {
    template: Template,
}

impl CompiledTemplate {
    /// Parse source text into a compiled template.
    ///
    /// Returns parse errors if the source contains invalid syntax.
    pub fn compile(source: &str) -> Result<Self, Vec<ParseError>> {
        let template = parser::parse(source)?;
        Ok(Self { template })
    }

    /// Evaluate this template against the given context and registry.
    pub fn evaluate(
        &self,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        evaluate(&self.template, ctx, registry)
    }

    /// Evaluate this template with custom options.
    pub fn evaluate_with_options(
        &self,
        ctx: &mut impl EvalContext,
        registry: &Registry,
        options: EvalOptions,
    ) -> Result<String, EvalError> {
        evaluate_with_options(&self.template, ctx, registry, options)
    }

    /// Access the underlying AST for inspection or analysis.
    pub fn ast(&self) -> &Template {
        &self.template
    }
}

/// A parsed expression that can be evaluated multiple times without re-parsing.
///
/// This is the expression-level counterpart to [`CompiledTemplate`].
/// Use it for lorebook activation conditions, configuration values,
/// or any context where you need to evaluate the same expression
/// repeatedly against different state.
///
/// ```rust
/// use weaver_lang::{CompiledExpr, SimpleContext, Registry, Value};
///
/// let expr = CompiledExpr::compile(r#"{{global:hp}} > 50"#).unwrap();
/// let registry = Registry::new();
///
/// let mut ctx = SimpleContext::new();
/// ctx.set("global", "hp", 75i64);
/// assert_eq!(expr.evaluate(&mut ctx, &registry).unwrap(), Value::Bool(true));
///
/// ctx.set("global", "hp", 30i64);
/// assert_eq!(expr.evaluate(&mut ctx, &registry).unwrap(), Value::Bool(false));
/// ```
pub struct CompiledExpr {
    expr: ast::Expr,
}

impl CompiledExpr {
    /// Parse an expression string into a compiled expression.
    pub fn compile(source: &str) -> Result<Self, Vec<ParseError>> {
        let expr = parse_expr(source)?;
        Ok(Self { expr })
    }

    /// Evaluate this expression against the given context and registry.
    pub fn evaluate(
        &self,
        ctx: &mut impl EvalContext,
        registry: &Registry,
    ) -> Result<Value, EvalError> {
        eval_expr_value(&self.expr, ctx, registry)
    }

    /// Access the underlying AST for inspection.
    pub fn ast(&self) -> &ast::Expr {
        &self.expr
    }
}
