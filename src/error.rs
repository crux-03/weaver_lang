//! Error types for parsing and evaluation.
//!
//! [`ParseError`] is produced during template parsing and carries source
//! spans for diagnostic formatting. [`EvalError`] is produced during
//! evaluation and can originate from the evaluator, the registry, or
//! the host's [`EvalContext`](crate::eval::EvalContext) implementation.

use crate::ast::span::Span;
use std::sync::Arc;
use thiserror::Error;

// ── Parse errors ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct ParseError {
    pub span: Span,
    pub message: String,
    pub hint: Option<String>,
}

impl ParseError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            hint: None,
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Format the error with source context for display
    pub fn format_with_source(&self, source: &str, entry_name: Option<&str>) -> String {
        let (line, col) = offset_to_line_col(source, self.span.start);
        let source_line = source.lines().nth(line.saturating_sub(1)).unwrap_or("");

        let location = if let Some(name) = entry_name {
            format!(" --> {name}:{line}:{col}")
        } else {
            format!(" --> {line}:{col}")
        };

        let pointer = " ".repeat(col.saturating_sub(1))
            + &"^".repeat((self.span.end - self.span.start).max(1));

        let mut output = format!("Error: {}\n{location}\n  |\n{line:>3} | {source_line}\n    | {pointer}", self.message);

        if let Some(hint) = &self.hint {
            output.push_str(&format!("\n  = hint: {hint}"));
        }

        output
    }
}

fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

// ── Eval errors ─────────────────────────────────────────────────────────

/// An error that occurs during template evaluation.
///
/// Carries a structured [`EvalErrorKind`], a human-readable message,
/// an optional source [`Span`], and an optional underlying error cause.
///
/// # Error chaining
///
/// When a host's [`EvalContext`](crate::eval::EvalContext) implementation
/// catches an underlying error (database, I/O, etc.), it can preserve the
/// original error chain using [`with_source`](EvalError::with_source):
///
/// ```rust
/// use weaver_lang::EvalError;
///
/// fn example() -> Result<(), EvalError> {
///     let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
///     Err(EvalError::host_error("failed to load entry").with_source(io_err))
/// }
/// ```
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct EvalError {
    pub kind: EvalErrorKind,
    pub span: Option<Span>,
    pub message: String,
    /// The underlying error that caused this evaluation error, if any.
    ///
    /// Wrapped in `Arc` so that `EvalError` remains `Clone`.
    #[source]
    pub source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl EvalError {
    pub fn new(kind: EvalErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            span: None,
            message: message.into(),
            source: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Attach an underlying error cause to this evaluation error.
    ///
    /// This preserves the full error chain for production logging and
    /// debugging. The source is wrapped in an `Arc` so that `EvalError`
    /// remains `Clone`.
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Arc::new(source));
        self
    }

    // Convenience constructors for common error types

    pub fn undefined_variable(scope: &str, name: &str) -> Self {
        Self::new(
            EvalErrorKind::UndefinedVariable,
            format!("undefined variable: {scope}:{name}"),
        )
    }

    pub fn undefined_processor(namespace: &str, name: &str) -> Self {
        Self::new(
            EvalErrorKind::UndefinedCallable,
            format!("undefined processor: {namespace}.{name}"),
        )
    }

    pub fn undefined_command(name: &str) -> Self {
        Self::new(
            EvalErrorKind::UndefinedCallable,
            format!("undefined command: {name}"),
        )
    }

    pub fn type_error(expected: &str, got: &str) -> Self {
        Self::new(
            EvalErrorKind::TypeError,
            format!("expected {expected}, got {got}"),
        )
    }

    pub fn not_iterable(got: &str) -> Self {
        Self::new(
            EvalErrorKind::NotIterable,
            format!("foreach requires an array, got {got}"),
        )
    }

    pub fn host_error(message: impl Into<String>) -> Self {
        Self::new(EvalErrorKind::HostError, message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalErrorKind {
    UndefinedVariable,
    UndefinedCallable,
    TypeError,
    NotIterable,
    ArithmeticError,
    TriggerNotFound,
    DocumentNotFound,
    HostError,
    RecursionLimit,
    /// The evaluation exceeded a configured resource limit (node count
    /// or iteration cap).
    ResourceLimit,
    /// The evaluation was cancelled via an external cancellation token.
    Cancelled,
}