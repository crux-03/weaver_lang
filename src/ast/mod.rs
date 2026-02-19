//! Abstract syntax tree types for weaver-lang.
//!
//! The AST has two layers:
//!
//! - **Template layer** ([`template`]): Nodes that compose the output string.
//!   A [`Template`] is a sequence of [`Node`]s whose string representations
//!   are concatenated.
//! - **Expression layer** ([`expr`]): Typed values used in conditions,
//!   arguments, and intermediate computation. Expression results are
//!   coerced to strings only when they appear at the template level.

pub mod expr;
pub mod span;
pub mod template;
pub mod value;

// Convenience re-exports
pub use expr::*;
pub use span::{Span, Spanned};
pub use template::*;
pub use value::Value;
