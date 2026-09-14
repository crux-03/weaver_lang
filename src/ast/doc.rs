//! Data-mode documents: a declared set of inputs and one value.
//!
//! A [`ValueDoc`] is what data mode parses. Its inputs are *declared* rather
//! than discovered, so an instantiation form can be generated from the
//! document instead of authors guessing which fields exist.

use super::expr::Expr;
use super::span::Span;

/// A parsed data-mode document.
#[derive(Debug, Clone)]
pub struct ValueDoc {
    /// Declarations from the `#inputs` block, in source order. Empty when
    /// the document has no block.
    pub inputs: Vec<InputDecl>,
    /// The document's value, evaluated once the inputs are bound.
    pub value: Expr,
}

/// One line of an `#inputs` block: `characters: [Ref<Character>]`.
#[derive(Debug, Clone)]
pub struct InputDecl {
    pub name: String,
    pub ty: InputType,
    /// The expression after `=`. An input without one is required.
    pub default: Option<Expr>,
    /// The declaration as written, for reporting a bad or missing value
    /// against the line that asked for it.
    pub span: Span,
}

impl InputDecl {
    pub fn is_required(&self) -> bool {
        self.default.is_none()
    }
}

/// The closed type vocabulary an input may declare.
///
/// It names a *kind*, not a shape. An entity has several representations in
/// a host application — an id, a full struct, a summary — and this models
/// none of them: the id is the value, and the rest are projections the host
/// resolves. Keeping the vocabulary closed is what lets a UI render
/// `[Ref<Character>]` as a character multi-picker and `enum(...)` as a
/// dropdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputType {
    String,
    Number,
    Bool,
    /// `enum("easy", "normal", "brutal")` — one of a fixed set of strings.
    Enum(Vec<String>),
    /// `[T]` — a list of the inner type.
    List(Box<InputType>),
    /// `Ref<Character>` — an id that must resolve to a live entity of that
    /// kind. The kind comes from the host-populated registry, and whether a
    /// given id resolves is answered by
    /// [`EvalContext::validate_input`](crate::EvalContext::validate_input).
    Ref(String),
}

impl std::fmt::Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::String => write!(f, "string"),
            InputType::Number => write!(f, "number"),
            InputType::Bool => write!(f, "bool"),
            InputType::Enum(variants) => {
                write!(f, "enum(")?;
                for (i, v) in variants.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v:?}")?;
                }
                write!(f, ")")
            }
            InputType::List(inner) => write!(f, "[{inner}]"),
            InputType::Ref(kind) => write!(f, "Ref<{kind}>"),
        }
    }
}
