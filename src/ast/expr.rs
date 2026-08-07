use super::{span::Spanned, value::Value};

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Literal value: "hello", 42, true, none
    Literal(Value),

    /// Array literal: [1, 2, "three"]
    ArrayLiteral(Vec<Expr>),

    /// Variable reference: {{scope:name}}
    Variable(VariableRef),

    /// Processor call: @[namespace.name(key: value)]
    ProcessorCall(ProcessorCall),

    /// Command call: $[command(arg1, arg2)]
    /// Commands can mutate state and their return value is optional.
    CommandCall(CommandCall),

    /// Deterministic trigger: <trigger id="entry-id">
    /// Returns the evaluated content of the target entry as a string.
    Trigger(TriggerRef),

    /// Document import: [[DOCUMENT_ID]]
    /// Returns the content of a reusable document block.
    Document(DocumentRef),

    /// Binary operation: a == b, a + b
    BinaryOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },

    /// Unary operation: !condition, -number
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
}

/// A variable reference: an optional scope, a name, and an optional dotted
/// path that indexes into the resolved value.
///
/// The split between [`name`](Self::name) and [`path`](Self::path) is the
/// boundary between the two halves of the language: the host owns the
/// name-space (which scopes exist, what names they hold) and the language
/// owns the value-space (everything after the first dot). In
/// `{{char:alice.stats.hp}}` the host is asked for `alice`; the evaluator
/// walks `stats.hp` into whatever it gets back.
#[derive(Debug, Clone)]
pub struct VariableRef {
    /// `None` for bare loop variables (`{{item}}`), `Some` for scoped
    /// variables (`{{global:name}}`).
    pub scope: Option<String>,
    /// The root variable name — the part the host resolves. Never contains
    /// a dot.
    pub name: String,
    /// Trailing path segments that index into the resolved value. Empty for
    /// a plain reference; `["stats", "hp"]` for `{{char:alice.stats.hp}}`.
    ///
    /// Every non-leaf segment must resolve to a [`crate::Value::Object`].
    pub path: Vec<String>,
}

impl VariableRef {
    /// All dotted segments, root first.
    ///
    /// `{{char:alice.stats.hp}}` yields `["alice", "stats", "hp"]`; a plain
    /// name yields a single element.
    pub fn path_segments(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.path.iter().map(String::as_str))
    }

    /// The dotted name as written, root included: `"alice.stats.hp"`.
    ///
    /// Used for diagnostics — the host never sees this form.
    pub fn full_name(&self) -> String {
        if self.path.is_empty() {
            return self.name.clone();
        }
        self.path_segments().collect::<Vec<_>>().join(".")
    }

    /// Reconstruct the source form, `{{scope:name.path}}` or `{{name}}`.
    ///
    /// This is what lenient mode emits in place of a reference it could not
    /// resolve, so the output is re-parseable.
    pub fn to_source(&self) -> String {
        match &self.scope {
            Some(scope) => format!("{{{{{}:{}}}}}", scope, self.full_name()),
            None => format!("{{{{{}}}}}", self.full_name()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessorCall {
    pub namespace: String,
    pub name: String,
    pub properties: Vec<ProcessorProperty>,
}

#[derive(Debug, Clone)]
pub struct ProcessorProperty {
    pub key: String,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct CommandCall {
    pub name: String,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct TriggerRef {
    /// The target entry id. Evaluates to a string at runtime — either a
    /// literal (`<trigger id="foo">`) or a dynamic expression
    /// (`<trigger id={{scope:name}}>`).
    pub entry_id: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct DocumentRef {
    /// The target document id. Evaluates to a string at runtime — either a
    /// bare identifier (`[[FOO]]`) or a dynamic expression (`[[{{name}}]]`).
    pub document_id: Box<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Comparison
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,

    // Logical
    And,
    Or,

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl BinOp {
    pub fn precedence(&self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::NotEq => 3,
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
}
