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

#[derive(Debug, Clone)]
pub struct VariableRef {
    /// `None` for bare loop variables (`{{item}}`), `Some` for scoped
    /// variables (`{{global:name}}`).
    pub scope: Option<String>,
    /// The variable name. For scoped variables this may be a dotted path
    /// (`{{char:alice.inventory}}` yields `name == "alice.inventory"`). The
    /// path is opaque to the evaluator and passed to the host verbatim;
    /// use [`VariableRef::path_segments`] to split it. Bare loop bindings
    /// are always single-segment.
    pub name: String,
}

impl VariableRef {
    /// Split [`name`](Self::name) into its dotted path segments.
    ///
    /// `"alice.inventory"` yields `["alice", "inventory"]`; a plain name
    /// yields a single-element slice. Useful for hosts that store nested
    /// state and want to walk the path themselves.
    pub fn path_segments(&self) -> impl Iterator<Item = &str> {
        self.name.split('.')
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
}

impl BinOp {
    pub fn precedence(&self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::NotEq => 3,
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
}
