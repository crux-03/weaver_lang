use super::span::Spanned;
use super::template::Template;
pub use super::value::PathSegment;
use super::value::Value;

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Literal value: "hello", 42, true, none
    Literal(Value),

    /// Array literal: `[1, 2, "three"]`, or a loop that yields elements.
    ArrayLiteral(Vec<ArrayItem>),

    /// Object literal: `{name: "Alice", hp: 10}`, or a loop that yields
    /// entries.
    ///
    /// Items are held in source order so a duplicate key can be reported
    /// against the second one; evaluation collects them into the sorted
    /// [`Value::Object`](crate::Value::Object) map.
    ObjectLiteral(Vec<ObjectItem>),

    /// A string literal that is itself a text-mode template.
    ///
    /// Only data mode builds these — in text mode a quoted string is a
    /// plain [`Literal`](ExprKind::Literal), because the prose around it is
    /// already the template. A raw string (`r"..."`) is always a literal.
    StringTemplate(Template),

    /// An enum variant: `Custom([...])`, `Pair(a, b)`.
    ///
    /// Sugar for serde's externally tagged representation — one value is
    /// wrapped as `{name: value}`, several as `{name: [values]}` — so a
    /// Rust enum can be written the way it reads in Rust.
    Variant { name: String, values: Vec<Expr> },

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

    /// Index into a value: `items[i]`, `obj["key"]`, `@[p()].name`.
    ///
    /// Only the dynamic cases reach here. A subscript written directly on a
    /// reference with a constant index (`{{c.items[0].name}}`) is folded
    /// into the reference's [`path`](VariableRef::path) at parse time, so it
    /// keeps the same host resolution and lenient-mode passthrough as a
    /// plain dotted path.
    Index { base: Box<Expr>, index: Box<Expr> },
}

/// One `key: value` pair in an object literal.
#[derive(Debug, Clone)]
pub struct ObjectEntry {
    /// An identifier key is a string literal; a quoted key in data mode is
    /// a template, which is how `"{{k}}": v` names a computed key. The
    /// expression must evaluate to a string.
    pub key: Expr,
    pub value: Expr,
}

impl ObjectEntry {
    /// The key if it is known without evaluating anything.
    ///
    /// Used to reject a duplicate at parse time. A computed key can only
    /// collide at evaluation time, where it is reported the same way.
    pub fn static_key(&self) -> Option<&str> {
        match &self.key.node {
            ExprKind::Literal(Value::String(key)) => Some(key),
            _ => None,
        }
    }
}

/// One item of an array literal.
///
/// An item is usually an element, but a loop or a conditional may stand in
/// its place and contribute however many elements it produces.
#[derive(Debug, Clone)]
pub enum ArrayItem {
    Element(Expr),
    ForEach(ValueForEach<ArrayItem>),
    If(ValueIf<ArrayItem>),
}

impl ArrayItem {
    /// The element, when this item is a plain one rather than a loop or a
    /// conditional standing in for one.
    pub fn as_element(&self) -> Option<&Expr> {
        match self {
            ArrayItem::Element(expr) => Some(expr),
            _ => None,
        }
    }
}

/// One item of an object literal — an entry, or something that yields
/// entries.
#[derive(Debug, Clone)]
pub enum ObjectItem {
    Entry(ObjectEntry),
    ForEach(ValueForEach<ObjectItem>),
    If(ValueIf<ObjectItem>),
}

impl ObjectItem {
    /// The entry, when this item is a plain one.
    pub fn as_entry(&self) -> Option<&ObjectEntry> {
        match self {
            ObjectItem::Entry(entry) => Some(entry),
            _ => None,
        }
    }
}

/// `{# foreach x in xs #}` in item position.
///
/// The type parameter is the item kind of the collection it sits in, which
/// is how an element in object position — or an entry in array position —
/// fails to parse rather than evaluating to something surprising.
#[derive(Debug, Clone)]
pub struct ValueForEach<T> {
    pub binding: String,
    pub iterable: Expr,
    pub body: Vec<T>,
}

/// `{# if c #}` in item position, with its `elif` and `else` branches.
#[derive(Debug, Clone)]
pub struct ValueIf<T> {
    /// Condition and body for `if` and each `elif`, in source order.
    pub branches: Vec<(Expr, Vec<T>)>,
    pub else_body: Option<Vec<T>>,
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
    /// Trailing segments that index into the resolved value. Empty for a
    /// plain reference; `[Key("stats"), Key("hp")]` for
    /// `{{char:alice.stats.hp}}`, `[Key("items"), Index(0)]` for
    /// `{{char:alice.items[0]}}`.
    ///
    /// Every non-leaf segment must resolve to something indexable: an
    /// object for a [`Key`](PathSegment::Key), an array for an
    /// [`Index`](PathSegment::Index).
    pub path: Vec<PathSegment>,
}

impl VariableRef {
    /// The leading run of [`Key`](PathSegment::Key) segments — the part of
    /// the path a host can resolve on its own.
    ///
    /// This is what reaches
    /// [`EvalContext::resolve_variable_path`](crate::EvalContext::resolve_variable_path).
    /// It stops at the first subscript, because a host that pushes paths
    /// into storage is indexing named fields, not array positions; whatever
    /// follows is walked by the evaluator against the value it gets back.
    pub fn host_path(&self) -> Vec<String> {
        self.path
            .iter()
            .map_while(|seg| match seg {
                PathSegment::Key(k) => Some(k.clone()),
                PathSegment::Index(_) => None,
            })
            .collect()
    }

    /// The segments after [`host_path`](Self::host_path), which the
    /// evaluator walks itself.
    pub fn local_path(&self) -> &[PathSegment] {
        let keys = self
            .path
            .iter()
            .take_while(|seg| matches!(seg, PathSegment::Key(_)))
            .count();
        &self.path[keys..]
    }

    /// The name as written, root included: `"alice.stats.hp"`,
    /// `"alice.items[0].name"`.
    ///
    /// Used for diagnostics — the host never sees this form.
    pub fn full_name(&self) -> String {
        let mut out = self.name.clone();
        for seg in &self.path {
            match seg {
                PathSegment::Key(k) => {
                    out.push('.');
                    out.push_str(k);
                }
                PathSegment::Index(i) => {
                    out.push('[');
                    out.push_str(&i.to_string());
                    out.push(']');
                }
            }
        }
        out
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
