use super::expr::{CommandCall, Expr, ExprKind};
use super::span::Spanned;

/// A template is the top-level AST unit. It contains a sequence of nodes
/// whose evaluated string outputs are concatenated to produce the final result.
#[derive(Debug, Clone)]
pub struct Template {
    pub nodes: Vec<Node>,
}

pub type Node = Spanned<NodeKind>;

/// The kinds of content that can appear in a template.
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// Raw text between weaver constructs.
    /// For example, `"Hello, "` in `Hello, {{local:name}}`.
    Literal(String),

    /// An expression in template position. The evaluated result is
    /// converted to a string via [`Value::to_output_string`](crate::Value::to_output_string).
    Expression(ExprKind),

    /// A command in template position. Commands can mutate state and
    /// their return value is optional. When a command appears alone on
    /// a line (standalone), the entire line including surrounding
    /// whitespace and newline is consumed.
    Command(CommandCall),

    /// Conditional block: `{# if ... #}...{# endif #}`
    IfBlock(IfBlock),

    /// Loop block: `{# foreach item in expr #}...{# endforeach #}`
    ForEach(ForEachBlock),
}

#[derive(Debug, Clone)]
pub struct IfBlock {
    pub condition: Expr,
    pub body: Template,
    pub elif_branches: Vec<ElifBranch>,
    pub else_body: Option<Template>,
}

#[derive(Debug, Clone)]
pub struct ElifBranch {
    pub condition: Expr,
    pub body: Template,
}

/// A `{# foreach binding in iterable #}...{# endforeach #}` block.
///
/// The binding is scoped to the loop body — it shadows any existing
/// variable with the same name in the `local` scope and does not
/// leak after the loop ends.
#[derive(Debug, Clone)]
pub struct ForEachBlock {
    /// Name of the variable bound on each iteration.
    pub binding: String,
    /// Expression that must evaluate to an `Array`.
    pub iterable: Expr,
    /// Template body, evaluated once per element.
    pub body: Template,
}
