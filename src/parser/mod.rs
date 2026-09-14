//! Template parser, built on [pest](https://pest.rs/).
//!
//! The grammar is defined in `weaver.pest`. This module converts pest's
//! parse tree into the typed AST defined in [`crate::ast`].
//!
//! Use [`parse`] to convert source text into a [`Template`] AST, which
//! can then be evaluated via [`crate::evaluate`].

use pest::Parser;
use pest_derive::Parser;

#[cfg(feature = "data")]
use crate::ast::doc::{InputDecl, InputType, ValueDoc};
use crate::ast::expr::*;
use crate::ast::span::{Span, Spanned};
use crate::ast::template::*;
use crate::ast::value::Value;
use crate::error::ParseError;

#[derive(Parser)]
#[grammar = "parser/weaver.pest"]
struct WeaverParser;

/// Parse source text into a [`Template`] AST.
///
/// Returns a list of [`ParseError`]s if the source contains invalid syntax.
/// Each error carries a source [`Span`](crate::Span) for diagnostic formatting.
pub fn parse(source: &str) -> Result<Template, Vec<ParseError>> {
    let pairs = WeaverParser::parse(Rule::template, source).map_err(|e| {
        let span = pest_span_to_span(&e);
        vec![ParseError::new(span, format!("parse error: {e}"))]
    })?;

    let mut nodes = Vec::new();

    // WeaverParser::parse returns a top-level `template` pair.
    // We need to iterate its inner children (the actual nodes).
    for pair in pairs {
        if pair.as_rule() == Rule::template {
            nodes = build_nodes(pair.into_inner())?;
        }
    }

    let mut template = Template { nodes };
    normalize_whitespace(&mut template);

    let mut flow_errors = Vec::new();
    validate_flow(&template, 0, &mut flow_errors);
    if !flow_errors.is_empty() {
        return Err(flow_errors);
    }

    Ok(template)
}

/// Reject `{# break #}` and `{# continue #}` outside a `foreach`.
///
/// Only `foreach` increments the depth — an `if` inside a loop is still
/// inside that loop, so every branch inherits the enclosing depth.
/// `{# return #}` is legal anywhere and is not checked here.
///
/// The walk stops at the entry boundary by construction: a document or
/// trigger is an expression, not a nested template, so there is nothing
/// to descend into.
fn validate_flow(template: &Template, loop_depth: usize, errors: &mut Vec<ParseError>) {
    for node in &template.nodes {
        match &node.node {
            NodeKind::Break | NodeKind::Continue if loop_depth == 0 => {
                let keyword = match node.node {
                    NodeKind::Break => "break",
                    _ => "continue",
                };
                errors.push(
                    ParseError::new(node.span, format!("`{keyword}` outside of a loop")).with_hint(
                        format!(
                            "`{keyword}` is only valid inside a                              {{# foreach ... #}} block"
                        ),
                    ),
                );
            }
            NodeKind::IfBlock(block) => {
                validate_flow(&block.body, loop_depth, errors);
                for elif in &block.elif_branches {
                    validate_flow(&elif.body, loop_depth, errors);
                }
                if let Some(else_body) = &block.else_body {
                    validate_flow(else_body, loop_depth, errors);
                }
            }
            NodeKind::ForEach(block) => validate_flow(&block.body, loop_depth + 1, errors),
            _ => {}
        }
    }
}

/// Parse a standalone expression from source text.
///
/// This parses a single weaver-lang expression (the same syntax used
/// inside `{# if ... #}` conditions, processor arguments, and command
/// arguments) without any surrounding template structure.
///
/// Use this for evaluating activation conditions, configuration
/// expressions, or any context where you need a typed [`Value`] result
/// rather than a rendered string.
///
/// # Examples
///
/// ```rust
/// use weaver_lang::parse_expr;
///
/// let expr = parse_expr(r#"{{state:level}} > 5"#).unwrap();
/// let expr = parse_expr(r#""hello" == "hello""#).unwrap();
/// let expr = parse_expr(r#"@[array.contains(items: ["a", "b"], value: "a")]"#).unwrap();
/// ```
pub fn parse_expr(source: &str) -> Result<Expr, Vec<ParseError>> {
    let pairs = WeaverParser::parse(Rule::expr, source).map_err(|e| {
        vec![ParseError::new(
            pest_span_to_span(&e),
            format!("expression parse error: {e}"),
        )]
    })?;
    let pair = pairs.into_iter().next().unwrap();
    build_expr(pair, Strings::Literal)
}

/// Parse a data-mode document: declared inputs, then one value.
///
/// The difference from [`parse`] is the entry rule, not the language. A
/// document is a value with holes rather than prose with holes; below that
/// it is the same expressions, the same literals, the same loops.
///
/// String literals inside a data document are themselves text-mode
/// templates, so prose keeps its own syntax where prose belongs. `r"..."`
/// opts a string out.
///
/// ```rust
/// use weaver_lang::parse_value_doc;
///
/// let doc = parse_value_doc(r#"
/// #inputs
/// difficulty: enum("easy", "normal") = "normal"
///
/// { mode: {{input:difficulty}} }
/// "#).unwrap();
/// assert_eq!(doc.inputs.len(), 1);
/// ```
#[cfg(feature = "data")]
pub fn parse_value_doc(source: &str) -> Result<ValueDoc, Vec<ParseError>> {
    let pairs = WeaverParser::parse(Rule::value_doc, source).map_err(|e| {
        vec![ParseError::new(
            pest_span_to_span(&e),
            format!("parse error: {e}"),
        )]
    })?;

    let doc = pairs.into_iter().next().unwrap();
    let mut inputs = Vec::new();
    let mut value = None;

    for pair in doc.into_inner() {
        match pair.as_rule() {
            Rule::inputs_block => inputs = build_inputs_block(pair)?,
            Rule::expr => value = Some(build_expr(pair, Strings::Template)?),
            _ => {}
        }
    }

    Ok(ValueDoc {
        inputs,
        value: value.expect("value_doc always carries an expr"),
    })
}

#[cfg(feature = "data")]
fn build_inputs_block(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Vec<InputDecl>, Vec<ParseError>> {
    let mut decls: Vec<InputDecl> = Vec::new();
    let mut errors = Vec::new();

    for decl_pair in pair.into_inner() {
        if decl_pair.as_rule() != Rule::input_decl {
            continue;
        }
        let span = pair_span(&decl_pair);
        let mut parts = decl_pair.into_inner();
        let name = parts.next().unwrap().as_str().to_string();
        let ty = build_input_type(parts.next().unwrap())?;
        // A default is an ordinary expression, evaluated at instantiation
        // when no value was supplied.
        let default = match parts.next() {
            Some(expr) => Some(build_expr(expr, Strings::Template)?),
            None => None,
        };

        if decls.iter().any(|d| d.name == name) {
            errors.push(
                ParseError::new(span, format!("duplicate input: {name}"))
                    .with_hint("each input may be declared only once"),
            );
            continue;
        }
        decls.push(InputDecl {
            name,
            ty,
            default,
            span,
        });
    }

    if errors.is_empty() {
        Ok(decls)
    } else {
        Err(errors)
    }
}

#[cfg(feature = "data")]
fn build_input_type(pair: pest::iterators::Pair<Rule>) -> Result<InputType, Vec<ParseError>> {
    let span = pair_span(&pair);
    match pair.as_rule() {
        Rule::prim_type => Ok(match pair.as_str() {
            "string" => InputType::String,
            "number" => InputType::Number,
            _ => InputType::Bool,
        }),
        Rule::list_type => Ok(InputType::List(Box::new(build_input_type(
            pair.into_inner().next().unwrap(),
        )?))),
        Rule::ref_type => Ok(InputType::Ref(
            pair.into_inner().next().unwrap().as_str().to_string(),
        )),
        Rule::enum_type => {
            let variants = pair
                .into_inner()
                .filter(|p| p.as_rule() == Rule::quoted_string)
                .map(|p| extract_string_content(p))
                .collect::<Vec<_>>();
            if variants.is_empty() {
                return Err(vec![
                    ParseError::new(span, "enum must list at least one value")
                        .with_hint(r#"for example: enum("easy", "normal")"#),
                ]);
            }
            Ok(InputType::Enum(variants))
        }
        other => Err(vec![ParseError::new(
            span,
            format!("unexpected rule in type position: {other:?}"),
        )]),
    }
}

fn pest_span_to_span(e: &pest::error::Error<Rule>) -> Span {
    match &e.location {
        pest::error::InputLocation::Pos(p) => Span::new(*p, *p + 1),
        pest::error::InputLocation::Span((s, e)) => Span::new(*s, *e),
    }
}

fn pair_span(pair: &pest::iterators::Pair<Rule>) -> Span {
    let s = pair.as_span();
    Span::new(s.start(), s.end())
}

/// Whether a quoted string literal is a value or a template.
///
/// This is the one place the two modes differ below the entry rule. In data
/// mode a string is a text-mode template (`prompt: "You are {{c.name}}"`),
/// so structure comes from data mode and prose from text mode, and an agent
/// system prompt stops being a special case. In text mode the prose around
/// the construct is already the template, so a string inside `$[...]` is
/// just a string.
///
/// The flag does not survive descending *into* a string: the template it
/// holds is parsed in text mode, so a string inside that is a plain literal
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// `Template` is only ever constructed by the data-mode entry point.
#[cfg_attr(not(feature = "data"), allow(dead_code))]
enum Strings {
    Literal,
    Template,
}

// -- Trim markers --------------------------------------------------------

/// The trim markers written on one construct's delimiters.
///
/// These never reach the AST. They are applied while it is built, by
/// trimming the neighbouring `Literal` nodes, so whitespace removal is a
/// property of the source text rather than of what happened to render.
#[derive(Debug, Clone, Copy, Default)]
struct Trim {
    left: bool,
    right: bool,
}

impl Trim {
    /// Read the markers off a construct's own delimiters.
    ///
    /// Only direct children are inspected, so markers belonging to a
    /// nested construct inside an expression are not picked up here.
    fn of(pair: &pest::iterators::Pair<Rule>) -> Self {
        let mut trim = Trim::default();
        for child in pair.clone().into_inner() {
            match child.as_rule() {
                Rule::trim_l => trim.left = true,
                Rule::trim_r => trim.right = true,
                _ => {}
            }
        }
        trim
    }
}

/// A pair's children with the trim markers filtered out, so the existing
/// positional builders keep working unchanged.
fn content_pairs(
    pair: pest::iterators::Pair<'_, Rule>,
) -> impl Iterator<Item = pest::iterators::Pair<'_, Rule>> {
    pair.into_inner()
        .filter(|p| !matches!(p.as_rule(), Rule::trim_l | Rule::trim_r))
}

/// Build a sibling node list and apply each node's outward trim markers to
/// its neighbours.
///
/// Trimming is local to one list: a marker on the first node of a block
/// body has no literal to its left and simply does nothing. Reaching
/// across the block boundary is the job of the block's own tag markers.
fn build_nodes<'a>(
    pairs: impl Iterator<Item = pest::iterators::Pair<'a, Rule>>,
) -> Result<Vec<Node>, Vec<ParseError>> {
    let mut items = Vec::new();
    for pair in pairs {
        if pair.as_rule() == Rule::EOI {
            break;
        }
        if let Some(built) = build_node(pair)? {
            items.push(built);
        }
    }

    let trims: Vec<Trim> = items.iter().map(|(_, t)| *t).collect();
    let mut nodes: Vec<Node> = items.into_iter().map(|(n, _)| n).collect();

    for (i, trim) in trims.iter().enumerate() {
        if trim.left && i > 0 {
            trim_literal_end(&mut nodes[i - 1]);
        }
        if trim.right && i + 1 < nodes.len() {
            trim_literal_start(&mut nodes[i + 1]);
        }
    }

    Ok(nodes)
}

/// Remove all trailing whitespace, newlines included, from a literal node.
/// Any other node kind is left alone.
fn trim_literal_end(node: &mut Node) {
    if let NodeKind::Literal(text) = &mut node.node {
        let trimmed = text.trim_end().len();
        text.truncate(trimmed);
    }
}

/// Remove all leading whitespace, newlines included, from a literal node.
fn trim_literal_start(node: &mut Node) {
    if let NodeKind::Literal(text) = &mut node.node {
        let offset = text.len() - text.trim_start().len();
        text.drain(..offset);
    }
}

/// Apply a `-#}` marker to the start of a block body.
fn trim_body_start(template: &mut Template) {
    if let Some(first) = template.nodes.first_mut() {
        trim_literal_start(first);
    }
}

/// Apply a `{#-` marker to the end of a block body.
fn trim_body_end(template: &mut Template) {
    if let Some(last) = template.nodes.last_mut() {
        trim_literal_end(last);
    }
}

// -- Node building -------------------------------------------------------

/// Build one node, along with the trim markers that face its siblings.
///
/// For a block the outward markers come from opposite tags: the left one
/// from `{#- if`, the right one from `endif -#}`. The markers facing
/// *into* the block are consumed by the block builder itself.
fn build_node(pair: pest::iterators::Pair<Rule>) -> Result<Option<(Node, Trim)>, Vec<ParseError>> {
    let span = pair_span(&pair);
    let trim = Trim::of(&pair);

    let node = match pair.as_rule() {
        Rule::literal_text => {
            let text = pair.as_str().to_string();
            NodeKind::Literal(text)
        }
        Rule::interpolation => NodeKind::Expression(build_interpolation(pair, Strings::Literal)?),
        Rule::processor_call => NodeKind::Expression(build_processor_call(pair, Strings::Literal)?),
        Rule::command_node => {
            // command_node wraps a command_call, which carries the markers.
            let inner = pair.into_inner().next().unwrap();
            let trim = Trim::of(&inner);
            let cmd = build_command_call(inner, Strings::Literal)?;
            return Ok(Some((Spanned::new(NodeKind::Command(cmd), span), trim)));
        }
        Rule::trigger => NodeKind::Expression(build_trigger(pair, Strings::Literal)?),
        Rule::document_ref => NodeKind::Expression(build_document_ref(pair, Strings::Literal)?),
        Rule::if_block => {
            let (block, trim) = build_if_block(pair)?;
            return Ok(Some((Spanned::new(NodeKind::IfBlock(block), span), trim)));
        }
        Rule::foreach_block => {
            let (block, trim) = build_foreach_block(pair)?;
            return Ok(Some((Spanned::new(NodeKind::ForEach(block), span), trim)));
        }
        Rule::break_stmt => NodeKind::Break,
        Rule::stop_stmt => NodeKind::Stop,
        Rule::continue_stmt => NodeKind::Continue,
        Rule::return_stmt => {
            // The inner `expr` is present only for `{# return expr #}`.
            let value = match content_pairs(pair).next() {
                Some(inner) => Some(build_expr(inner, Strings::Literal)?),
                None => None,
            };
            NodeKind::Return(value)
        }
        _ => return Ok(None),
    };

    Ok(Some((Spanned::new(node, span), trim)))
}

// -- Expression building -------------------------------------------------

/// Unwrap `{{ expr }}` down to the expression it holds.
///
/// The interpolation contributes nothing but its delimiters — the node's
/// own span already covers them — so only the inner expression survives.
fn build_interpolation(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<ExprKind, Vec<ParseError>> {
    let inner = content_pairs(pair).next().unwrap();
    Ok(build_expr(inner, strings)?.node)
}

fn build_reference(pair: pest::iterators::Pair<Rule>) -> Result<ExprKind, Vec<ParseError>> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::scoped_ref => {
            let mut parts = inner.into_inner();
            let scope = parts.next().unwrap().as_str().to_string();
            let (name, path) = split_var_path(parts.next().unwrap().as_str());
            Ok(ExprKind::Variable(VariableRef {
                scope: Some(scope),
                name,
                path,
            }))
        }
        Rule::bare_ref => {
            let (name, path) = split_var_path(inner.into_inner().next().unwrap().as_str());
            Ok(ExprKind::Variable(VariableRef {
                scope: None,
                name,
                path,
            }))
        }
        _ => unreachable!(),
    }
}

fn build_processor_call(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<ExprKind, Vec<ParseError>> {
    let mut inner = content_pairs(pair);
    let dotted = inner.next().unwrap().as_str().to_string();
    let (namespace, name) = split_dotted_name(&dotted);

    let mut properties = Vec::new();
    if let Some(prop_list) = inner.next()
        && prop_list.as_rule() == Rule::property_list
    {
        for prop_pair in prop_list.into_inner() {
            if prop_pair.as_rule() == Rule::property {
                let mut prop_inner = prop_pair.into_inner();
                let key = prop_inner.next().unwrap().as_str().to_string();
                let value_expr = build_expr(prop_inner.next().unwrap(), strings)?;
                properties.push(ProcessorProperty {
                    key,
                    value: value_expr,
                });
            }
        }
    }

    Ok(ExprKind::ProcessorCall(ProcessorCall {
        namespace,
        name,
        properties,
    }))
}

fn build_command_call(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<CommandCall, Vec<ParseError>> {
    let mut inner = content_pairs(pair);
    let name = inner.next().unwrap().as_str().to_string();

    let mut args = Vec::new();
    if let Some(arg_list) = inner.next()
        && arg_list.as_rule() == Rule::arg_list
    {
        for arg_pair in arg_list.into_inner() {
            if arg_pair.as_rule() == Rule::expr {
                args.push(build_expr(arg_pair, strings)?);
            }
        }
    }

    Ok(CommandCall { name, args })
}

fn build_trigger(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<ExprKind, Vec<ParseError>> {
    let inner = pair.into_inner().next().unwrap();
    let entry_id = build_id_expr(inner, strings)?;
    Ok(ExprKind::Trigger(TriggerRef {
        entry_id: Box::new(entry_id),
    }))
}

fn build_document_ref(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<ExprKind, Vec<ParseError>> {
    let inner = pair.into_inner().next().unwrap();
    let document_id = build_id_expr(inner, strings)?;
    Ok(ExprKind::Document(DocumentRef {
        document_id: Box::new(document_id),
    }))
}

/// Build the id expression shared by triggers and documents.
///
/// A bare `identifier` (the `[[FOO]]` form) is synthesized into a string
/// literal so the id is uniformly an `Expr`. The `expr` form (from the
/// parenthesized escape hatch) recurses through the operator parser;
/// everything else is a single atom.
fn build_id_expr(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    match pair.as_rule() {
        Rule::identifier => Ok(Spanned::new(
            ExprKind::Literal(Value::String(pair.as_str().to_string())),
            span,
        )),
        Rule::expr => build_expr(pair, strings),
        _ => build_atom(pair, strings),
    }
}

// -- Expression parser (handles operators) -------------------------------

fn build_expr(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    let _span = pair_span(&pair);
    let mut inner = pair.into_inner().peekable();

    // Parse first unary_expr
    let first = inner.next().unwrap();
    let mut left = build_unary_expr(first, strings)?;

    // Parse (bin_op ~ unary_expr)* pairs
    let mut ops: Vec<(BinOp, Expr)> = Vec::new();

    while inner.peek().is_some() {
        let op_pair = inner.next().unwrap();
        let op = parse_bin_op(op_pair.as_str());
        let right_pair = inner.next().unwrap();
        let right = build_unary_expr(right_pair, strings)?;
        ops.push((op, right));
    }

    if ops.is_empty() {
        return Ok(left);
    }

    // Build binary expression tree with operator precedence
    let (result, _) = fold_binary(left, &ops, 0, 0);
    left = result;

    Ok(left)
}

fn build_unary_expr(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    let mut inner = pair.into_inner();

    let first = inner.next().unwrap();

    if first.as_rule() == Rule::unary_op {
        let op = match first.as_str() {
            "!" => UnaryOp::Not,
            "-" => UnaryOp::Neg,
            _ => unreachable!(),
        };
        let operand = build_postfix_expr(inner.next().unwrap(), strings)?;
        Ok(Spanned::new(
            ExprKind::UnaryOp {
                op,
                operand: Box::new(operand),
            },
            span,
        ))
    } else {
        build_postfix_expr(first, strings)
    }
}

/// Build an atom and apply the `[...]` / `.name` suffixes written on it.
///
/// A suffix on a plain reference is folded back into the reference's path
/// when it is constant, so `{{c.items[0].name}}` stays one `VariableRef`
/// and keeps the resolution and lenient-mode behaviour of `{{c.stats.hp}}`.
/// Anything else — a computed index, a suffix on a call — becomes an
/// [`ExprKind::Index`] evaluated against whatever the base produced.
fn build_postfix_expr(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    if pair.as_rule() != Rule::postfix_expr {
        return build_atom(pair, strings);
    }

    let start = pair_span(&pair).start;
    let mut inner = pair.into_inner();
    let mut expr = build_atom(inner.next().unwrap(), strings)?;

    for suffix in inner {
        let span = Span::new(start, pair_span(&suffix).end);
        let index = match suffix.as_rule() {
            Rule::field_suffix => {
                let name = suffix.into_inner().next().unwrap().as_str().to_string();
                Spanned::new(ExprKind::Literal(Value::String(name)), span)
            }
            Rule::index_suffix => build_expr(suffix.into_inner().next().unwrap(), strings)?,
            _ => unreachable!(),
        };
        expr = apply_index(expr, index, span);
    }

    Ok(expr)
}

/// Fold a constant subscript into a reference path, or build an index node.
fn apply_index(base: Expr, index: Expr, span: Span) -> Expr {
    if let ExprKind::Variable(var) = &base.node
        && let ExprKind::Literal(literal) = &index.node
        && let Some(segment) = constant_segment(literal)
    {
        let mut var = var.clone();
        var.path.push(segment);
        return Spanned::new(ExprKind::Variable(var), span);
    }

    Spanned::new(
        ExprKind::Index {
            base: Box::new(base),
            index: Box::new(index),
        },
        span,
    )
}

/// The path segment a literal subscript denotes, if it denotes one.
///
/// A string is a key and a whole non-negative number is an index. A
/// negative or fractional number is neither — it is left to evaluation,
/// which reports it as the error it is rather than silently rounding.
fn constant_segment(literal: &Value) -> Option<PathSegment> {
    match literal {
        Value::String(key) => Some(PathSegment::Key(key.clone())),
        Value::Number(n) if n.fract() == 0.0 && *n >= 0.0 && n.is_finite() => {
            Some(PathSegment::Index(*n as usize))
        }
        _ => None,
    }
}

fn build_atom(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    let rule = pair.as_rule();

    match rule {
        Rule::atom => {
            // atom wraps the actual content â€” unwrap one level
            let inner = pair.into_inner().next().unwrap();
            build_atom(inner, strings)
        }
        Rule::postfix_expr => build_postfix_expr(pair, strings),
        Rule::expr => build_expr(pair, strings),
        Rule::interpolation => {
            let kind = build_interpolation(pair, strings)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::reference => {
            let kind = build_reference(pair)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::processor_call => {
            let kind = build_processor_call(pair, strings)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::command_call => {
            let cmd = build_command_call(pair, strings)?;
            Ok(Spanned::new(ExprKind::CommandCall(cmd), span))
        }
        Rule::trigger => {
            let kind = build_trigger(pair, strings)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::document_ref => {
            let kind = build_document_ref(pair, strings)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::quoted_string => build_string(pair, strings),
        Rule::number => {
            let n: f64 = pair.as_str().parse().map_err(|_| {
                vec![ParseError::new(
                    span,
                    format!("invalid number: {}", pair.as_str()),
                )]
            })?;
            Ok(Spanned::new(ExprKind::Literal(Value::Number(n)), span))
        }
        Rule::bool_literal => {
            let b = pair.as_str() == "true";
            Ok(Spanned::new(ExprKind::Literal(Value::Bool(b)), span))
        }
        Rule::none_literal => Ok(Spanned::new(ExprKind::Literal(Value::None), span)),
        Rule::array_literal => {
            let items = build_array_items(pair.into_inner().collect(), strings)?;
            Ok(Spanned::new(ExprKind::ArrayLiteral(items), span))
        }
        Rule::object_literal => {
            let items = build_object_items(pair.into_inner().collect(), strings)?;
            check_duplicate_keys(&items)?;
            Ok(Spanned::new(ExprKind::ObjectLiteral(items), span))
        }
        Rule::raw_string => {
            // `r"..."` — no escapes, and no template even in data mode.
            let inner = pair
                .into_inner()
                .find(|p| p.as_rule() == Rule::raw_inner)
                .map(|p| p.as_str().to_string())
                .unwrap_or_default();
            Ok(Spanned::new(ExprKind::Literal(Value::String(inner)), span))
        }
        Rule::variant_literal => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let mut values = Vec::new();
            for arg in inner {
                if arg.as_rule() == Rule::expr {
                    values.push(build_expr(arg, strings)?);
                }
            }
            Ok(Spanned::new(ExprKind::Variant { name, values }, span))
        }
        _ => Err(vec![ParseError::new(
            span,
            format!("unexpected rule in atom position: {:?}", rule),
        )]),
    }
}

// -- Collection items ----------------------------------------------------

/// Build the items of an array literal.
///
/// An item is an element, or a loop or conditional that stands in for one
/// and contributes however many elements it produces.
fn build_array_items(
    pairs: Vec<pest::iterators::Pair<Rule>>,
    strings: Strings,
) -> Result<Vec<ArrayItem>, Vec<ParseError>> {
    let mut items = Vec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::expr => items.push(ArrayItem::Element(build_expr(pair, strings)?)),
            Rule::array_foreach => {
                items.push(ArrayItem::ForEach(build_value_foreach(
                    pair,
                    strings,
                    build_array_items,
                )?));
            }
            Rule::array_if => {
                items.push(ArrayItem::If(build_value_if(
                    pair,
                    strings,
                    build_array_items,
                )?));
            }
            _ => {}
        }
    }
    Ok(items)
}

/// Build the items of an object literal — entries, and what yields entries.
fn build_object_items(
    pairs: Vec<pest::iterators::Pair<Rule>>,
    strings: Strings,
) -> Result<Vec<ObjectItem>, Vec<ParseError>> {
    let mut items = Vec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::object_entry => items.push(ObjectItem::Entry(build_object_entry(pair, strings)?)),
            Rule::object_foreach => {
                items.push(ObjectItem::ForEach(build_value_foreach(
                    pair,
                    strings,
                    build_object_items,
                )?));
            }
            Rule::object_if => {
                items.push(ObjectItem::If(build_value_if(
                    pair,
                    strings,
                    build_object_items,
                )?));
            }
            _ => {}
        }
    }
    Ok(items)
}

fn build_object_entry(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<ObjectEntry, Vec<ParseError>> {
    let mut parts = pair.into_inner();
    let key_pair = parts.next().unwrap();
    let key_span = pair_span(&key_pair);

    let key = match key_pair.as_rule() {
        // A quoted key follows whatever a quoted value does, so in data
        // mode `"{{k}}"` is a computed key with no syntax of its own.
        Rule::quoted_string => build_string(key_pair, strings)?,
        _ => Spanned::new(
            ExprKind::Literal(Value::String(key_pair.as_str().to_string())),
            key_span,
        ),
    };
    let value = build_expr(parts.next().unwrap(), strings)?;
    Ok(ObjectEntry { key, value })
}

/// Reject a key written twice.
///
/// Objects are sorted maps, so a repeat would silently discard one of the
/// two values. Only keys known without evaluating anything can be checked
/// here; a computed key that collides is caught at evaluation time.
fn check_duplicate_keys(items: &[ObjectItem]) -> Result<(), Vec<ParseError>> {
    let mut seen: Vec<&str> = Vec::new();
    let mut errors = Vec::new();
    for item in items {
        let ObjectItem::Entry(entry) = item else {
            continue;
        };
        let Some(key) = entry.static_key() else {
            continue;
        };
        if seen.contains(&key) {
            errors.push(
                ParseError::new(entry.key.span, format!("duplicate object key: {key}"))
                    .with_hint("each key may appear only once in an object literal"),
            );
        } else {
            seen.push(key);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// The item builder for one collection kind.
///
/// Passing it to the block builders is what keeps an element out of object
/// position and an entry out of array position: the loop body is built by
/// the same function that built its surroundings, so a mismatch has no rule
/// to parse against.
type ItemBuilder<T> =
    fn(Vec<pest::iterators::Pair<Rule>>, Strings) -> Result<Vec<T>, Vec<ParseError>>;

/// Build a `{# foreach #}` standing in item position.
fn build_value_foreach<T>(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
    build_items: ItemBuilder<T>,
) -> Result<ValueForEach<T>, Vec<ParseError>> {
    let mut inner = pair.into_inner();

    let open = inner.next().unwrap();
    let mut open_parts = content_pairs(open);
    let binding = open_parts.next().unwrap().as_str().to_string();
    let iterable = build_expr(open_parts.next().unwrap(), strings)?;

    let body_pairs = inner
        .filter(|p| p.as_rule() != Rule::foreach_close)
        .collect();

    Ok(ValueForEach {
        binding,
        iterable,
        body: build_items(body_pairs, strings)?,
    })
}

/// Build a `{# if #}` standing in item position, with its branches.
fn build_value_if<T>(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
    build_items: ItemBuilder<T>,
) -> Result<ValueIf<T>, Vec<ParseError>> {
    let mut branches: Vec<(Expr, Vec<T>)> = Vec::new();
    let mut else_body = None;

    // The first condition comes from `if_open`; the items that follow it,
    // up to the first branch tag, are its body.
    let mut condition: Option<Expr> = None;
    let mut body_pairs: Vec<pest::iterators::Pair<Rule>> = Vec::new();

    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::if_open => {
                condition = Some(build_expr(content_pairs(child).next().unwrap(), strings)?);
            }
            Rule::array_elif | Rule::object_elif => {
                if let Some(open) = condition.take() {
                    branches.push((open, build_items(std::mem::take(&mut body_pairs), strings)?));
                }
                let mut parts = child.into_inner();
                let tag = parts.next().unwrap();
                condition = Some(build_expr(content_pairs(tag).next().unwrap(), strings)?);
                body_pairs = parts.collect();
            }
            Rule::array_else | Rule::object_else => {
                if let Some(open) = condition.take() {
                    branches.push((open, build_items(std::mem::take(&mut body_pairs), strings)?));
                }
                let parts: Vec<_> = child
                    .into_inner()
                    .filter(|p| p.as_rule() != Rule::else_tag)
                    .collect();
                else_body = Some(build_items(parts, strings)?);
            }
            Rule::if_close => {}
            _ => body_pairs.push(child),
        }
    }

    // An `if` with no `elif` or `else` never hit a branch tag, so its body
    // is still pending here.
    if let Some(open) = condition.take() {
        branches.push((open, build_items(body_pairs, strings)?));
    }

    Ok(ValueIf {
        branches,
        else_body,
    })
}

/// Build a quoted string, as a value or as a nested text-mode template.
fn build_string(
    pair: pest::iterators::Pair<Rule>,
    strings: Strings,
) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    // The inner span excludes the quotes, so offsets inside the template
    // land where the reader sees them.
    let inner_start = pair
        .clone()
        .into_inner()
        .next()
        .map_or(span.start, |p| p.as_span().start());
    let raw = pair
        .clone()
        .into_inner()
        .next()
        .map_or("", |p| p.as_str())
        .to_string();
    let content = extract_string_content(pair);

    if strings == Strings::Literal {
        return Ok(Spanned::new(
            ExprKind::Literal(Value::String(content)),
            span,
        ));
    }

    let template = parse(&content).map_err(|errors| {
        // Escape sequences shift the offsets, so exact positions are only
        // available for a string that has none. Otherwise the whole literal
        // is the best honest span.
        let shifted = raw.len() == content.len();
        errors
            .into_iter()
            .map(|mut e| {
                e.span = if shifted {
                    Span::new(inner_start + e.span.start, inner_start + e.span.end)
                } else {
                    span
                };
                e
            })
            .collect::<Vec<_>>()
    })?;

    // A string with no constructs is just a string. Folding it keeps the
    // common case off the template path and leaves object keys statically
    // checkable.
    if let [node] = template.nodes.as_slice()
        && let NodeKind::Literal(text) = &node.node
    {
        return Ok(Spanned::new(
            ExprKind::Literal(Value::String(text.clone())),
            span,
        ));
    }
    if template.nodes.is_empty() {
        return Ok(Spanned::new(
            ExprKind::Literal(Value::String(String::new())),
            span,
        ));
    }

    Ok(Spanned::new(ExprKind::StringTemplate(template), span))
}

// -- Control flow building -----------------------------------------------

/// Build an if-block, returning it with the markers that face its siblings.
///
/// A block has four trim positions, and they land in different places:
/// `{#- if` and `endif -#}` face outward and are handed back to the
/// caller; `if -#}` and `{#- endif` face inward and are applied here.
///
/// The inward `{#- endif` marker is applied to the tail of *every* branch,
/// because any of them may be the one that runs. Trimming a branch that
/// does not run costs nothing.
fn build_if_block(pair: pest::iterators::Pair<Rule>) -> Result<(IfBlock, Trim), Vec<ParseError>> {
    let mut inner = pair.into_inner();

    let open = inner.next().unwrap();
    let open_trim = Trim::of(&open);
    let condition = build_expr(content_pairs(open).next().unwrap(), Strings::Literal)?;

    let mut body_pairs = Vec::new();
    let mut elif_branches: Vec<(ElifBranch, Trim)> = Vec::new();
    let mut else_branch: Option<(Template, Trim)> = None;
    let mut close_trim = Trim::default();

    for child in inner {
        match child.as_rule() {
            Rule::elif_branch => {
                let trim = Trim::of(&child);
                let mut elif_inner = content_pairs(child);
                let elif_condition = build_expr(elif_inner.next().unwrap(), Strings::Literal)?;
                let nodes = build_nodes(elif_inner)?;
                elif_branches.push((
                    ElifBranch {
                        condition: elif_condition,
                        body: Template { nodes },
                    },
                    trim,
                ));
            }
            Rule::else_branch => {
                let trim = Trim::of(&child);
                let nodes = build_nodes(content_pairs(child))?;
                else_branch = Some((Template { nodes }, trim));
            }
            Rule::if_close => close_trim = Trim::of(&child),
            _ => body_pairs.push(child),
        }
    }

    let mut body = Template {
        nodes: build_nodes(body_pairs.into_iter())?,
    };
    let mut elifs: Vec<ElifBranch> = Vec::with_capacity(elif_branches.len());
    let mut branch_trims = vec![open_trim];
    for (elif, trim) in elif_branches {
        elifs.push(elif);
        branch_trims.push(trim);
    }
    let mut else_body = None;
    if let Some((body, trim)) = else_branch {
        else_body = Some(body);
        branch_trims.push(trim);
    }

    // Branches are indexed in source order: 0 is the `if` body, then each
    // `elif`, then `else`. `branch_trims[i]` is the marker pair on the tag
    // that OPENS branch i — so its right half trims into branch i, and its
    // left half trims the tail of the branch before it.
    for (i, trim) in branch_trims.iter().enumerate() {
        if trim.right {
            trim_body_start(branch_mut(&mut body, &mut elifs, &mut else_body, i));
        }
        if trim.left && i > 0 {
            trim_body_end(branch_mut(&mut body, &mut elifs, &mut else_body, i - 1));
        }
    }

    // `{#- endif` trims the tail of every branch, since any of them could
    // be the one that runs. Trimming a branch that doesn't run costs
    // nothing, and this keeps the result independent of the condition.
    if close_trim.left {
        for i in 0..branch_trims.len() {
            trim_body_end(branch_mut(&mut body, &mut elifs, &mut else_body, i));
        }
    }

    Ok((
        IfBlock {
            condition,
            body,
            elif_branches: elifs,
            else_body,
        },
        Trim {
            left: open_trim.left,
            right: close_trim.right,
        },
    ))
}

/// Index into an if-block's branches in source order: 0 is the `if` body,
/// then each `elif` in turn, then `else`.
fn branch_mut<'t>(
    body: &'t mut Template,
    elifs: &'t mut [ElifBranch],
    else_body: &'t mut Option<Template>,
    index: usize,
) -> &'t mut Template {
    if index == 0 {
        return body;
    }
    match elifs.get_mut(index - 1) {
        Some(elif) => &mut elif.body,
        // Past the last elif, so this is the else branch. The index comes
        // from `branch_trims`, which is built to match, so it is in range.
        None => else_body.as_mut().expect("branch index out of range"),
    }
}

fn build_foreach_block(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(ForEachBlock, Trim), Vec<ParseError>> {
    let mut inner = pair.into_inner();

    let open = inner.next().unwrap();
    let open_trim = Trim::of(&open);
    // From foreach_open: identifier (binding) then expr (iterable)
    let mut open_inner = content_pairs(open);
    let binding = open_inner.next().unwrap().as_str().to_string();
    let iterable = build_expr(open_inner.next().unwrap(), Strings::Literal)?;

    let mut body_pairs = Vec::new();
    let mut close_trim = Trim::default();
    for child in inner {
        if child.as_rule() == Rule::foreach_close {
            close_trim = Trim::of(&child);
        } else {
            body_pairs.push(child);
        }
    }

    let mut body = Template {
        nodes: build_nodes(body_pairs.into_iter())?,
    };

    // Applied once to the body, so it takes effect on every iteration.
    if open_trim.right {
        trim_body_start(&mut body);
    }
    if close_trim.left {
        trim_body_end(&mut body);
    }

    Ok((
        ForEachBlock {
            binding,
            iterable,
            body,
        },
        Trim {
            left: open_trim.left,
            right: close_trim.right,
        },
    ))
}

// -- Helpers -------------------------------------------------------------

/// Precedence-climbing parser over a flat `(BinOp, Expr)` list.
///
/// Starting from `left` and position `pos` in `ops`, consumes operators
/// whose precedence is ≥ `min_prec`. Higher-precedence operators on the
/// right side are folded first, producing the correct tree shape.
///
/// For example, `a + b * c + d` with standard precedence yields
/// `(a + (b * c)) + d` rather than the naive left-fold `((a + b) * c) + d`.
fn fold_binary(
    mut left: Expr,
    ops: &[(BinOp, Expr)],
    mut pos: usize,
    min_prec: u8,
) -> (Expr, usize) {
    while pos < ops.len() && ops[pos].0.precedence() >= min_prec {
        let op = ops[pos].0;
        let mut right = ops[pos].1.clone();
        pos += 1;

        // Fold any following higher-precedence operators into the right side
        while pos < ops.len() && ops[pos].0.precedence() > op.precedence() {
            let (new_right, new_pos) = fold_binary(right, ops, pos, ops[pos].0.precedence());
            right = new_right;
            pos = new_pos;
        }

        let merged_span = left.span.merge(right.span);
        left = Spanned::new(
            ExprKind::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            merged_span,
        );
    }

    (left, pos)
}

fn split_dotted_name(dotted: &str) -> (String, String) {
    if let Some(pos) = dotted.rfind('.') {
        (dotted[..pos].to_string(), dotted[pos + 1..].to_string())
    } else {
        (String::new(), dotted.to_string())
    }
}

/// Split a variable's dotted name into its root and the path that indexes
/// into the resolved value.
///
/// The opposite grouping from [`split_dotted_name`], which splits a
/// processor's `namespace.name` at the *last* dot: a variable's host-facing
/// name is the *first* segment and everything after it belongs to the
/// value.
fn split_var_path(dotted: &str) -> (String, Vec<PathSegment>) {
    let mut segments = dotted.split('.');
    let root = segments.next().unwrap_or_default().to_string();
    (
        root,
        segments.map(|s| PathSegment::Key(s.to_string())).collect(),
    )
}

fn extract_string_content(pair: pest::iterators::Pair<Rule>) -> String {
    // quoted_string = ${ "\"" ~ string_inner ~ "\"" }
    let inner = pair.into_inner().next().map(|p| p.as_str()).unwrap_or("");

    // Process escape sequences. The set is JSON's, so a JSON string is a
    // weaver string without being rewritten.
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => result.push('\n'),
            Some('t') => result.push('\t'),
            Some('r') => result.push('\r'),
            Some('b') => result.push('\u{8}'),
            Some('f') => result.push('\u{c}'),
            Some('/') => result.push('/'),
            Some('"') => result.push('"'),
            Some('\\') => result.push('\\'),
            Some('u') => result.push(take_unicode_escape(&mut chars)),
            Some(c) => {
                result.push('\\');
                result.push(c);
            }
            None => result.push('\\'),
        }
    }
    result
}

/// Read the four hex digits of a `\uXXXX` escape, pairing surrogates.
///
/// The grammar has already established that four hex digits follow. A lone
/// or mismatched surrogate has no character to stand for and becomes the
/// replacement character rather than failing the parse — the same thing
/// every JSON reader does with one.
fn take_unicode_escape(chars: &mut std::iter::Peekable<std::str::Chars>) -> char {
    let mut code = take_hex4(chars);

    // A high surrogate is only half of a character; the low half follows as
    // its own escape.
    if (0xD800..0xDC00).contains(&code) {
        let mut lookahead = chars.clone();
        if lookahead.next() == Some('\\') && lookahead.next() == Some('u') {
            let low = take_hex4(&mut lookahead);
            if (0xDC00..0xE000).contains(&low) {
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                *chars = lookahead;
            }
        }
    }

    char::from_u32(code).unwrap_or('\u{fffd}')
}

fn take_hex4(chars: &mut std::iter::Peekable<std::str::Chars>) -> u32 {
    let mut code = 0;
    for _ in 0..4 {
        let Some(digit) = chars.peek().and_then(|c| c.to_digit(16)) else {
            break;
        };
        chars.next();
        code = code * 16 + digit;
    }
    code
}

fn parse_bin_op(s: &str) -> BinOp {
    match s {
        "==" => BinOp::Eq,
        "!=" => BinOp::NotEq,
        "<" => BinOp::Lt,
        ">" => BinOp::Gt,
        "<=" => BinOp::LtEq,
        ">=" => BinOp::GtEq,
        "&&" => BinOp::And,
        "||" => BinOp::Or,
        "+" => BinOp::Add,
        "-" => BinOp::Sub,
        "*" => BinOp::Mul,
        "/" => BinOp::Div,
        "%" => BinOp::Mod,
        _ => unreachable!("unknown operator: {s}"),
    }
}

// ── Block tag-line whitespace ───────────────────────────────────────────
//
// The newline that follows an opening or transition tag ({# if #},
// {# elif #}, {# else #}, {# foreach #}) always belongs to that tag's
// line, never to the body's content. Likewise the indent before a
// closing tag on its own line. Both are decidable from the source alone,
// so they are stripped here.
//
// Whether the *block as a whole* occupies a line of output is NOT
// decidable here — it depends on what the body rendered, and a block
// whose branch produces inline content occupies a line differently from
// one that produces a trailing newline or nothing at all. That decision
// belongs to the evaluator, which handles blocks through the same
// `check_standalone` path as expressions and commands.

fn normalize_whitespace(template: &mut Template) {
    // First, unconditionally strip the leading newline from every block
    // body.  The newline after {# if #}, {# elif #}, {# else #}, and
    // {# foreach #} is always a tag-line artifact, never content.
    for node in &mut template.nodes {
        strip_block_body_newlines(node);
    }

    // Recurse into block bodies
    for node in &mut template.nodes {
        match &mut node.node {
            NodeKind::IfBlock(block) => {
                normalize_whitespace(&mut block.body);
                for elif in &mut block.elif_branches {
                    normalize_whitespace(&mut elif.body);
                }
                if let Some(else_body) = &mut block.else_body {
                    normalize_whitespace(else_body);
                }
            }
            NodeKind::ForEach(block) => {
                normalize_whitespace(&mut block.body);
            }
            _ => {}
        }
    }
}

/// Unconditionally strip the leading newline from every branch body of
/// a block-level node.  This removes the newline that follows opening
/// and transition tags ({# if #}, {# elif #}, {# else #}, {# foreach #}).
fn strip_block_body_newlines(node: &mut Node) {
    match &mut node.node {
        NodeKind::IfBlock(block) => {
            strip_body_leading_newline(&mut block.body);
            strip_body_trailing_indent(&mut block.body);
            for elif in &mut block.elif_branches {
                strip_body_leading_newline(&mut elif.body);
                strip_body_trailing_indent(&mut elif.body);
            }
            if let Some(else_body) = &mut block.else_body {
                strip_body_leading_newline(else_body);
                strip_body_trailing_indent(else_body);
            }
        }
        NodeKind::ForEach(block) => {
            strip_body_leading_newline(&mut block.body);
            strip_body_trailing_indent(&mut block.body);
        }
        _ => {}
    }
}

/// Strip trailing spaces/tabs after the final `\n` from a block body's
/// last literal node. This whitespace sits between a newline and the
/// transition/closing tag that follows the body ({# elif #}, {# else #},
/// {# endif #}, {# endforeach #}) — i.e. it is the indent of a tag line
/// and never content. The `\n` guard means inline tags
/// (`stuff {# endif #}`) are untouched, since their preceding text is
/// not whitespace-only after the last newline.
fn strip_body_trailing_indent(template: &mut Template) {
    if let Some(last) = template.nodes.last_mut()
        && let NodeKind::Literal(text) = &mut last.node
        && let Some(ws) = trailing_ws_after_newline_norm(text)
        && ws > 0
    {
        text.truncate(text.len() - ws);
    }
}

/// Strip the leading `\n` (or `\r\n`) from a block body's first literal node.
fn strip_body_leading_newline(template: &mut Template) {
    if let Some(first) = template.nodes.first_mut()
        && let NodeKind::Literal(text) = &mut first.node
    {
        strip_leading_newline_mut(text);
    }
}

/// Remove a leading `\n` or `\r\n` from a string in place.
fn strip_leading_newline_mut(s: &mut String) {
    if s.starts_with("\r\n") {
        s.drain(..2);
    } else if s.starts_with('\n') {
        s.drain(..1);
    }
}

/// If `s` ends with `\n` followed by only spaces/tabs, return the count
/// of those trailing whitespace bytes. Returns `Some(0)` for a bare `\n`
/// at the end.
fn trailing_ws_after_newline_norm(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = bytes.len();

    // Skip trailing spaces/tabs
    while i > 0 && (bytes[i - 1] == b' ' || bytes[i - 1] == b'\t') {
        i -= 1;
    }

    // Must find a newline
    if i > 0 && bytes[i - 1] == b'\n' {
        Some(bytes.len() - i)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `Key` segments a dotted path produces, for comparing against
    /// what the parser built.
    fn keys<const N: usize>(names: [&str; N]) -> Vec<PathSegment> {
        names
            .iter()
            .map(|n| PathSegment::Key((*n).to_string()))
            .collect()
    }

    #[test]
    fn test_literal_text() {
        let template = parse("Hello, world!").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Literal(s) => assert_eq!(s, "Hello, world!"),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_variable() {
        let template = parse("{{global:name}}").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.scope, Some("global".to_string()));
                assert_eq!(v.name, "name");
            }
            _ => panic!("expected variable"),
        }
    }

    #[test]
    fn test_bare_variable() {
        let template = parse("{{item}}").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.scope, None);
                assert_eq!(v.name, "item");
            }
            _ => panic!("expected bare variable"),
        }
    }

    #[test]
    fn test_dotted_scoped_variable() {
        // The root is the host-facing name; the rest indexes into the value.
        let template = parse("{{char:alice.inventory}}").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.scope, Some("char".to_string()));
                assert_eq!(v.name, "alice");
                assert_eq!(v.path, keys(["inventory"]));
                assert_eq!(v.host_path(), ["inventory"]);
                assert_eq!(v.full_name(), "alice.inventory");
            }
            _ => panic!("expected variable"),
        }
    }

    #[test]
    fn test_deep_dotted_scoped_variable() {
        let template = parse("{{world:region.north.weather}}").unwrap();
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.scope, Some("world".to_string()));
                assert_eq!(v.name, "region");
                assert_eq!(v.path, keys(["north", "weather"]));
            }
            _ => panic!("expected variable"),
        }
    }

    #[test]
    fn test_bare_variable_takes_a_path() {
        // A loop binding is still a single-segment name, but the value it
        // holds can be indexed: {# foreach npc in ... #}{{npc.name}}.
        let template = parse("{{item.field}}").unwrap();
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.scope, None);
                assert_eq!(v.name, "item");
                assert_eq!(v.path, keys(["field"]));
            }
            _ => panic!("expected bare variable"),
        }
    }

    #[test]
    fn test_plain_variable_has_empty_path() {
        let template = parse("{{global:hp}}").unwrap();
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Variable(v)) => {
                assert_eq!(v.name, "hp");
                assert!(v.path.is_empty());
                assert_eq!(v.full_name(), "hp");
            }
            _ => panic!("expected variable"),
        }
    }

    #[test]
    fn test_processor_namespace_splits_at_last_dot() {
        // Variables split at the FIRST dot, processors at the last — the
        // two dotted forms mean different things and must not converge.
        let template = parse("@[core.text.upper(text: \"x\")]").unwrap();
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::ProcessorCall(p)) => {
                assert_eq!(p.namespace, "core.text");
                assert_eq!(p.name, "upper");
            }
            _ => panic!("expected processor call"),
        }
    }

    #[test]
    fn test_mixed_template() {
        let template = parse("Hello, {{local:name}}! You have {{global:count}} items.").unwrap();
        // "Hello, " + var + "! You have " + var + " items."
        assert_eq!(template.nodes.len(), 5);
    }

    #[test]
    fn test_processor_call() {
        let template = parse(r#"@[core.weaver.rng(min: 1, max: 10)]"#).unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::ProcessorCall(p)) => {
                assert_eq!(p.namespace, "core.weaver");
                assert_eq!(p.name, "rng");
                assert_eq!(p.properties.len(), 2);
                assert_eq!(p.properties[0].key, "min");
                assert_eq!(p.properties[1].key, "max");
            }
            _ => panic!("expected processor call"),
        }
    }

    #[test]
    fn test_command_call() {
        let template = parse(r#"$[set_var("global:name", "Alice")]"#).unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Command(cmd) => {
                assert_eq!(cmd.name, "set_var");
                assert_eq!(cmd.args.len(), 2);
            }
            _ => panic!("expected command"),
        }
    }

    #[test]
    fn test_processor_with_array_property() {
        let template = parse(r#"@[core.weaver.wildcard(items: ["a", "b", "c"])]"#).unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::ProcessorCall(p)) => {
                assert_eq!(p.namespace, "core.weaver");
                assert_eq!(p.name, "wildcard");
                assert_eq!(p.properties.len(), 1);
                assert_eq!(p.properties[0].key, "items");
            }
            _ => panic!("expected processor call"),
        }
    }

    #[test]
    fn test_if_block() {
        let template = parse("{# if {{global:x}} == 5 #}yes{# else #}no{# endif #}").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::IfBlock(block) => {
                assert_eq!(block.body.nodes.len(), 1);
                assert!(block.else_body.is_some());
            }
            _ => panic!("expected if block"),
        }
    }

    #[test]
    fn test_foreach_block() {
        let template =
            parse(r#"{# foreach item in ["a", "b", "c"] #}{{local:item}}, {# endforeach #}"#)
                .unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::ForEach(block) => {
                assert_eq!(block.binding, "item");
            }
            _ => panic!("expected foreach"),
        }
    }

    #[test]
    fn test_trigger() {
        let template = parse(r#"<trigger id="dark_forest">"#).unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Trigger(t)) => match &t.entry_id.node {
                ExprKind::Literal(Value::String(s)) => assert_eq!(s, "dark_forest"),
                other => panic!("expected string literal id, got {other:?}"),
            },
            _ => panic!("expected trigger"),
        }
    }

    #[test]
    fn test_document_ref() {
        let template = parse("[[LORE_INTRO]]").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Document(d)) => match &d.document_id.node {
                ExprKind::Literal(Value::String(s)) => assert_eq!(s, "LORE_INTRO"),
                other => panic!("expected string literal id, got {other:?}"),
            },
            _ => panic!("expected document ref"),
        }
    }
}

#[cfg(test)]
mod multiline_tests {
    use super::*;

    #[test]
    fn test_multiline_processor() {
        let src = r#"@[core.pick_random(
    items: [
        "item1",
        12,
        @[core.rand_range(min: 1, max: 10)],
        <trigger id="some_entry">
    ]
)]"#;
        let result = parse(src);
        if let Err(errs) = &result {
            for e in errs {
                eprintln!("PARSE ERROR: {}", e.message);
            }
        }
        let template = result.unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::ProcessorCall(p)) => {
                assert_eq!(p.namespace, "core");
                assert_eq!(p.name, "pick_random");
                assert_eq!(p.properties.len(), 1);
                assert_eq!(p.properties[0].key, "items");
                // The value should be an array with 4 elements
                match &p.properties[0].value.node {
                    ExprKind::ArrayLiteral(elems) => {
                        assert_eq!(elems.len(), 4);
                        // Check types: string, number, processor, trigger
                        let kind = |i: usize| &elems[i].as_element().unwrap().node;
                        assert!(matches!(kind(0), ExprKind::Literal(Value::String(_))));
                        assert!(matches!(kind(1), ExprKind::Literal(Value::Number(_))));
                        assert!(matches!(kind(2), ExprKind::ProcessorCall(_)));
                        assert!(matches!(kind(3), ExprKind::Trigger(_)));
                    }
                    other => panic!("expected array literal, got {other:?}"),
                }
            }
            other => panic!("expected processor call, got {other:?}"),
        }
    }
}
