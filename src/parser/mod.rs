//! Template parser, built on [pest](https://pest.rs/).
//!
//! The grammar is defined in `weaver.pest`. This module converts pest's
//! parse tree into the typed AST defined in [`crate::ast`].
//!
//! Use [`parse`] to convert source text into a [`Template`] AST, which
//! can then be evaluated via [`crate::evaluate`].

use pest::Parser;
use pest_derive::Parser;

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
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::EOI => break,
                    _ => {
                        if let Some(node) = build_node(inner)? {
                            nodes.push(node);
                        }
                    }
                }
            }
        }
    }

    let mut template = Template { nodes };
    normalize_whitespace(&mut template);
    Ok(template)
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

// -- Node building -------------------------------------------------------

fn build_node(pair: pest::iterators::Pair<Rule>) -> Result<Option<Node>, Vec<ParseError>> {
    let span = pair_span(&pair);

    match pair.as_rule() {
        Rule::literal_text => {
            let text = pair.as_str().to_string();
            Ok(Some(Spanned::new(NodeKind::Literal(text), span)))
        }
        Rule::variable => {
            let expr_kind = build_variable(pair)?;
            Ok(Some(Spanned::new(NodeKind::Expression(expr_kind), span)))
        }
        Rule::processor_call => {
            let expr_kind = build_processor_call(pair)?;
            Ok(Some(Spanned::new(NodeKind::Expression(expr_kind), span)))
        }
        Rule::command_node => {
            // command_node wraps a command_call
            let inner = pair.into_inner().next().unwrap();
            let cmd = build_command_call(inner)?;
            Ok(Some(Spanned::new(NodeKind::Command(cmd), span)))
        }
        Rule::trigger => {
            let expr_kind = build_trigger(pair)?;
            Ok(Some(Spanned::new(NodeKind::Expression(expr_kind), span)))
        }
        Rule::document_ref => {
            let expr_kind = build_document_ref(pair)?;
            Ok(Some(Spanned::new(NodeKind::Expression(expr_kind), span)))
        }
        Rule::if_block => {
            let if_block = build_if_block(pair)?;
            Ok(Some(Spanned::new(NodeKind::IfBlock(if_block), span)))
        }
        Rule::foreach_block => {
            let foreach = build_foreach_block(pair)?;
            Ok(Some(Spanned::new(NodeKind::ForEach(foreach), span)))
        }
        _ => Ok(None),
    }
}

// -- Expression building -------------------------------------------------

fn build_variable(pair: pest::iterators::Pair<Rule>) -> Result<ExprKind, Vec<ParseError>> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::scoped_var => {
            let mut parts = inner.into_inner();
            let scope = parts.next().unwrap().as_str().to_string();
            let name = parts.next().unwrap().as_str().to_string();
            Ok(ExprKind::Variable(VariableRef { scope: Some(scope), name }))
        }
        Rule::bare_var => {
            let name = inner.into_inner().next().unwrap().as_str().to_string();
            Ok(ExprKind::Variable(VariableRef { scope: None, name }))
        }
        _ => unreachable!(),
    }
}

fn build_processor_call(pair: pest::iterators::Pair<Rule>) -> Result<ExprKind, Vec<ParseError>> {
    let mut inner = pair.into_inner();
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
                let value_expr = build_expr(prop_inner.next().unwrap())?;
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

fn build_command_call(pair: pest::iterators::Pair<Rule>) -> Result<CommandCall, Vec<ParseError>> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut args = Vec::new();
    if let Some(arg_list) = inner.next()
        && arg_list.as_rule() == Rule::arg_list
    {
        for arg_pair in arg_list.into_inner() {
            if arg_pair.as_rule() == Rule::expr {
                args.push(build_expr(arg_pair)?);
            }
        }
    }

    Ok(CommandCall { name, args })
}

fn build_trigger(pair: pest::iterators::Pair<Rule>) -> Result<ExprKind, Vec<ParseError>> {
    let mut inner = pair.into_inner();
    let entry_id = extract_string_content(inner.next().unwrap());
    Ok(ExprKind::Trigger(TriggerRef { entry_id }))
}

fn build_document_ref(pair: pest::iterators::Pair<Rule>) -> Result<ExprKind, Vec<ParseError>> {
    let mut inner = pair.into_inner();
    let document_id = inner.next().unwrap().as_str().to_string();
    Ok(ExprKind::Document(DocumentRef { document_id }))
}

// -- Expression parser (handles operators) -------------------------------

fn build_expr(pair: pest::iterators::Pair<Rule>) -> Result<Expr, Vec<ParseError>> {
    let _span = pair_span(&pair);
    let mut inner = pair.into_inner().peekable();

    // Parse first unary_expr
    let first = inner.next().unwrap();
    let mut left = build_unary_expr(first)?;

    // Parse (bin_op ~ unary_expr)* pairs
    let mut ops: Vec<(BinOp, Expr)> = Vec::new();

    while inner.peek().is_some() {
        let op_pair = inner.next().unwrap();
        let op = parse_bin_op(op_pair.as_str());
        let right_pair = inner.next().unwrap();
        let right = build_unary_expr(right_pair)?;
        ops.push((op, right));
    }

    if ops.is_empty() {
        return Ok(left);
    }

    // Build binary expression tree (left-to-right)
    for (op, right) in ops {
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

    Ok(left)
}

fn build_unary_expr(pair: pest::iterators::Pair<Rule>) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    let mut inner = pair.into_inner();

    let first = inner.next().unwrap();

    if first.as_rule() == Rule::unary_op {
        let op = match first.as_str() {
            "!" => UnaryOp::Not,
            "-" => UnaryOp::Neg,
            _ => unreachable!(),
        };
        let operand = build_atom(inner.next().unwrap())?;
        Ok(Spanned::new(
            ExprKind::UnaryOp {
                op,
                operand: Box::new(operand),
            },
            span,
        ))
    } else {
        build_atom(first)
    }
}

fn build_atom(pair: pest::iterators::Pair<Rule>) -> Result<Expr, Vec<ParseError>> {
    let span = pair_span(&pair);
    let rule = pair.as_rule();

    match rule {
        Rule::atom => {
            // atom wraps the actual content â€” unwrap one level
            let inner = pair.into_inner().next().unwrap();
            build_atom(inner)
        }
        Rule::expr => build_expr(pair),
        Rule::variable => {
            let kind = build_variable(pair)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::processor_call => {
            let kind = build_processor_call(pair)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::command_call => {
            let cmd = build_command_call(pair)?;
            Ok(Spanned::new(ExprKind::CommandCall(cmd), span))
        }
        Rule::trigger => {
            let kind = build_trigger(pair)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::document_ref => {
            let kind = build_document_ref(pair)?;
            Ok(Spanned::new(kind, span))
        }
        Rule::quoted_string => {
            let s = extract_string_content(pair);
            Ok(Spanned::new(ExprKind::Literal(Value::String(s)), span))
        }
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
            let mut elements = Vec::new();
            for inner_pair in pair.into_inner() {
                if inner_pair.as_rule() == Rule::expr {
                    elements.push(build_expr(inner_pair)?);
                }
            }
            Ok(Spanned::new(ExprKind::ArrayLiteral(elements), span))
        }
        _ => Err(vec![ParseError::new(
            span,
            format!("unexpected rule in atom position: {:?}", rule),
        )]),
    }
}

// -- Control flow building -----------------------------------------------

fn build_if_block(pair: pest::iterators::Pair<Rule>) -> Result<IfBlock, Vec<ParseError>> {
    let mut inner = pair.into_inner();

    // First child is the condition (from if_open)
    let condition = build_expr(inner.next().unwrap())?;

    // Collect body nodes until we hit elif_branch, else_branch, or end
    let mut body_nodes = Vec::new();
    let mut elif_branches = Vec::new();
    let mut else_body = None;

    for child in inner {
        match child.as_rule() {
            Rule::elif_branch => {
                let mut elif_inner = child.into_inner();
                let elif_condition = build_expr(elif_inner.next().unwrap())?;
                let mut elif_nodes = Vec::new();
                for elif_child in elif_inner {
                    if let Some(node) = build_node(elif_child)? {
                        elif_nodes.push(node);
                    }
                }
                elif_branches.push(ElifBranch {
                    condition: elif_condition,
                    body: Template { nodes: elif_nodes },
                });
            }
            Rule::else_branch => {
                let mut else_nodes = Vec::new();
                for else_child in child.into_inner() {
                    if let Some(node) = build_node(else_child)? {
                        else_nodes.push(node);
                    }
                }
                else_body = Some(Template { nodes: else_nodes });
            }
            _ => {
                if let Some(node) = build_node(child)? {
                    body_nodes.push(node);
                }
            }
        }
    }

    Ok(IfBlock {
        condition,
        body: Template { nodes: body_nodes },
        elif_branches,
        else_body,
    })
}

fn build_foreach_block(pair: pest::iterators::Pair<Rule>) -> Result<ForEachBlock, Vec<ParseError>> {
    let mut inner = pair.into_inner();

    // From foreach_open: identifier (binding) then expr (iterable)
    let binding = inner.next().unwrap().as_str().to_string();
    let iterable = build_expr(inner.next().unwrap())?;

    let mut body_nodes = Vec::new();
    for child in inner {
        if let Some(node) = build_node(child)? {
            body_nodes.push(node);
        }
    }

    Ok(ForEachBlock {
        binding,
        iterable,
        body: Template { nodes: body_nodes },
    })
}

// -- Helpers -------------------------------------------------------------

fn split_dotted_name(dotted: &str) -> (String, String) {
    if let Some(pos) = dotted.rfind('.') {
        (dotted[..pos].to_string(), dotted[pos + 1..].to_string())
    } else {
        (String::new(), dotted.to_string())
    }
}

fn extract_string_content(pair: pest::iterators::Pair<Rule>) -> String {
    // quoted_string = ${ "\"" ~ string_inner ~ "\"" }
    let inner = pair.into_inner().next().map(|p| p.as_str()).unwrap_or("");

    // Process escape sequences
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }
    result
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
        _ => unreachable!("unknown operator: {s}"),
    }
}

// ── Standalone block whitespace normalization ───────────────────────────
//
// Control flow tags ({# if #}, {# elif #}, {# else #}, {# endif #},
// {# foreach #}, {# endforeach #}) that appear alone on a line should
// not produce blank lines in the output. This normalization pass strips
// the newlines that belong to "tag lines" from the literal text nodes
// in the AST.
//
// The rules mirror the existing standalone command behavior:
//
// 1. A block-level node (IfBlock, ForEach) is "standalone" if:
//    - The preceding literal ends with \n (+ optional whitespace), OR
//      the block is the first node.
//    - The following literal starts with \n, OR the block is the last
//      node (but not if it's the ONLY node — that's inline).
//
// 2. For standalone blocks, we strip:
//    - Trailing whitespace indent from the preceding literal (same line
//      as the opening tag, e.g. spaces before {# if #}).
//    - Leading \n from the following literal (the newline after {# endif #}
//      or {# endforeach #}).
//    - Leading \n from each block body's first literal (the newline after
//      {# if #}, {# elif #}, {# else #}, {# foreach #}).

fn normalize_whitespace(template: &mut Template) {
    // First, unconditionally strip the leading newline from every block
    // body.  The newline after {# if #}, {# elif #}, {# else #}, and
    // {# foreach #} is always a tag-line artifact, never content.
    for node in &mut template.nodes {
        strip_block_body_newlines(node);
    }

    // Then handle the outer context: for standalone blocks, trim the
    // surrounding whitespace and the newline after the closing tag.
    normalize_standalone_blocks(&mut template.nodes);

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
            for elif in &mut block.elif_branches {
                strip_body_leading_newline(&mut elif.body);
            }
            if let Some(else_body) = &mut block.else_body {
                strip_body_leading_newline(else_body);
            }
        }
        NodeKind::ForEach(block) => {
            strip_body_leading_newline(&mut block.body);
        }
        _ => {}
    }
}

fn normalize_standalone_blocks(nodes: &mut [Node]) {
    let len = nodes.len();

    // Process in reverse so trimming earlier nodes doesn't affect later indices
    for i in (0..len).rev() {
        let is_block = matches!(&nodes[i].node, NodeKind::IfBlock(_) | NodeKind::ForEach(_));
        if !is_block {
            continue;
        }

        // ── Check preceding context ────────────────────────────────
        let at_start = i == 0;
        let (preceding_ok, ws_trim) = if at_start {
            (true, 0)
        } else {
            match &nodes[i - 1].node {
                NodeKind::Literal(text) => {
                    if text.is_empty() {
                        // Empty literal (e.g. from a prior strip) — treat
                        // like being at the start of a line.
                        (true, 0)
                    } else if let Some(ws) = trailing_ws_after_newline_norm(text) {
                        (true, ws)
                    } else {
                        (false, 0)
                    }
                }
                _ => (false, 0),
            }
        };

        if !preceding_ok {
            continue;
        }

        // ── Check following context ────────────────────────────────
        let has_following = i + 1 < nodes.len();
        let following_ok = if !has_following {
            true
        } else {
            match &nodes[i + 1].node {
                NodeKind::Literal(text) => text.starts_with('\n') || text.starts_with("\r\n"),
                _ => false,
            }
        };

        if !following_ok {
            continue;
        }

        // ── Standalone confirmed — apply trimming ──────────────────

        // 1. Trim trailing whitespace (indent) from preceding literal
        if ws_trim > 0
            && let NodeKind::Literal(text) = &mut nodes[i - 1].node
        {
            text.truncate(text.len() - ws_trim);
        }

        // 2. Strip leading newline from following literal
        if has_following && let NodeKind::Literal(text) = &mut nodes[i + 1].node {
            strip_leading_newline_mut(text);
        }
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
            NodeKind::Expression(ExprKind::Trigger(t)) => {
                assert_eq!(t.entry_id, "dark_forest");
            }
            _ => panic!("expected trigger"),
        }
    }

    #[test]
    fn test_document_ref() {
        let template = parse("[[LORE_INTRO]]").unwrap();
        assert_eq!(template.nodes.len(), 1);
        match &template.nodes[0].node {
            NodeKind::Expression(ExprKind::Document(d)) => {
                assert_eq!(d.document_id, "LORE_INTRO");
            }
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
                        assert!(matches!(
                            &elems[0].node,
                            ExprKind::Literal(Value::String(_))
                        ));
                        assert!(matches!(
                            &elems[1].node,
                            ExprKind::Literal(Value::Number(_))
                        ));
                        assert!(matches!(&elems[2].node, ExprKind::ProcessorCall(_)));
                        assert!(matches!(&elems[3].node, ExprKind::Trigger(_)));
                    }
                    other => panic!("expected array literal, got {other:?}"),
                }
            }
            other => panic!("expected processor call, got {other:?}"),
        }
    }
}