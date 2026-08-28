//! Whitespace around `{# if #}` and `{# foreach #}` blocks.
//!
//! Blocks used to be resolved by a parser pass that decided from source
//! shape alone whether a block occupied its own line, while expressions
//! and commands were resolved in the evaluator from what actually
//! rendered. Two mechanisms, two answers, and the results disagreed.

use weaver_lang::{ClosureCommand, Registry, SimpleContext, Value, render};

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_command(ClosureCommand::new("noop", |_| Ok(None)));
    reg
}

fn ctx() -> SimpleContext {
    let mut ctx = SimpleContext::new();
    ctx.set("g", "empty", Value::None);
    ctx
}

fn out(src: &str) -> String {
    render(src, &mut ctx(), &registry()).unwrap()
}

// ── the regression ──────────────────────────────────────────────────────

#[test]
fn an_inline_block_body_keeps_the_newline_after_its_closing_tag() {
    // Previously "A\nyesB": the parser marked the block standalone because
    // it sat between two newlines in the source, and stripped the trailing
    // one — without knowing the body renders inline content that supplies
    // no terminator of its own. It could not know; it runs before
    // evaluation.
    assert_eq!(out("A\n{# if true #}yes{# endif #}\nB"), "A\nyes\nB");
}

#[test]
fn inline_and_multiline_block_bodies_agree() {
    // The two ways of writing the same block must produce the same output.
    // This is the property the split broke.
    let inline = "A\n{# if true #}yes{# endif #}\nB";
    let multiline = "A\n{# if true #}\nyes\n{# endif #}\nB";
    assert_eq!(out(inline), "A\nyes\nB");
    assert_eq!(out(multiline), out(inline));
}

#[test]
fn the_same_holds_for_foreach() {
    assert_eq!(
        out("A\n{# foreach x in [1, 2] #}{{x}}{# endforeach #}\nB"),
        "A\n12\nB"
    );
    assert_eq!(
        out("A\n{# foreach x in [1, 2] #}\n{{x}}\n{# endforeach #}\nB"),
        "A\n1\n2\nB"
    );
}

// ── the three rendered-output cases ─────────────────────────────────────

#[test]
fn a_block_that_renders_nothing_consumes_its_line() {
    assert_eq!(out("A\n{# if false #}yes{# endif #}\nB"), "A\nB");
    assert_eq!(out("A\n{# if false #}\nyes\n{# endif #}\nB"), "A\nB");
    assert_eq!(
        out("A\n{# foreach x in [] #}{{x}}{# endforeach #}\nB"),
        "A\nB"
    );
}

#[test]
fn a_block_that_renders_whitespace_also_consumes_its_line() {
    assert_eq!(out("A\n{# if true #}{{g:empty}}{# endif #}\nB"), "A\nB");
}

#[test]
fn a_block_whose_body_ends_in_a_newline_supplies_the_terminator() {
    // The body's own last line ends it, so the closing tag's newline is
    // dropped rather than doubled.
    assert_eq!(out("A\n{# if true #}\nyes\n{# endif #}\nB"), "A\nyes\nB");
}

// ── indentation ─────────────────────────────────────────────────────────

#[test]
fn the_indent_before_an_opening_tag_belongs_to_the_tag_line() {
    // The body carries its own indent; keeping the tag's too would double
    // it. Contrast with an expression, where the indent is the rendered
    // line's own — see `indent_before_an_expression_is_content` below.
    assert_eq!(
        out("A\n  {# if true #}\n  in\n  {# endif #}\nB"),
        "A\n  in\nB"
    );
}

#[test]
fn indent_before_an_expression_is_content() {
    let mut c = ctx();
    c.set("g", "name", "Alice");
    assert_eq!(
        render("A\n  {{g:name}}\nB", &mut c, &registry()).unwrap(),
        "A\n  Alice\nB"
    );
}

#[test]
fn nested_blocks_do_not_accumulate_indent() {
    let src = "<Tag>\n{# if true #}\n  {# if true #}\n  B\n  {# endif #}\n{# endif #}\n</Tag>";
    assert_eq!(out(src), "<Tag>\n  B\n</Tag>");
}

#[test]
fn deeply_nested_inline_blocks() {
    assert_eq!(
        out("{# if true #}{# if true #}deep{# endif #}{# endif #}\nB"),
        "deep\nB"
    );
}

// ── blocks and commands on the same footing ─────────────────────────────

#[test]
fn a_standalone_command_still_consumes_its_line() {
    // Commands are the one construct that always consumes, since their
    // return value is discarded in template position.
    assert_eq!(out("A\n$[noop()]\nB"), "A\nB");
    assert_eq!(out("A\n  $[noop()]\nB"), "A\nB");
}

#[test]
fn a_block_not_at_a_line_start_is_not_standalone() {
    // "lead " precedes it, so the block is mid-line and nothing is
    // consumed on either side.
    assert_eq!(out("lead {# if true #}yes{# endif #}\nB"), "lead yes\nB");
}

#[test]
fn a_block_followed_by_text_rather_than_a_newline_is_not_standalone() {
    assert_eq!(
        out("A\n{# if true #}yes{# endif #} tail\nB"),
        "A\nyes tail\nB"
    );
}

// ── still decided in the parser, correctly ──────────────────────────────

#[test]
fn the_newline_after_an_opening_tag_is_always_a_tag_line_artifact() {
    // Source-only and unambiguous: the newline right after `{# if #}`
    // belongs to that tag's line, never to the body.
    assert_eq!(out("{# if true #}\nbody{# endif #}"), "body");
    assert_eq!(out("{# foreach x in [1] #}\n{{x}}{# endforeach #}"), "1");
}

#[test]
fn transition_tags_get_the_same_treatment() {
    assert_eq!(out("{# if false #}\na\n{# else #}\nb\n{# endif #}"), "b\n");
    assert_eq!(
        out("{# if false #}\na\n{# elif true #}\nb\n{# endif #}"),
        "b\n"
    );
}

// ── blank lines are content, not scaffolding ────────────────────────────

#[test]
fn blank_lines_between_blocks_survive() {
    assert_eq!(
        out("{# if true #}a{# endif #}\n\n{# if true #}b{# endif #}"),
        "a\n\nb"
    );
}

#[test]
fn a_blank_line_inside_a_body_survives() {
    assert_eq!(out("{# if true #}\nx\n\ny\n{# endif #}"), "x\n\ny\n");
}
