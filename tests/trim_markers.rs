//! Explicit whitespace control: `{#- -#}`, `{{- -}}`, `$[- -]`, `@[- -]`.
//!
//! A `-` immediately inside a delimiter removes all whitespace on that
//! side, newlines included.

use weaver_lang::{
    ClosureCommand, ClosureProcessor, Registry, SimpleContext, Value, parse, render,
};

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_command(ClosureCommand::new("noop", |_| Ok(None)));
    reg.register_command(ClosureCommand::new("val", |a| Ok(a.into_iter().next())));
    reg.register_processor(ClosureProcessor::new("t", "id", |props| {
        Ok(props.get("v").cloned().unwrap_or(Value::None))
    }));
    reg
}

fn ctx() -> SimpleContext {
    let mut ctx = SimpleContext::new();
    ctx.set("g", "name", "Alice");
    ctx.set("g", "hp", 12i64);
    ctx.set("g", "empty", Value::None);
    ctx
}

fn out(src: &str) -> String {
    render(src, &mut ctx(), &registry()).unwrap()
}

#[test]
fn left_marker_eats_preceding_whitespace() {
    assert_eq!(out("A   {{-g:name}}"), "A".to_owned() + "Alice");
}

#[test]
fn right_marker_eats_following_whitespace() {
    assert_eq!(out("{{g:name-}}   B"), "AliceB");
}

#[test]
fn both_markers() {
    assert_eq!(out("A   {{-g:name-}}   B"), "AAliceB");
}

#[test]
fn markers_eat_newlines_too() {
    assert_eq!(out("A\n\n{{-g:name-}}\n\nB"), "AAliceB");
}

#[test]
fn a_marker_with_no_neighbour_is_harmless() {
    assert_eq!(out("{{-g:name-}}"), "Alice");
}

#[test]
fn unmarked_variables_are_unaffected() {
    assert_eq!(out("A   {{g:name}}   B"), "A   Alice   B");
}

#[test]
fn command_markers() {
    assert_eq!(out("A   $[-val(\"x\")-]   B"), "AxB");
}

#[test]
fn processor_markers() {
    assert_eq!(out("A   @[-t.id(v: \"x\")-]   B"), "AxB");
}

#[test]
fn markers_do_not_disturb_negative_arguments() {
    assert_eq!(out("$[val(-1)]"), "-1");
    assert_eq!(out("A  $[-val(-1)-]  B"), "A-1B");
}

#[test]
fn if_open_and_close_markers() {
    assert_eq!(out("A\n{#- if true -#}\n  yes\n{#- endif -#}\nB"), "AyesB");
}

#[test]
fn close_marker_trims_whichever_branch_runs() {
    let src = "{# if {{g:hp}} < 20 #}low   {#- else #}high   {#- endif #}!";
    assert_eq!(out(src), "low!");

    let mut c = ctx();
    c.set("g", "hp", 99i64);
    assert_eq!(render(src, &mut c, &registry()).unwrap(), "high!");
}

#[test]
fn elif_marker_trims_the_preceding_branch() {
    let src = "{# if false #}a   {#- elif true #}b{# endif #}";
    assert_eq!(out(src), "b");
}

#[test]
fn else_marker_trims_the_preceding_branch() {
    let src = "{# if true #}a   {#- else #}b{# endif #}";
    assert_eq!(out(src), "a");
}

#[test]
fn foreach_markers_apply_on_every_iteration() {
    assert_eq!(
        out("{# foreach x in [1, 2, 3] -#}\n   {{x}}\n{#- endforeach #}"),
        "123"
    );
}

#[test]
fn foreach_outer_markers() {
    assert_eq!(
        out("A\n\n{#- foreach x in [1, 2] #}{{x}}{# endforeach -#}\n\nB"),
        "A12B"
    );
}

#[test]
fn flow_statements_accept_markers() {
    assert_eq!(
        out("{# foreach x in [1, 2, 3] #}{{x}}  {#- break -#}  {# endforeach #}"),
        "1"
    );
    assert_eq!(out("a  {#- return -#}  b"), "");
    assert_eq!(out("x{#- return \"v\" -#}y"), "v");
}

#[test]
fn trailing_marker_after_a_bare_condition() {
    assert_eq!(out("{# if true -#}\n  yes\n{# endif #}"), "yes\n");
}

#[test]
fn trailing_marker_after_a_subtraction() {
    assert_eq!(
        out("{# if {{g:hp}} - 12 == 0 -#}\n  zero\n{# endif #}"),
        "zero\n"
    );
    assert_eq!(out("{# if 3 - 1 -#}\ntruthy\n{# endif #}"), "truthy\n");
}

#[test]
fn unary_minus_in_a_condition_still_works() {
    assert_eq!(out("{# if -1 < 0 #}neg{# endif #}"), "neg");
    assert_eq!(out("{# if -1 < 0 -#}\nneg\n{# endif #}"), "neg\n");
}

#[test]
fn marker_and_subtraction_in_a_foreach_iterable() {
    assert_eq!(
        out("{# foreach x in [3 - 1] -#}\n{{x}}\n{# endforeach #}"),
        "2\n"
    );
}

#[test]
fn blank_lines_can_be_written_for_readability_and_removed() {
    let src = "\
Name: {{g:name}}

{#- if {{g:hp}} < 20 #}
Status: wounded
{#- endif #}
Done";
    assert_eq!(out(src), "Name: AliceStatus: wounded\nDone");
}

#[test]
fn where_a_marker_leaves_the_block_decides_the_rest() {
    assert_eq!(out("A\n{#- if true #}\nx\n{# endif #}\nB"), "Ax\n\nB");
    assert_eq!(out("\n{#- if true #}\nx\n{# endif #}\nB"), "x\nB");
}

#[test]
fn a_marker_is_all_or_nothing_across_newlines() {
    assert_eq!(out("A\n\n\n{{-g:name}}"), "AAlice");
    assert_eq!(out("A\n\n\n{{g:name}}"), "A\n\n\nAlice");
}

#[test]
fn markers_compose_with_standalone_line_consumption() {
    assert_eq!(out("A\n$[noop()]\nB"), "A\nB");
    assert_eq!(out("A\n$[-noop()-]\nB"), "AB");
}

#[test]
fn markers_on_a_standalone_block_tag() {
    assert_eq!(out("A\n{# if true #}\nx\n{# endif #}\nB"), "A\nx\nB");
    assert_eq!(out("A\n{#- if true #}\nx\n{# endif -#}\nB"), "Ax\nB");
}

#[test]
fn a_marker_suppresses_the_implicit_tag_line_rule() {
    assert_eq!(out("A\n{# if true #}\nx\n{# endif #}\nB"), "A\nx\nB");
    assert_eq!(out("A\n{#- if true #}\nx\n{# endif #}\nB"), "Ax\n\nB");
}

#[test]
fn an_all_whitespace_literal_can_be_fully_consumed() {
    assert_eq!(out("{{g:name}}   {{-g:name}}"), "AliceAlice");
}

#[test]
fn trimming_does_not_depend_on_what_rendered() {
    let src = "A   {{-g:empty-}}   B";
    assert_eq!(out(src), "AB");

    let mut c = ctx();
    c.set("g", "empty", Value::String("X".into()));
    assert_eq!(render(src, &mut c, &registry()).unwrap(), "AXB");
}

#[test]
fn markers_inside_expressions_parse_and_are_inert() {
    assert!(parse("{# if {{-g:hp}} > 1 #}x{# endif #}").is_ok());
    assert_eq!(out("{# if {{-g:hp-}} > 1 #}x{# endif #}"), "x");
    assert_eq!(out("$[val({{-g:name-}})]"), "Alice");
}

#[test]
fn a_nested_markers_does_not_leak_to_the_outer_tag() {
    assert_eq!(out("A   {# if {{-g:hp-}} > 1 #}x{# endif #}"), "A   x");
}
