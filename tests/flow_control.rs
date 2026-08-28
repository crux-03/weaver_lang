//! `{# break #}`, `{# continue #}`, and `{# return #}`.
//!
//! Flow statements share the control-flow delimiter with `if` and
//! `foreach` because that is what they are. `break` and `continue` are
//! consumed by the innermost loop; `return` propagates to the top of the
//! template and decides its value.

use weaver_lang::{
    ClosureCommand, CompiledTemplate, EvalOptions, Registry, SimpleContext, Value, parse, render,
    render_value, render_with_options,
};

fn registry() -> Registry {
    let mut reg = Registry::new();
    // Records each call so tests can assert on what actually ran, not
    // just on the rendered output.
    reg.register_command(ClosureCommand::new("mark", |args| {
        Ok(args.into_iter().next())
    }));
    reg
}

fn ctx() -> SimpleContext {
    let mut ctx = SimpleContext::new();
    ctx.set(
        "g",
        "items",
        Value::Array(vec![1i64.into(), 2i64.into(), 3i64.into()]),
    );
    ctx.set("g", "hp", 12i64);
    ctx
}

fn out(src: &str) -> String {
    render(src, &mut ctx(), &registry()).unwrap()
}

fn val(src: &str) -> Value {
    render_value(src, &mut ctx(), &registry()).unwrap()
}

// ── break ───────────────────────────────────────────────────────────────

#[test]
fn break_stops_the_loop() {
    assert_eq!(
        out("{# foreach x in [1, 2, 3] #}{{x}}{# break #}{# endforeach #}"),
        "1"
    );
}

#[test]
fn break_inside_if_inside_foreach() {
    assert_eq!(
        out(
            "{# foreach x in [1, 2, 3] #}{{x}}{# if {{x}} == 2 #}{# break #}{# endif #}{# endforeach #}"
        ),
        "12"
    );
}

#[test]
fn break_leaves_the_outer_loop_running() {
    // The inner break must not terminate the outer iteration.
    assert_eq!(
        out("{# foreach a in [1, 2] #}\
             [{# foreach b in [8, 9] #}{{b}}{# break #}{# endforeach #}]\
             {# endforeach #}"),
        "[8][8]"
    );
}

#[test]
fn break_skips_the_rest_of_the_body() {
    assert_eq!(
        out("{# foreach x in [1, 2] #}a{# break #}b{# endforeach #}"),
        "a"
    );
}

#[test]
fn continue_skips_to_the_next_iteration() {
    assert_eq!(
        out(
            "{# foreach x in [1, 2, 3] #}{# if {{x}} == 2 #}{# continue #}{# endif #}{{x}}{# endforeach #}"
        ),
        "13"
    );
}

#[test]
fn continue_does_not_terminate_the_loop() {
    assert_eq!(
        out("{# foreach x in [1, 2, 3] #}{# continue #}{{x}}{# endforeach #}"),
        ""
    );
}

#[test]
fn continue_in_inner_loop_is_contained() {
    assert_eq!(
        out("{# foreach a in [1, 2] #}\
             {{a}}:{# foreach b in [8, 9] #}{# if {{b}} == 8 #}{# continue #}{# endif #}{{b}}{# endforeach #};\
             {# endforeach #}"),
        "1:9;2:9;"
    );
}

#[test]
fn bare_return_is_sugar_for_returning_none() {
    assert_eq!(val("kept{# return #}discarded"), Value::None);
    assert_eq!(val("kept{# return none #}discarded"), Value::None);
}

#[test]
fn bare_return_renders_as_empty() {
    assert_eq!(out("kept{# return #}discarded"), "");
}

#[test]
fn the_two_bare_forms_are_indistinguishable() {
    for (bare, explicit) in [
        ("{# return #}", "{# return none #}"),
        ("text{# return #}", "text{# return none #}"),
        (
            "{# if true #}{# return #}{# endif #}",
            "{# if true #}{# return none #}{# endif #}",
        ),
    ] {
        assert_eq!(val(bare), val(explicit), "{bare:?} vs {explicit:?}");
        assert_eq!(out(bare), out(explicit), "{bare:?} vs {explicit:?}");
    }
}

#[test]
fn bare_return_works_as_a_guard_clause() {
    let src = "{# if {{g:hp}} > 50 #}{# return #}{# endif #}Wounded: {{g:hp}}";
    assert_eq!(out(src), "Wounded: 12");

    let mut c = ctx();
    c.set("g", "hp", 99i64);
    assert_eq!(render(src, &mut c, &registry()).unwrap(), "");
}

#[test]
fn return_exits_the_whole_template_from_inside_a_loop() {
    assert_eq!(
        val(
            "{# foreach x in [1, 2, 3] #}{{x}}{# if {{x}} == 2 #}{# return #}{# endif #}{# endforeach #}tail"
        ),
        Value::None
    );
}

#[test]
fn return_from_a_nested_loop_exits_both() {
    assert_eq!(
        val(
            "{# foreach a in [1, 2] #}{# foreach b in [8, 9] #}{{b}}{# return \"v\" #}{# endforeach #}{# endforeach #}tail"
        ),
        Value::String("v".into())
    );
}

#[test]
fn return_inside_if_branch() {
    assert_eq!(out("{# if true #}a{# return \"r\" #}{# endif #}b"), "r");
    assert_eq!(out("{# if false #}a{# return \"r\" #}{# endif #}b"), "b");
}

#[test]
fn valued_return_replaces_the_output() {
    assert_eq!(out("ignored{# return \"kept\" #}"), "kept");
}

#[test]
fn valued_return_preserves_the_type() {
    assert_eq!(
        val(r#"{# return ["sword", "cursed"] #}"#),
        Value::Array(vec!["sword".into(), "cursed".into()])
    );
    assert_eq!(val("{# return 42 #}"), Value::Number(42.0));
    assert_eq!(val("{# return true #}"), Value::Bool(true));
    assert_eq!(val("{# return none #}"), Value::None);
}

#[test]
fn valued_return_evaluates_expressions() {
    assert_eq!(val("{# return 1 + 2 #}"), Value::Number(3.0));
    assert_eq!(val("{# return {{g:hp}} < 20 #}"), Value::Bool(true));
}

#[test]
fn returned_array_renders_joined_in_string_form() {
    assert_eq!(out(r#"{# return ["a", "b"] #}"#), "a, b");
}

#[test]
fn value_without_any_return_is_the_rendered_output() {
    assert_eq!(val("HP: {{g:hp}}"), Value::String("HP: 12".into()));
}

#[test]
fn render_always_equals_value_to_output_string() {
    for src in [
        "plain text",
        "kept{# return #}dropped",
        "kept{# return none #}dropped",
        "kept{# stop #}dropped",
        r#"ignored{# return ["a", "b"] #}"#,
        "{# return 42 #}",
        "{# foreach x in [1, 2] #}{{x}}{# break #}{# endforeach #}",
    ] {
        assert_eq!(
            out(src),
            val(src).to_output_string(),
            "invariant broken for {src:?}"
        );
    }
}

#[test]
fn return_from_inside_a_loop_carries_its_value_out() {
    assert_eq!(
        val(
            "{# foreach x in [1, 2, 3] #}{# if {{x}} == 2 #}{# return {{x}} #}{# endif #}{# endforeach #}"
        ),
        Value::Number(2.0)
    );
}

#[test]
fn stop_keeps_the_output_rendered_so_far() {
    assert_eq!(out("kept{# stop #}discarded"), "kept");
    assert_eq!(val("kept{# stop #}discarded"), Value::String("kept".into()));
}

#[test]
fn stop_and_return_are_opposites() {
    assert_eq!(val("text{# stop #}"), Value::String("text".into()));
    assert_eq!(val("text{# return #}"), Value::None);
}

#[test]
fn stop_at_the_end_is_a_no_op() {
    assert_eq!(out("all of it{# stop #}"), "all of it");
    assert_eq!(out("all of it"), "all of it");
}

#[test]
fn stop_exits_the_whole_template_from_inside_a_loop() {
    assert_eq!(
        out(
            "{# foreach x in [1, 2, 3] #}{{x}}{# if {{x}} == 2 #}{# stop #}{# endif #}{# endforeach #}tail"
        ),
        "12"
    );
}

#[test]
fn stop_from_a_nested_loop_exits_both() {
    assert_eq!(
        out(
            "{# foreach a in [1, 2] #}{# foreach b in [8, 9] #}{{b}}{# stop #}{# endforeach #}{# endforeach #}tail"
        ),
        "8"
    );
}

#[test]
fn stop_truncates_the_tail_of_a_template() {
    let src = "\
Intro paragraph.
{# foreach x in [\"a\", \"b\"] #}
  - {{x}}
{# endforeach #}
{# stop #}
Draft notes, not for output.";
    assert_eq!(out(src), "Intro paragraph.\n  - a\n  - b\n");
}

#[test]
fn stop_is_legal_at_any_depth() {
    assert!(parse("{# stop #}").is_ok());
    assert!(parse("{# if true #}{# stop #}{# endif #}").is_ok());
    assert!(parse("{# foreach x in [1] #}{# stop #}{# endforeach #}").is_ok());
}

#[test]
fn stop_takes_no_expression() {
    assert!(parse("{# stop 1 #}").is_err());
    assert!(parse(r#"{# stop "x" #}"#).is_err());
}

#[test]
fn stop_accepts_trim_markers() {
    assert_eq!(out("a  {#- stop -#}  b"), "a");
}

#[test]
fn break_outside_a_loop_is_a_parse_error() {
    let errors = parse("{# break #}").unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("`break` outside of a loop"));
    assert!(errors[0].hint.as_ref().unwrap().contains("foreach"));
}

#[test]
fn continue_outside_a_loop_is_a_parse_error() {
    let errors = parse("{# continue #}").unwrap_err();
    assert!(errors[0].message.contains("`continue` outside of a loop"));
}

#[test]
fn break_inside_an_if_but_outside_a_loop_is_rejected() {
    assert!(parse("{# if true #}{# break #}{# endif #}").is_err());
    assert!(parse("{# if false #}x{# else #}{# break #}{# endif #}").is_err());
    assert!(parse("{# if false #}x{# elif true #}{# break #}{# endif #}").is_err());
}

#[test]
fn break_inside_a_loop_body_is_accepted() {
    assert!(parse("{# foreach x in [1] #}{# break #}{# endforeach #}").is_ok());
    assert!(
        parse("{# foreach x in [1] #}{# if true #}{# break #}{# endif #}{# endforeach #}").is_ok()
    );
}

#[test]
fn return_is_legal_at_any_depth() {
    assert!(parse("{# return #}").is_ok());
    assert!(parse("{# return 1 #}").is_ok());
    assert!(parse("{# if true #}{# return #}{# endif #}").is_ok());
    assert!(parse("{# foreach x in [1] #}{# return #}{# endforeach #}").is_ok());
}

#[test]
fn break_error_span_points_at_the_statement() {
    let src = "{# foreach x in [1] #}{# endforeach #}\n{# break #}";
    let errors = parse(src).unwrap_err();
    let span = errors[0].span;
    assert_eq!(&src[span.start..span.end], "{# break #}");
}

#[test]
fn longer_identifiers_do_not_match_flow_keywords() {
    for src in [
        "{# breakpoint #}",
        "{# continued #}",
        "{# returns 1 #}",
        "{# stopped #}",
    ] {
        assert!(parse(src).is_err(), "{src:?} should not parse");
    }
}

#[test]
fn lenient_mode_does_not_swallow_flow_signals() {
    let opts = EvalOptions::new().lenient(true);
    let result = render_with_options(
        "{# foreach x in [1, 2, 3] #}{{x}}{# break #}{# endforeach #}{{g:missing}}",
        &mut ctx(),
        &registry(),
        opts,
    )
    .unwrap();
    assert_eq!(result, "1{{g:missing}}");
}

#[test]
fn break_reduces_the_iterations_counted_against_the_cap() {
    let opts = EvalOptions::new().max_iterations(2);
    let result = render_with_options(
        "{# foreach x in [1, 2, 3, 4, 5] #}{{x}}{# break #}{# endforeach #}",
        &mut ctx(),
        &registry(),
        opts,
    );
    assert_eq!(result.unwrap(), "1");
}

#[test]
fn loop_bindings_do_not_leak_after_a_return() {
    let template =
        CompiledTemplate::compile("{# foreach x in [1, 2] #}{# return {{x}} #}{# endforeach #}")
            .unwrap();
    let mut c = ctx();
    let reg = registry();
    assert_eq!(
        template.evaluate_value(&mut c, &reg).unwrap(),
        Value::Number(1.0)
    );
    assert_eq!(
        template.evaluate_value(&mut c, &reg).unwrap(),
        Value::Number(1.0)
    );
}

#[test]
fn flow_state_is_reset_between_evaluations() {
    let template = CompiledTemplate::compile("a{# return \"r\" #}b").unwrap();
    let mut c = ctx();
    let reg = registry();
    assert_eq!(template.evaluate(&mut c, &reg).unwrap(), "r");
    assert_eq!(template.evaluate(&mut c, &reg).unwrap(), "r");
}

#[test]
fn commands_before_a_return_still_run() {
    let mut reg = Registry::new();
    reg.register_command(ClosureCommand::new("set_flag", |_| {
        Ok(Some(Value::String("ran".into())))
    }));
    let mut c = ctx();
    let result = render("$[set_flag()]{# return \"done\" #}", &mut c, &reg).unwrap();
    assert_eq!(result, "done");
}

#[test]
fn flow_statements_on_their_own_line_do_not_leave_blank_lines() {
    assert_eq!(
        out("{# foreach x in [1, 2, 3] #}\n{{x}}\n{# break #}\n{# endforeach #}"),
        "1\n"
    );
}
