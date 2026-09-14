//! What WTN does and does not change about text mode.
//!
//! WTN is an entry rule over a shared value grammar, which raises a
//! fair question: how much of it leaks into an ordinary template? These
//! tests answer it, and they are deliberately *not* gated on the `data`
//! feature — the guarantee is that text mode behaves identically whether
//! the feature is on or off, so the test has to run in both builds.

use weaver_lang::{Registry, SimpleContext, parse, render};

fn eval(source: &str) -> String {
    let mut ctx = SimpleContext::new();
    render(source, &mut ctx, &Registry::new()).unwrap()
}

/// Prose is prose. Nothing WTN added is a construct starter, so
/// `literal_text` ends at exactly the six delimiters it always did.
#[test]
fn text_mode_prose_is_untouched() {
    for prose in [
        "// this is not a comment, it is prose",
        "a URL: https://example.com/path",
        r#"r"this is not a raw string""#,
        "Custom(1) is just text",
        "null and 1e5 and true",
        r#"an object? {name: "Alice"} in prose"#,
        r"C:\path\to\file and a \u0041 escape",
    ] {
        assert_eq!(eval(prose), prose, "prose should render verbatim");
    }
}

/// The input block is reachable only from the WTN entry rule. In a
/// template it is text — not a declaration, and not an error either.
#[test]
fn an_inputs_block_in_a_template_is_just_text() {
    let source = concat!(
        "#inputs\n",
        "characters: [Ref<Character>]\n",
        "difficulty: enum(\"easy\", \"brutal\") = \"normal\"\n",
        "\n",
        "Hello, {{global:name}}!"
    );

    let mut ctx = SimpleContext::new();
    ctx.set("global", "name", "Alice");
    let out = render(source, &mut ctx, &Registry::new()).unwrap();

    assert!(out.starts_with("#inputs\ncharacters: [Ref<Character>]"));
    assert!(out.ends_with("Hello, Alice!"));
    // The declarations rendered verbatim rather than being consumed.
    assert!(out.contains(r#"difficulty: enum("easy", "brutal") = "normal""#));
}

/// The shared half really is shared: what WTN added to expressions is
/// available inside a template's constructs too. This is the deliberate
/// half of the boundary, and it is additive — each of these was a parse
/// error before.
#[test]
fn the_expression_additions_are_shared_on_purpose() {
    assert_eq!(eval("{{ Custom(1) }}"), r#"{"Custom":1}"#);
    assert_eq!(eval(r#"{{ r"a {{b}} c" }}"#), "a {{b}} c");
    assert_eq!(eval("{# if true // note\n #}yes{# endif #}"), "yes");
    assert_eq!(eval("{{ 1e3 }}"), "1000");
    assert_eq!(
        eval("{{ [{# foreach n in [1, 2] #} n * 10 {# endforeach #}] }}"),
        "10, 20"
    );
}

/// A string inside a construct stays a string in text mode. Templating
/// strings is a property of the WTN entry point, not of the grammar.
#[test]
fn a_string_in_a_template_is_never_a_template() {
    assert_eq!(eval(r#"{{ "literal {{braces}}" }}"#), "literal {{braces}}");
    assert_eq!(eval(r#"{{ "see [[DOC]]" }}"#), "see [[DOC]]");
    assert_eq!(eval(r#"{{ "run $[cmd()]" }}"#), "run $[cmd()]");
}

/// `null` joins `true`, `false` and `none` as a word that cannot be read as
/// a reference. It is the one place the JSON compatibility work reaches
/// into text mode, so it is worth stating out loud.
#[test]
fn null_is_a_reserved_word_in_expression_position() {
    // Binding it is still allowed, but reading it gives the literal.
    assert_eq!(
        eval("{# foreach null in [1] #}{{null}}{# endforeach #}"),
        ""
    );
    // Names that merely start with a keyword are unaffected.
    assert_eq!(
        eval(r#"{# foreach nullable in ["x"] #}{{nullable}}{# endforeach #}"#),
        "x"
    );
    assert!(parse("{{null}}").is_ok());
}
