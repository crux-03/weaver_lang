#[allow(unused_imports)]
use weaver_lang::{evaluate, EvalContext, EvalError, EvalErrorKind, Registry, SimpleContext, Value};
use weaver_macros::{weaver_command, weaver_processor};

// ── Basic processor: picks first item from an array ─────────────────────

#[weaver_processor(namespace = "test", name = "first")]
fn first_item(items: Vec<Value>) -> Result<Value, EvalError> {
    items.into_iter().next().ok_or_else(|| {
        EvalError::new(EvalErrorKind::HostError, "empty array")
    })
}

// ── String processor: wraps text in brackets ────────────────────────────

#[weaver_processor(namespace = "test", name = "bracket")]
fn bracket(text: String) -> Result<Value, EvalError> {
    Ok(Value::String(format!("[{text}]")))
}

// ── Numeric processor: doubles a number ─────────────────────────────────

#[weaver_processor(namespace = "test", name = "double")]
fn double_number(n: f64) -> Result<Value, EvalError> {
    Ok(Value::Number(n * 2.0))
}

// ── Bool processor: returns "yes" or "no" ───────────────────────────────

#[weaver_processor(namespace = "test", name = "yesno")]
fn yes_no(value: bool) -> Result<Value, EvalError> {
    Ok(Value::String(if value { "yes" } else { "no" }.to_string()))
}

// ── Multi-param processor ───────────────────────────────────────────────

#[weaver_processor(namespace = "test", name = "repeat")]
fn repeat_text(text: String, count: f64) -> Result<Value, EvalError> {
    Ok(Value::String(text.repeat(count.round() as usize)))
}

// ── Tests ───────────────────────────────────────────────────────────────

fn make_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_processor(FirstItemProcessor);
    registry.register_processor(BracketProcessor);
    registry.register_processor(DoubleNumberProcessor);
    registry.register_processor(YesNoProcessor);
    registry.register_processor(RepeatTextProcessor);
    registry
}

#[test]
fn test_macro_array_processor() {
    let template = weaver_lang::parse(
        r#"@[test.first(items: ["alpha", "beta", "gamma"])]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "alpha");
}

#[test]
fn test_macro_string_processor() {
    let template = weaver_lang::parse(
        r#"@[test.bracket(text: "hello")]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "[hello]");
}

#[test]
fn test_macro_number_processor() {
    let template = weaver_lang::parse(
        r#"@[test.double(n: 21)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "42");
}

#[test]
fn test_macro_bool_processor() {
    let template = weaver_lang::parse(
        r#"@[test.yesno(value: true)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "yes");
}

#[test]
fn test_macro_multi_param_processor() {
    let template = weaver_lang::parse(
        r#"@[test.repeat(text: "ab", count: 3)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "ababab");
}

#[test]
fn test_macro_type_validation_rejects_wrong_type() {
    // Pass a number where a string is expected
    let template = weaver_lang::parse(
        r#"@[test.bracket(text: 42)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind, EvalErrorKind::TypeError);
    assert!(err.message.contains("string"));
}

#[test]
fn test_macro_missing_required_property() {
    // Call repeat without the "count" property
    let template = weaver_lang::parse(
        r#"@[test.repeat(text: "ab")]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.message.contains("count"));
}

#[test]
fn test_macro_processor_with_variable_input() {
    let template = weaver_lang::parse(
        r#"@[test.bracket(text: {{global:name}})]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    ctx.set("global", "name", "world");
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "[world]");
}

#[test]
fn test_macro_processor_in_template_context() {
    // Use a macro-defined processor inside a larger template
    let template = weaver_lang::parse(
        r#"Result: @[test.double(n: 5)] and @[test.bracket(text: "done")]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "Result: 10 and [done]");
}

// ═══════════════════════════════════════════════════════════════════════
// Command macro tests
// ═══════════════════════════════════════════════════════════════════════

// ── Pure command: no ctx or registry access ──────────────────────────

#[weaver_command(name = "greet")]
fn greet(name: String) -> Result<Option<Value>, EvalError> {
    Ok(Some(Value::String(format!("Hello, {name}!"))))
}

// ── Command returning None (side-effect only, no ctx needed here) ───

#[weaver_command(name = "noop")]
fn noop() -> Result<Option<Value>, EvalError> {
    Ok(None)
}

// ── Command with ctx access: sets a variable ────────────────────────

#[weaver_command(name = "set_var")]
fn set_var(key: String, value: Value, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
    if let Some(pos) = key.find(':') {
        ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
    }
    Ok(None)
}

// ── Command with ctx access: reads and returns a variable ───────────

#[weaver_command(name = "get_var")]
fn get_var(key: String, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
    if let Some(pos) = key.find(':') {
        let scope = &key[..pos];
        let name = &key[pos + 1..];
        match ctx.resolve_variable(scope, name)? {
            Some(val) => Ok(Some(val)),
            None => Ok(Some(Value::None)),
        }
    } else {
        Ok(Some(Value::None))
    }
}

// ── Command with numeric arg ────────────────────────────────────────

#[weaver_command(name = "double")]
fn cmd_double(n: f64) -> Result<Option<Value>, EvalError> {
    Ok(Some(Value::Number(n * 2.0)))
}

// ── Command with bool arg ───────────────────────────────────────────

#[weaver_command(name = "yesno")]
fn cmd_yesno(flag: bool) -> Result<Option<Value>, EvalError> {
    Ok(Some(Value::String(
        if flag { "yes" } else { "no" }.to_string(),
    )))
}

// ── Command with array arg ──────────────────────────────────────────

#[weaver_command(name = "first")]
fn cmd_first(items: Vec<Value>) -> Result<Option<Value>, EvalError> {
    Ok(items.into_iter().next())
}

// ── Tests ───────────────────────────────────────────────────────────

fn make_cmd_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register_command(GreetCommand);
    registry.register_command(NoopCommand);
    registry.register_command(SetVarCommand);
    registry.register_command(GetVarCommand);
    registry.register_command(CmdDoubleCommand);
    registry.register_command(CmdYesnoCommand);
    registry.register_command(CmdFirstCommand);
    registry
}

#[test]
fn test_cmd_macro_pure_command() {
    let template = weaver_lang::parse(
        r#"$[greet("world")]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "Hello, world!");
}

#[test]
fn test_cmd_macro_returns_none() {
    let template = weaver_lang::parse(
        r#"before$[noop()]after"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "beforeafter");
}

#[test]
fn test_cmd_macro_with_ctx_sets_variable() {
    let src = "Name: {{global:name}}\n$[set_var(\"global:name\", \"Alice\")]\nNew name: {{global:name}}";
    let template = weaver_lang::parse(src).unwrap();
    let mut ctx = SimpleContext::new();
    ctx.set("global", "name", "Sarah");
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "Name: Sarah\nNew name: Alice");
}

#[test]
fn test_cmd_macro_with_ctx_reads_variable() {
    let template = weaver_lang::parse(
        r#"$[get_var("global:score")]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    ctx.set("global", "score", 42i64);
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "42");
}

#[test]
fn test_cmd_macro_number_arg() {
    let template = weaver_lang::parse(
        r#"$[double(21)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "42");
}

#[test]
fn test_cmd_macro_bool_arg() {
    let template = weaver_lang::parse(
        r#"$[yesno(true)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "yes");
}

#[test]
fn test_cmd_macro_array_arg() {
    let template = weaver_lang::parse(
        r#"$[first(["alpha", "beta"])]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "alpha");
}

#[test]
fn test_cmd_macro_type_validation_rejects_wrong_type() {
    // Pass a number where a string is expected
    let template = weaver_lang::parse(
        r#"$[greet(42)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind, EvalErrorKind::TypeError);
    assert!(err.message.contains("string"));
}

#[test]
fn test_cmd_macro_missing_required_arg() {
    // Call greet without any arguments
    let template = weaver_lang::parse(
        r#"$[greet()]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.message.contains("name"));
}

#[test]
fn test_cmd_macro_standalone_line_eating() {
    // set_var on its own line should be consumed cleanly
    let src = "Line 1\n$[set_var(\"global:x\", \"val\")]\nLine 3";
    let template = weaver_lang::parse(src).unwrap();
    let mut ctx = SimpleContext::new();
    let registry = make_cmd_registry();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "Line 1\nLine 3");
}

#[test]
fn test_cmd_macro_in_template_with_processor() {
    // Mix macro-defined commands and processors in the same template
    let mut registry = make_cmd_registry();
    registry.register_processor(BracketProcessor);

    let template = weaver_lang::parse(
        r#"@[test.bracket(text: "hi")] $[double(5)]"#,
    )
    .unwrap();
    let mut ctx = SimpleContext::new();
    let result = evaluate(&template, &mut ctx, &registry).unwrap();
    assert_eq!(result, "[hi] 10");
}
