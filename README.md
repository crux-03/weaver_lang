# weaver-lang

A template language for procedural content generation in Rust. Parse text with embedded expressions, control flow, and host-defined callables, then evaluate it into a final string.

```
Hello, {{global:name}}! You have {{global:hp}} HP.
{# if {{global:hp}} < 20 #}
You're badly wounded.
{# elif {{global:hp}} < 50 #}
You've seen better days.
{# else #}
You're in fighting shape.
{# endif #}
```

weaver-lang separates the **language** (parsing and evaluation) from the **host** (state, triggers, documents) through a trait boundary. The library has no opinion about where your data lives — you provide it through an `EvalContext` implementation.

## Quick start

```rust
use weaver_lang::{render, SimpleContext, Registry};

let mut ctx = SimpleContext::new();
ctx.set("global", "name", "Alice");

let registry = Registry::new();
let output = render("Hello, {{global:name}}!", &mut ctx, &registry).unwrap();
assert_eq!(output, "Hello, Alice!");
```

## Compiled templates

Parse once, evaluate many times:

```rust
use weaver_lang::{CompiledTemplate, SimpleContext, Registry};

let template = CompiledTemplate::compile("HP: {{global:hp}}").unwrap();
let registry = Registry::new();
let mut ctx = SimpleContext::new();

ctx.set("global", "hp", 100i64);
assert_eq!(template.evaluate(&mut ctx, &registry).unwrap(), "HP: 100");

ctx.set("global", "hp", 42i64);
assert_eq!(template.evaluate(&mut ctx, &registry).unwrap(), "HP: 42");
```

## Syntax reference

### Variables

```
{{scope:name}}
```

Look up a variable by scope and name. `global` and `local` are conventional, but hosts can define any scope.

### Processors — `@[namespace.name(key: value)]`

Pure computations with named properties. No access to evaluation state.

```
@[math.add(a: 1, b: 2)]
@[core.weaver.rng(min: 1, max: 100)]
```

### Commands — `$[name(arg1, arg2)]`

Stateful operations with positional arguments. Can read/write variables through the evaluation context.

```
$[set_var("global:name", "Alice")]
$[greet("world")]
```

When a command appears alone on a line, the entire line is consumed — no blank line is left in the output:

```
Line before
$[set_var("global:x", "val")]
Line after
```

Evaluates to `Line before\nLine after`.

### Triggers and documents

```
<trigger id="dark_forest">     // Activate another entry, splice its output
[[LORE_INTRO]]                 // Import a reusable content block
```

Both are expressions — they can appear in arrays, processor arguments, conditions, etc.

### Control flow

```
{# if {{global:hp}} < 20 #}
  Critical condition!
{# elif {{global:hp}} < 50 #}
  Wounded.
{# else #}
  Healthy.
{# endif #}

{# foreach item in ["sword", "shield", "potion"] #}
  - {{local:item}}
{# endforeach #}
```

Control flow tags on their own lines don't produce blank lines in the output.

### Expressions and operators

Expressions appear in conditions, arguments, and inline. Types are preserved internally (string, number, bool, array, none) and coerced to strings only at the template output level.

**Comparison:** `==`, `!=`, `<`, `>`, `<=`, `>=`
**Logical:** `&&`, `||`, `!`
**Arithmetic:** `+`, `-`, `*`, `/`

`+` concatenates when either operand is a string. Division by zero returns an error. Operator precedence is flat left-to-right — use parentheses for clarity: `(a + b) * c`.

**Truthiness:** empty string, `0`, `false`, empty array, and `none` are falsy. Everything else is truthy.

## Registering processors and commands

### Closures

```rust
use weaver_lang::{Registry, ClosureCommand, ClosureProcessor, Value};

let mut registry = Registry::new();

registry.register_processor(ClosureProcessor::new("math", "add", |props| {
    let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
    let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
    Ok(Value::Number(a + b))
}));

registry.register_command(ClosureCommand::new("echo", |args| {
    Ok(args.into_iter().next())
}));
```

### Proc macros

The `weaver-macros` crate generates trait implementations from function signatures with automatic type validation:

```rust
use weaver_lang::{Value, EvalError};
use weaver_macros::weaver_processor;

#[weaver_processor(namespace = "text", name = "repeat")]
fn repeat_text(text: String, count: f64) -> Result<Value, EvalError> {
    Ok(Value::String(text.repeat(count as usize)))
}

// Generates `RepeatTextProcessor` struct implementing `WeaverProcessor`.
// registry.register_processor(RepeatTextProcessor);
```

Commands can opt into context access by naming a parameter `ctx`:

```rust
use weaver_lang::{Value, EvalError, EvalContext};
use weaver_macros::weaver_command;

#[weaver_command(name = "set_var")]
fn set_var(key: String, value: Value, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
    if let Some(pos) = key.find(':') {
        ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
    }
    Ok(None)
}

// Generates `SetVarCommand` struct implementing `WeaverCommand`.
```

Supported parameter types: `Value` (any), `String`, `f64`, `bool`, `Vec<Value>`.

### Trait implementations

For full control, implement `WeaverCommand` or `WeaverProcessor` directly:

```rust
use weaver_lang::{Value, EvalError, EvalContext, Registry};
use weaver_lang::registry::{WeaverCommand, CommandSignature};

struct MyCommand;

impl WeaverCommand for MyCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        // Full access to args, context, and registry
        Ok(None)
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "my_command".to_string(),
            params: Vec::new(),
        }
    }
}
```

## Implementing EvalContext

`SimpleContext` works for testing. For production, implement the `EvalContext` trait to connect weaver-lang to your application's state:

```rust
use weaver_lang::{EvalContext, EvalError, Value, Registry};

struct GameContext { /* your state */ }

impl EvalContext for GameContext {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        // Look up variables from your storage.
        // Return Ok(None) for undefined variables.
        todo!()
    }

    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        // Persist variable changes
        todo!()
    }

    fn fire_trigger(&mut self, entry_id: &str, registry: &Registry) -> Result<String, EvalError> {
        // Look up the target entry, evaluate it, return the output.
        // You are responsible for cycle detection and depth limiting.
        todo!()
    }

    fn resolve_document(&mut self, document_id: &str, registry: &Registry) -> Result<String, EvalError> {
        // Return document content (raw or pre-evaluated)
        todo!()
    }
}
```

The evaluator manages temporary scopes internally (foreach bindings). Only named scope operations like `"global"` and `"local"` reach the host.

## Evaluation options

Configure resource limits, cancellation, and lenient mode:

```rust
use weaver_lang::{render_with_options, EvalOptions, SimpleContext, Registry};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

let mut ctx = SimpleContext::new();
let registry = Registry::new();

let cancel = Arc::new(AtomicBool::new(false));

let opts = EvalOptions::new()
    .max_node_evaluations(10_000)   // cap AST node evaluations
    .max_iterations(1_000)          // cap total loop iterations
    .cancellation_token(cancel)     // abort from another thread
    .lenient(true);                 // undefined vars render as raw syntax

let result = render_with_options(
    "Hello, {{global:missing}}!",
    &mut ctx,
    &registry,
    opts,
).unwrap();
assert_eq!(result, "Hello, {{global:missing}}!");
```

## Error reporting

Parse errors carry source spans. Use `format_with_source` for diagnostics:

```
Error: undefined variable: global:player_name
 --> Dark Forest:12:6
  |
 12 |  {# if {{global:player_name}} #}
    |        ^^^^^^^^^^^^^^^^^^^^^^^
  = hint: did you mean to define this variable first?
```

Eval errors support error chaining for host-originated failures:

```rust
use weaver_lang::EvalError;

let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
let err = EvalError::host_error("failed to load entry").with_source(io_err);
// The full error chain is preserved via std::error::Error::source()
```

## Known limitations

- Operator precedence is flat left-to-right. Use parentheses: `(a + b) * c`.
- All numbers are `f64`. Large integers above 2^53 lose precision.
- No assignment syntax in the language. Variable mutation goes through commands.
- Document evaluation depends on the host's `resolve_document` implementation.

## Dependencies

- [pest](https://pest.rs/) — PEG parser generator
- [thiserror](https://github.com/dtolnay/thiserror) — error derive macros
- [syn](https://github.com/dtolnay/syn), [quote](https://github.com/dtolnay/quote), [proc-macro2](https://github.com/alexcrichton/proc-macro2) — proc macro infrastructure (weaver-macros only)

## License

MIT
