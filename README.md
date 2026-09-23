# weaver-lang

A template language for procedural content generation, embeddable in Rust.

A Weaver template is text with holes in it: expressions, control flow, and
calls into functions you register. Evaluating one produces a string, or a
structured value when the template asks for one.

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

The library has no opinion about where your data lives. You implement one
trait, `EvalContext`, to answer questions like "what is `global:hp`?", and you
fill a `Registry` with the functions templates are allowed to call. Parsing,
evaluation, resource limits and error reporting are the library's side of the
line.

## Installation
Install with `cargo add weaver_lang` or:
```toml
[dependencies]
weaver_lang = "0.8"
```

| feature  | default | what it adds |
|----------|---------|--------------|
| `macros` | yes     | `#[weaver_processor]` and `#[weaver_command]` attribute macros |
| `wtn`    | no      | [Weaver Template Notation](#weaver-template-notation-wtn): documents that are one value rather than prose |
| `serde`  | no      | `Serialize`/`Deserialize` on signature metadata, for shipping the registry's vocabulary to an editor frontend |

## Quick start

```rust
use weaver_lang::{render, SimpleContext, Registry};

let mut ctx = SimpleContext::new();
ctx.set("global", "name", "Alice");

let registry = Registry::new();
let output = render("Hello, {{global:name}}!", &mut ctx, &registry).unwrap();
assert_eq!(output, "Hello, Alice!");
```

`SimpleContext` is an in-memory context meant for tests and small programs;
a real host implements [`EvalContext`](#evalcontext) itself.

To parse once and evaluate many times, compile the template:

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

`CompiledExpr` is the same idea for a bare expression, which is useful for
things like activation conditions that are evaluated often and produce a
value rather than text:

```rust
use weaver_lang::{CompiledExpr, SimpleContext, Registry, Value};

let cond = CompiledExpr::compile("global:hp > 50").unwrap();
let registry = Registry::new();
let mut ctx = SimpleContext::new();

ctx.set("global", "hp", 75i64);
assert_eq!(cond.evaluate(&mut ctx, &registry).unwrap(), Value::Bool(true));
```

## Syntax at a glance

| type        | syntax                                                             |
|-------------|--------------------------------------------------------------------|
| variables   | `{{scope:name}}`, `{{scope:name.path.into.object}}`                |
| expressions | `{{ 1 + 2 }}`, `{{ global:gold - 10 }}`                            |
| literals    | `{name: "Alice", hp: 10}`, `["sword", "shield"]`                   |
| indexing    | `{{items[0]}}`, `{{obj["key"]}}`, `{{party[i].name}}`              |
| processors  | `@[namespace.name(foo: value1, bar: value2)]`                      |
| commands    | `$[name(foo, bar)]`, `$[namespace.name(foo, bar)]`                  |
| triggers    | `<trigger id="some_id">`                                           |
| documents   | `[[some_id]]`                                                      |
| if/else     | `{# if foo == bar #} baz {# endif #}`                              |
| foreach     | `{# foreach foo in bar #} - {{foo}} {# endforeach #}`              |
| flow        | `{# break #}`, `{# continue #}`, `{# return expr #}`, `{# stop #}` |
| trim        | `{#- ... -#}`, `{{- ... -}}`, `$[- ... -]`                         |
| comments    | `// to end of line`                                                |

## Values and expressions

Values are strings, numbers, booleans, arrays, objects, and `none`. Types are
preserved throughout evaluation and coerced to text only when a value reaches
template output.

`{{ }}` holds any expression, not just a name, so simple arithmetic needs no
processor:

```
{{ 1 + 2 }}
{{ char:alice.stats.hp * 2 }}
{{ {name: "Alice", hp: 10} }}
```

**Operators.** Comparison `==` `!=` `<` `>` `<=` `>=`; logical `&&` `||` `!`;
arithmetic `+` `-` `*` `/` `%`. `+` concatenates when either side is a string.
Division by zero is an error.

**Precedence**, highest to lowest: indexing (`[]`, `.`), unary (`!`, `-`),
arithmetic (`*`, `/`, `+`, `-`), comparison, logical. Parentheses override it.

**Truthiness.** Empty string, `0`, `false`, empty array, empty object and
`none` are falsy; everything else is truthy.

**Rendering.** Strings render as themselves, numbers in their shortest form,
`none` as the empty string. A top-level array joins its elements with `, `.
An object renders as compact JSON with keys sorted, so output is deterministic
across runs. That includes an array reached *inside* an object, which stays
JSON rather than joining.

**Line breaks.** An expression continues across lines after an operator, but a
binary operator must start on the same line as its left operand:

```
@[p(x: 1 +
       2)]      one expression
@[p(x: 1
     + 2)]      two items, not one expression
```

That rule is what lets a bare newline separate list elements.

**Comments.** `//` runs to end of line, anywhere whitespace is allowed between
tokens. There is no block comment form.

**Raw strings.** `r"..."` is a string with no escape sequences, and, in
[WTN](#weaver-template-notation-wtn), no template parsing. Use `r#"..."#` when
the text itself contains a quote.

## Variables

```
{{scope:name}}
{{char:alice.stats.hp}}
{{npc.name}}
```

A reference is a scope, a name, and an optional dotted path. The scope and
name are handed to the host to resolve; every path segment after the name is
indexed into whatever the host returned.

```
{{char:alice.stats.hp}}
   │     │     └── path: indexed by the language
   │     └──────── name: resolved by the host
   └────────────── scope
```

`global` and `local` are conventions, not built-ins. A host can define any
scopes it likes.

A reference is an ordinary expression, so the `{{ }}` delimiters are only
needed where the surrounding text is prose. Inside a construct, write the
reference on its own:

```
{# if global:hp > 5 #}...{# endif #}
{# foreach c in global:party #}...{# endforeach #}
$[set_var("local:n", n + 1)]
```

A **bare** reference such as `{{n}}` resolves against loop bindings only.
Outside a loop it is an error, even if the host holds a `local:n`. Host state
is reachable only by naming its scope, so a template never silently reads
state it did not ask for.

A path that runs off the end of an object is reported exactly like a variable
that does not exist: an `UndefinedVariable` error naming the full path, or,
in [lenient mode](#evaluation-options), the reference passed through
unevaluated. Indexing into something that *isn't* an object is a `TypeError`
instead, because it can never be satisfied.

### Indexing

A subscript indexes an array by position or an object by key. `.name` and
`["name"]` mean the same thing and mix freely:

```
{{char:alice.gear[0]}}            first item
{{char:party[0].stats.hp}}        subscript, then a dotted path
{{char:alice["stats"]["hp"]}}     same as {{char:alice.stats.hp}}
{{char:alice.gear[global:slot]}}  computed subscript
{{ ["a", "b"][1] }}               a literal is indexable too
```

Three rules govern subscripts:

- **Out of range is absent.** `{{items[9]}}` on a three-element array behaves
  like a missing object key: an error, or a pass-through in lenient mode.
  There is no silent `none`.
- **Negative does not wrap.** `items[-1]` is a `TypeError` rather than the
  last element, and so is a fractional index. An off-by-one stays loud.
- **Nothing is coerced.** A number indexes an array and a string indexes an
  object; `items["0"]` and `obj[0]` are type errors, not guesses.

A subscript must be **glued** to the value it indexes, with no space before
`[`.
That is what keeps a newline usable as a separator: in

```
[a
 [1]]
```

the second line is an element, not a subscript of the first.

## Arrays and objects

Arrays are ordered lists; objects are string-keyed maps. Both can come from
the host or be written as literals:

```
["sword", "shield", "potion"]
{name: "Alice", hp: 10}
{party: [{name: "Alice"}, {name: "Bob"}]}
```

Object keys are identifiers or quoted strings (`{"needs quoting": 1}`), so
JSON parses as written. Values are arbitrary expressions. A duplicate key is a
parse error rather than a silent discard.

Inside arrays, objects, argument lists and property lists, items are separated
by a comma, a newline, or both, and a trailing separator is allowed:

```
{
  name: "Rags to Riches"
  agents: [
    {name: "Alice", role: "thief"}
    {name: "Bob", role: "fence"}
  ]
}
```

`{# foreach #}` and `{# if #}` may stand where an element or an entry would:

```
[{# foreach n in [1, 2, 3] #} n * 10 {# endforeach #}]   → [10, 20, 30]
```

An array item is an element and an object item is an entry; putting one where
the other belongs is a parse error rather than a surprise during evaluation.

Two things objects deliberately do not do: they are not iterable (`foreach`
requires an array), and they do not compare equal with `==`, the same as
arrays.

Building one from Rust:

```rust
use weaver_lang::Value;

let alice = Value::object([
    ("name", Value::String("Alice".into())),
    ("stats", Value::object([("hp", 10i64)])),
]);
assert_eq!(alice.to_json(), r#"{"name":"Alice","stats":{"hp":10}}"#);
```

There is no assignment syntax. A literal builds a value; binding it to a name
goes through a command the host defines, such as `set_var`. `Value::set_path`
is provided so such a command can fold a dotted path into nested objects the
same way reads unfold one.

## Control flow

```
{# if {{global:hp}} < 20 #}
  Critical condition!
{# elif {{global:hp}} < 50 #}
  Wounded.
{# else #}
  Healthy.
{# endif #}

{# foreach item in ["sword", "shield", "potion"] #}
  - {{item}}
{# endforeach #}
```

`foreach` binds each element of an array to a name. The binding is a bare
reference, and a dotted path works on it (`{{npc.name}}`), which is what makes
iterating an array of objects useful.

### Flow statements

```
{# break #}       stop the innermost foreach
{# continue #}    skip to the next iteration
{# return expr #} stop evaluating this template; expr is its value
{# return #}      sugar for {# return none #}
{# stop #}        stop evaluating; keep the output rendered so far
```

`return` and `stop` are legal anywhere. `break` and `continue` are rejected at
parse time outside a `foreach`. An `if` does not introduce a loop, so
`{# if ... #}{# break #}{# endif #}` at the top level is an error while the
same thing inside a loop body is fine.

Flow never crosses an entry boundary. Triggers and documents re-enter through
the host with a fresh evaluator, so a `break` inside an included document
cannot terminate the including template's loop.

### Blank lines

A construct on a line of its own does not leave an empty line behind:

```
Line before
$[set_var("global:x", "val")]
Line after
```

renders as `Line before\nLine after`.

The rule reads *rendered output*, not source. A construct counts as occupying
its own line when it starts a line in the output and is followed by a newline;
what happens then depends on what it rendered:

- nothing, or only whitespace: the line is consumed entirely
- text ending in a newline: that supplies the terminator, so the line's own
  newline is dropped rather than doubled
- text not ending in a newline: the line's newline is still needed and kept

A standalone command always consumes its line, since its return value is
discarded in template position.

Because the rule reads output, the two ways of writing a block agree. Both of
these render `"yes\n"`:

```
{# if true #}yes{# endif #}

{# if true #}
yes
{# endif #}
```

### Whitespace trim markers

A `-` immediately inside a delimiter removes all whitespace on that side,
newlines included. It works on every construct that produces output:

```
{#- if {{global:hp}} < 20 -#}
{{- global:name -}}
$[- set_var("global:x", "v") -]
@[- math.add(a: 1, b: 2) -]
```

```rust
use weaver_lang::{render, SimpleContext, Registry};

let mut ctx = SimpleContext::new();
ctx.set("global", "name", "Alice");
let registry = Registry::new();

assert_eq!(
    render("A   {{global:name}}   B", &mut ctx, &registry).unwrap(),
    "A   Alice   B",
);
assert_eq!(
    render("A   {{-global:name-}}   B", &mut ctx, &registry).unwrap(),
    "AAliceB",
);
```

A marker is all or nothing: it cannot collapse a run of blank lines down to
exactly one newline. You choose between all the whitespace and none of it.

There is no ambiguity with subtraction or unary minus. `{# if x - 1 -#}`
parses as `x - 1` followed by a marker, and `{# if -1 < 0 #}` is unaffected.

## Return values

A template's result is a `Value`. Its rendered text is that value passed
through `to_output_string()`, so `evaluate` is a thin wrapper over
`evaluate_value`. A template that never returns yields a `Value::String`
holding everything it rendered.

`{# return expr #}` makes `expr` the result and discards whatever was rendered
before it, which is how a template guards itself:

```
{# if {{global:hp}} > 50 #}{# return #}{# endif #}
Wounded: {{global:hp}} HP
```

An entry written like that renders nothing at all when the condition holds.

`{# stop #}` is the other half: it produces no value of its own and **keeps**
the text rendered so far, so it truncates rather than replaces.

```
Intro paragraph.
{# foreach x in {{global:items}} #}
  - {{x}}
{# endforeach #}
{# stop #}

Draft notes that never reach the output.
```

| terminal state      | `evaluate_value`  | `evaluate`                |
|---------------------|-------------------|---------------------------|
| ran to the end      | `String(output)`  | the output                |
| `{# stop #}`        | `String(output)`  | output up to the stop     |
| `{# return #}`      | `None`            | `""`                      |
| `{# return expr #}` | `expr`            | `expr.to_output_string()` |

```rust
use weaver_lang::{render, render_value, SimpleContext, Registry, Value};

let mut ctx = SimpleContext::new();
let registry = Registry::new();

let src = r#"ignored{# return ["sword", "cursed"] #}"#;
assert_eq!(
    render_value(src, &mut ctx, &registry).unwrap(),
    Value::Array(vec!["sword".into(), "cursed".into()]),
);
// The string form is the same value, joined.
assert_eq!(render(src, &mut ctx, &registry).unwrap(), "sword, cursed");
```

A template that needs to emit prose *and* hand the host a payload should stash
the payload through the context with something like
`$[set_var("local:tags", ...)]`, and use no `return`, so the prose stays the
result. `{# return #}` is not the way to do
it: it returns `none` and throws the prose away.

To let one entry return a value to *another*, override
`EvalContext::resolve_document_value`; see
[Documents that return values](#documents-that-return-values).

## Processors and commands

Templates call two kinds of host-defined function.

**Processors**, written `@[namespace.name(...)]`, are pure computations with
no access to evaluation state. They return a value.

```
@[math.add(a: 1, b: 2)]
@[text.repeat("ab", count: 3)]
```

**Commands**, written `$[name(...)]`, are stateful operations that can read
and write variables through the evaluation context. They may return a value,
but in template position the value is discarded.

```
$[set_var("global:name", "Alice")]
$[greet("world")]
```

Nothing is registered by default; namespaces like `math` and `text` above are
whatever you choose to provide.

### Arguments

Arguments may be written named or positional on either kind of callable:

```
@[math.add(a: 1, b: 2)]      @[math.add(1, 2)]
$[greet(name: "world")]      $[greet("world")]
```

Named arguments may appear in any order, and the two forms mix as long as
positional ones come first:

```
@[text.repeat("ab", count: 3)]
$[join(1, c: 3, b: 2)]
```

Under the hood a processor receives named properties and a command receives a
positional list; matching one form to the other goes through the callable's
**declared signature**. Anything registered with the `#[weaver_processor]` /
`#[weaver_command]` macros declares its parameters automatically and supports
both forms. A closure registered without `.property()` or `.param()` declares
nothing to match against, so it keeps working in its own form and reports
clearly if you use the other.

Giving a name twice, naming a slot a positional argument already filled, or
writing a positional argument after a named one is an error.

**A leading `name:` is always an argument marker.** At the top level of an
argument list, `$[cmd(local:x)]` names an argument `local`. It does not pass
the reference `local:x`. Interpolate or parenthesise to pass one:

```
$[cmd({{local:x}})]      reference
$[cmd((local:x))]        reference
@[p(v: [local:a, b:c])]  nested, so both are references
```

The rule reaches only that top level; inside any nested literal a colon is an
ordinary scoped reference.

## Triggers and documents

```
<trigger id="dark_forest">     // mark another entry for activation
[[LORE_INTRO]]                 // splice in another entry's output

<trigger id=({{user:name}} + "_inventory")>   // ids can be expressions
```

Both are expressions, so they can appear in arrays, arguments, conditions and
anywhere else a value is allowed. They differ in intent: a trigger marks
another entry for activation and is not meant to produce a value, while a
document is replaced by the output of the entry it names.

Neither does anything on its own. Looking the target up, evaluating it, and
detecting cycles are all your `EvalContext` implementation's job.

## EvalContext

`EvalContext` is how the language asks the host for anything it does not own.
`SimpleContext` covers tests; a real host implements the trait:

```rust
use weaver_lang::{EvalContext, EvalError, Value, Registry};

struct GameContext { /* your state */ }

impl EvalContext for GameContext {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        // Look up a variable in your storage. `name` is always a single
        // segment; dotted paths are indexed into the value you return.
        // Ok(None) means undefined.
        todo!()
    }

    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        // Persist a variable change.
        todo!()
    }

    fn fire_trigger(&mut self, entry_id: &str, registry: &Registry) -> Result<String, EvalError> {
        // Look up the target entry, evaluate it, return its output.
        // Cycle detection and depth limits are yours to enforce.
        todo!()
    }

    fn resolve_document(&mut self, document_id: &str, registry: &Registry) -> Result<String, EvalError> {
        // Return the document's content, raw or pre-evaluated.
        todo!()
    }
}
```

Loop bindings and other temporary scopes are managed inside the evaluator;
only named scopes such as `"global"` and `"local"` ever reach the host.

### Documents that return values

`resolve_document_value` has a default implementation that wraps
`resolve_document` in a `Value::String`. Override it when your entries use
`{# return #}`, and a document becomes a value-producing unit other entries
can consume:

```rust,ignore
fn resolve_document_value(
    &mut self,
    document_id: &str,
    registry: &Registry,
) -> Result<Value, EvalError> {
    let source = self.source_of(document_id);
    let template = parse(&source).map_err(|_| EvalError::host_error("parse failed"))?;
    evaluate_value(&template, self, registry)
}
```

```
// LOOT_TABLE:  {# return ["sword", "shield", "potion"] #}

{# foreach item in [[LOOT_TABLE]] #}
  - {{item}}
{# endforeach #}
```

There is no trigger counterpart, since a trigger marks an entry for activation
rather than producing content.

### Pushing paths down into storage

`resolve_variable_path` has a default implementation that resolves the root
through `resolve_variable` and walks the path with `Value::get_path`. That is
correct for every host, but it materializes the whole root value (an entire
character sheet, say) to read one field. Override it when your storage can
index the path directly:

```rust,ignore
fn resolve_variable_path(
    &self,
    scope: &str,
    name: &str,
    path: &[String],
) -> Result<Option<Value>, EvalError> {
    // Ok(None):       the root or some segment is absent (undefined).
    // Err(TypeError): a non-leaf segment existed but was not an object.
    todo!()
}
```

`path` holds **named** segments only and stops at the first subscript. In
`{{char:party[0].stats.hp}}` the host is asked for `party` with an empty path
and the evaluator walks `[0].stats.hp` into the result. A field is something
storage can plausibly address; a position in a returned array is not.

## Registering callables

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

Add `.describe()`, `.returns()`, `.param()` and `.property()` to give a
closure a declared signature.

### Attribute macros

The `weaver_macros` crate generates the trait implementation from a function
signature, including argument type validation and the signature an editor
reads:

```rust
use weaver_lang::{Value, EvalError};
use weaver_lang::macros::weaver_processor;

/// Repeat a string.
#[weaver_processor(namespace = "text", name = "repeat", returns = "string")]
fn repeat_text(text: String, count: f64) -> Result<Value, EvalError> {
    Ok(Value::String(text.repeat(count as usize)))
}

// Generates `RepeatTextProcessor`, which implements `WeaverProcessor`:
// registry.register_processor(RepeatTextProcessor);
```

A command opts into context access by naming a parameter `ctx`:

```rust
use weaver_lang::{Value, EvalError, EvalContext};
use weaver_lang::macros::weaver_command;

/// Set a variable in a writable scope.
#[weaver_command(name = "set_var", returns = "none")]
fn set_var(key: String, value: Value, ctx: &mut dyn EvalContext) -> Result<Option<Value>, EvalError> {
    if let Some(pos) = key.find(':') {
        ctx.set_variable(&key[..pos], &key[pos + 1..], value)?;
    }
    Ok(None)
}
```

`namespace` (processors only), `name` and `returns` are required; `returns` is
one of `string`, `number`, `bool`, `array`, `none`, `any`. The function's doc
comment becomes the description, and `params(key = "...")` supplies per-
argument descriptions.

Parameter types are `Value` (anything), `String`, `f64`, `bool`, `Vec<Value>`,
and `Option<T>` of any of those. `Option<T>` is how you declare an optional
argument: an absent argument, or an explicit `none`, arrives as `None` instead
of erroring.

### Trait implementations

For full control, implement `WeaverCommand` or `WeaverProcessor` directly:

```rust
use weaver_lang::{Value, EvalError, EvalContext, Registry};
use weaver_lang::registry::{WeaverCommand, CommandSignature, ValueType};

struct MyCommand;

impl WeaverCommand for MyCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        // Full access to args, context, and registry.
        Ok(None)
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "my_command".to_string(),
            description: "Do the thing".to_string(),
            returns: ValueType::None,
            params: Vec::new(),
            mutates_context: true,
        }
    }
}
```

### Introspection

`Registry::command_signatures()` and `processor_signatures()` return what has
been registered: names, parameters, descriptions and declared return types.
Enable the `serde` feature to serialize that metadata and ship it to an editor
frontend for autocompletion.

## Evaluation options

`EvalOptions` configures resource limits, cancellation, and lenient mode:

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

Lenient mode passes an unresolvable reference through as its own source text,
which is what you want while an author is still typing. The one case it cannot
pass through is a computed subscript. The source text of `items[i]` does not
say what `i` was, so that yields `none` instead.

## Error reporting

Parse errors carry source spans. `format_with_source` turns one into a
diagnostic:

```
Error: undefined variable: global:player_name
 --> Dark Forest:12:6
  |
 12 |  {# if {{global:player_name}} #}
    |        ^^^^^^^^^^^^^^^^^^^^^^^
  = hint: did you mean to define this variable first?
```

Evaluation errors support chaining, so a host-originated failure keeps its
cause:

```rust
use weaver_lang::EvalError;

let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
let err = EvalError::host_error("failed to load entry").with_source(io_err);
// The full chain is reachable through std::error::Error::source().
```

## Weaver Template Notation (WTN)

WTN is a declarative template format. You write down the data structure you
want, holes and all, and evaluating it hands back that same structure with the
holes filled in.

A WTN document and a text template differ in what the document *is*. A text
template is the output text, with constructs embedded in it. A WTN document is
the output data, with the same constructs embedded in it. Where a text
template that needed to produce JSON would have to assemble that JSON as a
string, a WTN document simply is the JSON:

```
#inputs
characters: [Ref<Character>]
rounds: number = 3

{
  rounds: {{input:rounds}}
  agents: [
    {# foreach c in {{input:characters}} #}
    { character: c, prompt: "You are {{c}}." }
    {# endforeach #}
  ]
}
```

Nothing there describes how to build the result. It states what the result
looks like, and the `{{ }}` and `{# #}` mark the parts that vary from one
instantiation to the next. Evaluating it produces a `Value` of that shape, not
a string that has to be parsed back.

Below the entry rule it is the same language: the same expressions, literals,
loops, `EvalContext`, `Registry` and `EvalOptions`. Reach for WTN when the
thing you are templating is structured (a config file, a character sheet, a
set of agent definitions) and you want loops and inputs inside it without
giving up its shape.

Enable the `wtn` feature:

```toml
weaver_lang = { version = "0.8", features = ["wtn"] }
```

```rust
use std::collections::BTreeMap;
use weaver_lang::{Registry, SimpleContext, expand_value_doc};

let mut ctx = SimpleContext::new();
let value = expand_value_doc(
    r#"{ name: "Rags to Riches", rounds: 2 + 1 }"#,
    &BTreeMap::new(),
    &mut ctx,
    &Registry::new(),
).unwrap();
assert_eq!(value.to_json(), r#"{"name":"Rags to Riches","rounds":3}"#);
```

`expand_value_doc` parses and evaluates in one step; use `parse_value_doc` and
`evaluate_value_doc` to parse once and instantiate many times.

Expansion produces a `Value` and stops there. Deserializing it into your own
type with `serde` is the structural check. The language does not build a
second type system to describe types Rust already describes.

### A JSON superset

Existing JSON parses as written, including `null` (an alias for `none`),
exponents (`1e5`), and JSON's full escape set (`\/`, `\b`, `\f`, `\uXXXX` with
surrogate pairs). A character file becomes a template the moment someone wants
a loop in it; there is no migration step.

On top of JSON: unquoted identifier keys, comma-*or*-newline separators,
trailing separators, comments, arbitrary expressions, and the three features
below.

### Loops and conditionals in value position

```
{
  agents: [
    {# foreach c in {{input:characters}} #}
    { character: c, prompt: "You are {{c}}." }
    {# endforeach #}
  ]
  stats: {
    {# foreach k in {{input:tracked}} #}
    "{{k}}": 0
    {# endforeach #}
  }
}
```

A loop body yields **elements** in array context and **entries** in object
context, and a mismatch is a parse error rather than a strange value later.
`{# if #}` / `{# elif #}` / `{# else #}` work the same way and are how you
filter; `{# break #}` and `{# continue #}` are not valid here.

### Strings nest text mode

A string literal in a data document is itself a text-mode template, so
structure comes from WTN and prose from text mode:

```
prompt: "You are {{c.name}}, {{c.role}}."
```

Quoting therefore decides the type: `"{{c.hp}}"` is the string `"10"` while
`{{c.hp}}` is the number `10`.

Object keys are strings, so a computed key needs no syntax of its own.
`"{{k}}"` is how a loop in object position names its entries. A key produced
twice is an error, exactly like a duplicate written by hand.

Because every construct starter is live inside a string, a string that should
*not* be a template needs a raw string:

```
literal: r"Answer using {{placeholder}}."
quoted:  r#"Say "hello" using {{x}}."#
```

### Enum variants

`Custom([...])` is sugar for serde's externally tagged form. One value is
carried directly, several as a list:

```
scheduler: Custom([{agent: "alice", turns: 2}])   → {"scheduler": {"Custom": [...]}}
span: Range(1, 10)                                → {"span": {"Range": [1, 10]}}
mode: "RoundRobin"                                → a unit variant is its string
```

The parenthesis must be glued to the name, so a reference on one line and a
parenthesised expression on the next stay two items. A struct variant is
written as the object serde reads it from: `{Custom: {a: 1}}`.

### Declared inputs

A data document can declare the inputs it expects, so a UI can generate the
form for instantiating it instead of making authors guess which fields exist:

```
#inputs
characters: [Ref<Character>]
difficulty: enum("easy", "normal", "brutal") = "normal"
rounds: number = 3

{ ... }
```

The type vocabulary is small and closed: `string`, `number`, `bool`,
`enum(...)`, `range(lo, hi)`, `span(lo, hi)`, `[T]`, `{field: T, ...}`, and
`Ref<Kind>`. That closure is what makes the form generable, since
`[Ref<Character>]` renders as a multi-picker and `enum(...)` as a dropdown.

`range` and `span` are the two sliders. A `range(0, 2)` input is one number
inside those inclusive bounds; a `span(1, 50)` is a `{from, to}` pair inside
them, with `from` at or below `to`. An object type names a fixed set of
fields, which is how an input carries more than an id:

```
#inputs
participants: [{char: Ref<Character>, talkativeness: number}]
temperature: range(0, 2) = 0.8
turns: span(1, 50) = {from: 4, to: 8}
```

Fields are checked in declaration order, and a value carrying a field the type
never declared is an error — the same reading as a value supplied for an input
nobody declared.

An input with no `=` is required. Defaults are ordinary expressions, applied
by the evaluator rather than by every host. Read the declarations off a parsed
document with `doc.inputs`.

Declared inputs are reachable as the `input` scope:

```
{{input:difficulty}}
{{input:characters[0]}}
{# foreach c in {{input:characters}} #}
```

In WTN the evaluator owns that scope. In a text template `input` is an
ordinary host scope like any other.

### Kinds and validation

`Ref<Character>` means "an id that must resolve to a live Character". The set
of kinds comes from a host-populated registry, the same way callables do, and
whether a given id resolves is a host callback:

```rust,ignore
// On your EvalContext:
fn validate_input(&self, kind: &str, value: &Value) -> Result<(), EvalError> {
    // Ok(()): the id names a live entity of this kind.
    // Err(_): reported against the declaration that asked for the value.
    todo!()
}

// Wherever you build the registry:
let mut registry = Registry::new();
registry.register_kind("Character");
```

The language checks the shapes it named (string, number, bool, enum
membership, bounds, list-of, object-of) and hands `Ref<Kind>` to the host, which is the only
party that can answer it. Everything is checked before expansion begins, and
every failure is reported against the declaration that asked for the value.

### References, not snapshots

A bare `c` in value position serializes to whatever the host put in the array:
the id, not a copy of the entity. Expanded documents therefore *reference*
their inputs, and editing the entity propagates to every instance.
Snapshotting is explicit and visible in the template:

```
agents: [
  {# foreach c in {{input:characters}} #}
  { character: c,                          // stores the id
    card: @[character.summary(of: c)] }    // embeds a summary
  {# endforeach #}
]
```

Each projection is a registered processor returning an object, so the summary
type lives in Rust rather than in the grammar.

### WTN and text templates

One grammar file serves both, and the entry rule is what separates them.
Anything WTN-specific (`#inputs`, the type vocabulary, strings-as-templates)
is reachable only from the WTN entry rule, whether or not the feature is on.
An `#inputs` block written in a text template is ordinary literal text:

```
#inputs                              renders as
characters: [Ref<Character>]    →    those three lines, verbatim
```

Everything else listed above belongs to the shared *value* grammar, so it is
reachable in a template too, but only inside a construct, never in prose:

```
{{ Custom(1) }}                                            → {"Custom":1}
{{ [{# foreach n in [1, 2] #} n * 10 {# endforeach #}] }}   → 10, 20
Custom(1) in prose                                         → Custom(1) in prose
```

A literal therefore means the same thing everywhere, and prose in a template
is byte-for-byte unaffected by WTN. The one word to watch is `null`,
which joins `true`, `false` and `none` as a word that cannot be read as a
reference. `{{null}}` is the none literal, so a loop binding by that name is
unreadable.

## Limitations

### The language

These hold for text templates and WTN alike, since the value grammar is
shared.

- All numbers are `f64`. Integers above 2^53 lose precision.
- No assignment syntax. Variable mutation goes through host-defined commands.
- A bare reference (`{{n}}`) reads loop bindings only, never the host's
  `local` scope. Write `{{local:n}}` for that.
- Objects do not iterate and do not compare equal with `==`.
- No slicing (`items[1:3]`). A subscript selects one element.
- `{# break #}` and `{# continue #}` are not valid in item position.
  Filter with `{# if ... #}` instead.
- Comments are `//` to end of line; there is no block comment form.
- A callable can only be called in the form its trait receives (named for a
  processor, positional for a command) unless it declares a signature to map
  through. The macros do this automatically; a bare closure does not.

### Text templates

- Trim markers are all-or-nothing, as in Jinja and Liquid. In item position
  there is no surrounding text to trim, so a marker written there parses and
  does nothing.
- Trigger and document behavior depends entirely on the host's
  `fire_trigger` and `resolve_document` implementations.
- `#inputs` is a WTN construct; in a text template it is literal text.

### WTN

- Every construct starter is live inside a string. A string containing
  `{{placeholder}}` meant for another tool, or a `[[wiki link]]`, will be
  evaluated and most likely error. Use `r"..."` for any string that should be
  taken literally.
- One `#inputs` block, at the top of the document. There is no way to share or
  import declarations across documents.
- The `input` scope belongs to the evaluator here and to the host in a text
  template, so the same reference resolves differently depending on which
  entry point parsed the document.
- Expansion produces a `Value` and stops. Nothing checks that it matches the
  Rust type you are about to deserialize into; that is `serde`'s job.

## Dependencies

- [pest](https://pest.rs/): PEG parser generator
- [thiserror](https://github.com/dtolnay/thiserror): error derive macros
- [syn](https://github.com/dtolnay/syn), [quote](https://github.com/dtolnay/quote),
  and [proc-macro2](https://github.com/alexcrichton/proc-macro2): proc macro
  infrastructure (`weaver-macros` only)

## License

MIT
