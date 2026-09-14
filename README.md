# The Weaver Language

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

### Overview
| type       | syntax                                                |
|------------|-------------------------------------------------------|
| variables  | `{{scope:name}}`, `{{scope:name.path.into.object}}`    |
| expressions| `{{ 1 + 2 }}`, `{{ global:gold - 10 }}`               |
| literals   | `{name: "Alice", hp: 10}`, `["sword", "shield"]`      |
| indexing   | `{{items[0]}}`, `{{obj["key"]}}`, `{{party[i].name}}` |
| processors | `@[namespace.name(foo: value1, bar: value2)]`         |
| commands   | `$[name(foo, bar)]`                                   |
| arguments  | named or positional on either: `@[p(1, 2)]`, `$[c(a: 1)]` |
| triggers   | `<trigger id="some_id">`                              |
| documents  | `[[some_id]]`                                         |
| if/else    | `{# if foo == bar #} baz {# endif #}`                 |
| foreach    | `{# foreach foo in bar #} - {{foo}} {# endforeach #}` |
| flow       | `{# break #}`, `{# continue #}`, `{# return expr #}`, `{# stop #}` |
| trim       | `{#- ... -#}`, `{{- ... -}}`, `$[- ... -]`            |
| comments   | `// to end of line`                                   |

### Variables

```
{{scope:name}}
{{char:alice.stats.hp}}
{{npc.name}}
```

Look up a variable by scope and name. `global` and `local` are conventional, but hosts can define any scope.

A reference may carry a **dotted path**. The first segment is the variable name — the part the host resolves — and every segment after it indexes into the resolved value:

```
{{char:alice.stats.hp}}
   │     │     └── path: indexed by the language
   │     └──────── name: resolved by the host
   └────────────── scope
```

Every non-leaf segment must be indexable — an object for a named segment, an array for a subscript; see [Objects](#objects). Paths work on bare loop bindings too (`{{npc.name}}`), which is what makes `foreach` over an array of objects useful.

A reference is an ordinary expression atom, so the delimiters are only needed where the surrounding text is prose. Inside a construct, write the reference on its own:

```
{# if global:hp > 5 #}...{# endif #}
{# foreach c in global:party #}...{# endforeach #}
$[set_var("local:n", n + 1)]
```

A **bare** reference resolves against loop bindings only. `{{n}}` inside a `foreach` is the binding; outside one it is an error, even if the host holds a `local:n`. The host's scopes are reachable only through `scope:name`, so a template never silently reads state it did not name.

A path that runs off the end of an object — `{{char:alice.stats.luck}}` where `luck` is absent — is reported exactly like a variable that does not exist: an `UndefinedVariable` error naming the full path, or, in lenient mode, the reference passed through unevaluated. Indexing something that *isn't* an object (`{{char:alice.stats.hp.max}}`) is a `TypeError` instead, because it can never be satisfied.

### Indexing — `items[0]`, `obj["key"]`

A subscript indexes an array by position or an object by key. `.name` and `["name"]` are the same thing, so a path may be written either way and the two mix freely:

```
{{char:alice.gear[0]}}          first item
{{char:party[0].stats.hp}}      subscript, then a dotted path
{{char:alice["stats"]["hp"]}}   same as {{char:alice.stats.hp}}
{{char:alice.gear[global:slot]}}  computed subscript
{{ ["a", "b"][1] }}             a literal is indexable too
```

A subscript must be **glued** to the value it indexes — no space before `[`. That is what keeps a newline usable as a separator: in

```
[a
 [1]]
```

the second line is an element, not a subscript of the first.

Three rules govern what a subscript means:

- **Out of range is absent.** `{{items[9]}}` on a three-element array behaves exactly like a missing object key: an `UndefinedVariable` error, or the reference passed through in lenient mode. There is no silent `none`.
- **Negative does not wrap.** `items[-1]` is a `TypeError`, not the last element. So is a fractional index. An off-by-one stays loud.
- **Nothing is coerced.** A number indexes an array and a string indexes an object; `items["0"]` and `obj[0]` are type errors rather than guesses.

A computed subscript is the one place lenient mode cannot pass source through — the text of `items[i]` depends on what `i` was — so it yields `none` instead. A constant subscript is folded into the reference's path at parse time and passes through like any other path.

### Objects

Objects are string-keyed maps. They can be supplied by the host or written as a literal:

```
$[set_var("local:c", {name: "Alice", hp: 10})]
{{ {party: [{name: "Alice"}, {name: "Bob"}]} }}
```

Keys are identifiers or quoted strings (`{"needs quoting": 1}`), so JSON parses as written. Values are arbitrary expressions. A **duplicate key is a parse error** rather than a silent discard, since the map is sorted and one of the two values would have to lose.

Objects are otherwise deliberately minimal:

- **Truthiness** — a non-empty object is truthy, an empty one is falsy, matching arrays and strings.
- **Rendering** — an object in template position renders as **compact JSON** with keys in sorted order, so output is deterministic across runs. Hosts wanting pretty-printed output can register a processor that formats the value themselves.
- **Iteration** — objects are not iterable. `foreach` still requires an array.
- **Equality** — like arrays, objects do not compare equal with `==`.

Note one asymmetry: a top-level array joins its elements (`{{tags}}` → `a, b`), but an array reached *inside* an object renders as JSON (`{{char:alice}}` → `{"tags":["a","b"]}`). The join predates objects and existing templates depend on it.

```rust
use weaver_lang::Value;

let alice = Value::object([
    ("name", Value::String("Alice".into())),
    ("stats", Value::object([("hp", 10i64)])),
]);
assert_eq!(alice.to_json(), r#"{"name":"Alice","stats":{"hp":10}}"#);
```

**Assignment** is still the host's business — a literal builds a value, but binding it to a name goes through a command such as `set_var`. `Value::set_path` is provided so that command folds paths into objects the same way reads unfold them (creating intermediate objects as needed, refusing to overwrite a non-object).

### Processors — `@[namespace.name(...)]`

Pure computations. No access to evaluation state.

```
@[math.add(a: 1, b: 2)]
@[core.weaver.rng(min: 1, max: 100)]
```

### Commands — `$[name(...)]`

Stateful operations. Can read and write variables through the evaluation context.

```
$[set_var("global:name", "Alice")]
$[greet("world")]
```

### Arguments

**Property markers are optional on both.** A processor's `call` receives named properties and a command's receives a positional list, but that is a detail of the traits, not something a template should have to track. Write whichever reads better:

```
@[math.add(a: 1, b: 2)]      @[math.add(1, 2)]
$[greet(name: "world")]      $[greet("world")]
```

Named arguments may be written in any order, and the two forms mix as long as positional ones come first:

```
@[text.repeat("ab", count: 3)]
$[join(1, c: 3, b: 2)]
```

Matching one form to the other goes through the **declared signature**, so a callable registered by the `#[weaver_processor]` / `#[weaver_command]` macros — which declare their parameters automatically — supports both with no extra work. A closure registered without `.property()` or `.param()` declares nothing to match against, so it keeps working in its own form and says so plainly if you use the other.

A name given twice, or given to a slot a positional argument already filled, is an error. So is a positional argument after a named one.

**A leading `name:` is always a marker.** At the top level of an argument list, `$[cmd(local:x)]` names an argument `local` — it does not pass the reference `local:x`. Interpolate or parenthesise to pass one:

```
$[cmd({{local:x}})]      reference
$[cmd((local:x))]        reference
@[p(v: [local:a, b:c])]  nested, so both are references
```

The rule reaches only that top level; inside any nested literal a colon is an ordinary scoped reference.

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

<trigger id=({{user:name}} + "_inventory")>     // Complex expressions are also supported!
```

Both are expressions — they can appear in arrays, processor arguments, conditions, etc.

Triggers and documents are almost functionally identical, but they differ semantically:
- Triggers are meant to mark another entry for activation. It should not return a value.
- Documents replace the expression with the flat output of another entry.

> Evaluation is not performed automatically for documents, and triggers won't be activated automatically. You need to implement this logic in your `EvalContext` implementation.

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
  - {{item}}
{# endforeach #}
```

Control flow tags on their own lines don't produce blank lines in the output.

A construct is treated as occupying its own line when it sits at a line
start in the **rendered output** and is followed by a newline. What happens
then depends only on what it rendered:

- nothing, or only whitespace — the line is consumed entirely
- text ending in a newline — it supplies the line terminator, so the line's
  own trailing newline is dropped rather than doubled
- text not ending in a newline — the line's newline is still needed and is kept

Commands are the one exception: a standalone command always consumes its
line, because its return value is discarded in template position.

Because the rule reads rendered output, the two ways of writing a block
agree:

```
{# if true #}yes{# endif #}      both render "yes\n"
{# if true #}
yes
{# endif #}
```

### Flow statements

```
{# break #}       stop the innermost foreach
{# continue #}    skip to the next iteration
{# return expr #} stop evaluating this template; expr is its value
{# return #}      sugar for {# return none #}
{# stop #}        stop evaluating; keep the output rendered so far
```

`return` and `stop` are legal anywhere. `break` and `continue` are rejected
at parse time outside a `foreach`. An `if` does not introduce a loop, so `{# if ... #}{# break #}{# endif #}` at
the top level is an error, while the same thing inside a loop body is fine.

Flow never crosses an entry boundary. Triggers and documents re-enter
through the host with a fresh evaluator, so a `break` inside an included
document cannot terminate the including template's loop.

### Return values

Weaver is designed to both be functional and act as a template language,
because of this, it provides an alternative way to structure your files
in the form of return values. This means that your template will return
either whatever is passed into the first evaluated return statement,
or falls back to the template's rendered content.

A template's result is a **`Value`**, and its rendered text is that value
passed through `to_output_string()`. `evaluate` is a thin wrapper over
`evaluate_value`.

`{# return expr #}` makes `expr` the result, discarding whatever was
rendered before it. A bare `{# return #}` is sugar for `{# return none #}`.

```
{# if {{global:hp}} > 50 #}{# return #}{# endif #}
Wounded: {{global:hp}} HP
```

An entry guarded like this renders nothing at all when the condition holds.

`{# stop #}` is the other half: it produces no value and **keeps** the text
rendered so far, so it truncates rather than replaces.

```
Intro paragraph.
{# foreach x in {{global:items}} #}
  - {{x}}
{# endforeach #}
{# stop #}

Draft notes that never reach the output.
```

| | value | rendered text |
|---|---|---|
| `{# return expr #}` | `expr` | discarded |
| `{# return #}` | `none` | discarded |
| `{# stop #}` | the text | kept |

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

| terminal state       | `evaluate_value`      | `evaluate`                |
|----------------------|-----------------------|---------------------------|
| ran to the end       | `String(output)`      | output                    |
| `{# stop #}`         | `String(output)`      | output up to the stop     |
| `{# return #}`       | `None`                | `""`                      |
| `{# return expr #}`  | `expr`                | `expr.to_output_string()` |

`evaluate_value` is defined for every template, not only ones that return,
so hosts can adopt it without changing any existing template.

An entry that needs to emit prose *and* hand the host a payload should
stash the payload through the context — `$[set_var("local:tags", ...)]` —
and use no `return` at all — or `{# stop #}` — so the prose stays the
result. Note that `{# return #}` is *not* the way to do this: it returns
none and discards the prose.

To let one entry return a value to *another*, override
`EvalContext::resolve_document_value`; see
[Implementing EvalContext](#implementing-evalcontext).

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

There is no ambiguity with subtraction or unary minus: `{# if x - 1 -#}`
parses as `x - 1` followed by a marker, and `{# if -1 < 0 #}` is unaffected.

Note that a marker is **all or nothing**. It cannot collapse a run of blank lines
  down to exactly one newline. You choose between all the whitespace and
  none of it.

### Expressions and operators

Expressions appear in conditions, arguments, literals, and inside `{{ }}`. Types are preserved internally (string, number, bool, array, object, none) and coerced to strings only at the template output level.

`{{ }}` holds any expression, not just a name, so inline arithmetic needs no processor:

```
{{ 1 + 2 }}
{{ char:alice.stats.hp * 2 }}
{{ {name: "Alice", hp: 10} }}
```

**Comparison:** `==`, `!=`, `<`, `>`, `<=`, `>=`
**Logical:** `&&`, `||`, `!`
**Arithmetic:** `+`, `-`, `*`, `/`, `%`

`+` concatenates when either operand is a string. Division by zero returns an error.

**Truthiness:** empty string, `0`, `false`, empty array, and `none` are falsy. Everything else is truthy.

**Precedence** (highest to lowest): indexing (`[]`, `.`), unary (!, -), arithmetic (*, /, +, -), comparison (==, !=, <, >, <=, >=), logical (&&, ||). Parentheses override precedence.

**Line breaks.** An expression continues freely across lines *after* an operator, but a binary operator must start on the same line as its left operand:

```
@[p(x: 1 +
       2)]      one expression
@[p(x: 1
     + 2)]      not one expression
```

This is what lets a newline separate elements. Without it `[1` / `-2]` on two lines would be the single element `-1` rather than the two elements written.

**Comments.** `//` runs to the end of the line, anywhere whitespace is allowed between tokens. A comment ends the line for the operator rule too, so it can never glue two items together.

**Raw strings.** `r"..."` is a string with no escape sequences — and, in [data mode](#data-mode), no template parsing. Use `r#"..."#` when the text contains a quote.

**Loops in a literal.** `{# foreach #}` and `{# if #}` may stand where an element or an entry would:

```
[{# foreach n in [1, 2, 3] #} n * 10 {# endforeach #}]   → [10, 20, 30]
```

An array item is an element and an object item is an entry; putting one where the other belongs is a parse error rather than a surprise at evaluation time.

**Separators.** Inside arrays, objects, argument lists and property lists, elements are separated by a comma, a newline, or both, and a trailing separator is allowed:

```
{
  name: "Rags to Riches"
  agents: [
    {name: "Alice", role: "thief"}
    {name: "Bob", role: "fence"}
  ]
}
```

## Data mode

Realms are expensive to author by hand. A template lets one well-built realm be reinstantiated with new inputs instead of editing every agent by hand — which needs a document that is **structured** rather than textual.

Data mode is a second entry rule, not a second language. Text mode treats a document as prose with holes; data mode treats it as a value with holes. Below the entry rule it is the same expressions, the same literals, the same loops, the same `EvalContext`, `Registry` and `EvalOptions`.

Enable the `data` feature:

```toml
weaver_lang = { version = "0.7", features = ["data"] }
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

Expansion produces a `Value`. Deserializing that into your own type with `serde` is the structural check — the language deliberately does not build a second type system to describe types Rust already describes.

Use `parse_value_doc` and `evaluate_value_doc` when you want to parse once and instantiate many times; `expand_value_doc` does both.

### A JSON superset

Existing JSON parses as written — including `null`, exponents (`1e5`), and JSON's full escape set (`\/`, `\b`, `\f`, `\uXXXX` with surrogate pairs). `null` is an alias for `none`. There is no migration step: a character file becomes a template the moment someone wants a loop in it.

On top of JSON: unquoted identifier keys, comma-*or*-newline separators, trailing separators, comments, expressions, and the three things below.

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

A loop body yields **elements** in array context and **entries** in object context. The two have separate grammar rules, so a mismatch is a parse error rather than something that surfaces as a strange value later. `{# if #}`/`{# elif #}`/`{# else #}` work the same way and are how you filter — `{# break #}` and `{# continue #}` are not valid here.

### Strings nest text mode

A string literal in a data document is itself a text-mode template. Structure comes from data mode, prose from text mode, and an agent system prompt stops being a special case:

```
prompt: "You are {{c.name}}, {{c.role}}."
```

The corollary is that quoting decides the type: `"{{c.hp}}"` is the string `"10"` and `{{c.hp}}` is the number `10`.

Object keys are strings, so a computed key needs no syntax of its own — `"{{k}}"` is how a loop in object position names its entries. A key produced twice is an error, exactly like a duplicate written by hand.

For a string that should *not* be a template — a prompt containing `{{placeholder}}` meant for some other tool — use a raw string:

```
literal: r"Answer using {{placeholder}}."
quoted:  r#"Say "hello" using {{x}}."#
```

### Enum variants

`Custom([...])` is sugar for serde's externally tagged form. One value is carried directly, several as a list:

```
scheduler: Custom([{agent: "alice", turns: 2}])   → {"scheduler": {"Custom": [...]}}
span: Range(1, 10)                                → {"span": {"Range": [1, 10]}}
mode: "RoundRobin"                                → a unit variant is its string
```

The parenthesis must be glued to the name, so a reference on one line and a parenthesised expression on the next stay two items.

### Declared inputs

```
#inputs
characters: [Ref<Character>]
difficulty: enum("easy", "normal", "brutal") = "normal"
rounds: number = 3

{ ... }
```

Inputs are **declared and typed**, so a UI can generate the instantiation form instead of making authors guess which fields exist. Read them off a parsed document with `doc.inputs`.

The vocabulary is small and closed — `string`, `number`, `bool`, `enum(...)`, `[T]`, and `Ref<Kind>` — because that closure is what makes the form generable: `[Ref<Character>]` renders as a character multi-picker, `enum(...)` as a dropdown.

An input with no `=` is required. Defaults are ordinary expressions, applied by the evaluator rather than by every host.

Declared inputs are reachable as the `input` scope:

```
{{input:difficulty}}
{{input:characters[0]}}
{# foreach c in {{input:characters}} #}
```

In data mode the evaluator owns that scope. In text mode `input` is an ordinary host scope like any other.

### Kinds and validation

`Ref<Character>` means "a Snowflake that must resolve to a live Character". The set of kinds comes from a host-populated registry, the same way processors and commands do, and whether a given id resolves is a host callback:

```rust
use weaver_lang::{EvalContext, EvalError, Registry, Value};

// On your EvalContext:
fn validate_input(&self, kind: &str, value: &Value) -> Result<(), EvalError> {
    // Ok(())  — the id names a live entity of this kind.
    // Err(_)  — reported against the declaration that asked for the value.
    todo!()
}

// Wherever you build the registry:
let mut registry = Registry::new();
registry.register_kind("Character");
```

The language checks the shapes it named — string, number, bool, enum membership, list-of — and hands `Ref<Kind>` to the host, which is the only party that can answer it. Everything is checked before expansion begins, and every failure is reported against the declaration that asked for the value.

### Refs, not snapshots

A bare `c` in value position serializes to whatever the host put in the array — the Snowflake, not a copy of the character. Expanded realms therefore *reference* characters, and editing a character propagates to every instance. Snapshotting is explicit and visible in the template:

```
agents: [
  {# foreach c in {{input:characters}} #}
  { character: c,                          // stores the id
    card: @[character.summary(of: c)] }    // embeds a summary
  {# endforeach #}
]
```

Each projection is a registered processor returning a `Value::Object`. The summary type exists in Rust and nowhere in the grammar.

### Instantiation

Expand once, at instantiation. A template produces a concrete document that is then ordinary data, so debugging a broken instance means reading real data rather than re-running an expansion.

### What the feature gates

pest compiles one grammar file, so the `data` feature cannot remove rules from it. What separates the modes is the **entry rule**, and it separates them completely for everything data-mode-specific:

**Reachable only from the data-mode entry rule.** `#inputs`, the type vocabulary, and strings-as-templates. None of these exist in a template, with the feature on or off. An `#inputs` block written in a text-mode entry is ordinary literal text — not a declaration, and not an error:

```
#inputs                              renders as
characters: [Ref<Character>]    →    those three lines, verbatim
```

`#` is not a construct starter, and the additions below are not either, so `literal_text` still ends at exactly the six delimiters it always did. **Prose in a template is byte-for-byte unaffected.**

**Shared, by design.** Enum variants, raw strings, `//` comments, loops in literals, `null`, exponents and JSON's escape set are part of the value grammar, which both modes use. They are reachable in a template, but only *inside* a construct — `{{ }}`, `{# #}`, `@[ ]`, `$[ ]` — never in prose:

```
{{ Custom(1) }}                                     → {"Custom":1}
{{ [{# foreach n in [1, 2] #} n * 10 {# endforeach #}] }}   → 10, 20
Custom(1) in prose                                  → Custom(1) in prose
```

This is the point of one grammar rather than two: a literal means the same thing everywhere, and there is no second dialect to drift. Each of these was a parse error before, so nothing that used to parse changed meaning.

The one exception is `null`, which joins `true`, `false` and `none` as a word that cannot be read as a reference. `{{null}}` is the none literal, so a loop binding by that name is unreadable.

`tests/mode_boundary.rs` pins all of this down, and is not gated on the feature — the guarantee is that text mode behaves identically either way.

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
        // Look up variables from your storage. `name` is always a single
        // segment — dotted paths are indexed into the value you return.
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

### Documents that return values

`resolve_document_value` has a default implementation that wraps
`resolve_document` in a `Value::String`, which is exactly the old
behaviour. Override it when your entries use `{# return #}`, and a
document becomes a value-producing unit that other entries can consume:

```rust
# use weaver_lang::{EvalContext, EvalError, Registry, Value, parse, evaluate_value};
# struct GameContext;
# impl GameContext { fn source_of(&self, _id: &str) -> String { String::new() } }
# impl EvalContext for GameContext {
#     fn resolve_variable(&self, _s: &str, _n: &str) -> Result<Option<Value>, EvalError> { Ok(None) }
#     fn set_variable(&mut self, _s: &str, _n: &str, _v: Value) -> Result<(), EvalError> { Ok(()) }
#     fn fire_trigger(&mut self, _i: &str, _r: &Registry) -> Result<String, EvalError> { Ok(String::new()) }
#     fn resolve_document(&mut self, id: &str, r: &Registry) -> Result<String, EvalError> {
#         Ok(self.resolve_document_value(id, r)?.to_output_string())
#     }
fn resolve_document_value(
    &mut self,
    document_id: &str,
    registry: &Registry,
) -> Result<Value, EvalError> {
    let source = self.source_of(document_id);
    let template = parse(&source).map_err(|_| EvalError::host_error("parse failed"))?;
    evaluate_value(&template, self, registry)
}
# }
```

```
// LOOT_TABLE:  {# return ["sword", "shield", "potion"] #}

{# foreach item in [[LOOT_TABLE]] #}
  - {{item}}
{# endforeach #}
```

There is deliberately no trigger counterpart. A trigger marks another
entry for activation rather than producing content, so it has no value to
carry.

The evaluator manages temporary scopes internally (foreach bindings). Only named scope operations like `"global"` and `"local"` reach the host.

### Pushing paths down into storage

`resolve_variable_path` has a default implementation that resolves the root through `resolve_variable` and walks the path with `Value::get_path`. That is correct for every host, but it has to materialize the whole root value — an entire character sheet, say — to read one field. Override it when your storage can index the path directly:

```rust
fn resolve_variable_path(
    &self,
    scope: &str,
    name: &str,
    path: &[String],
) -> Result<Option<Value>, EvalError> {
    // Ok(None)  — root or some segment is absent (treated as undefined).
    // Err(TypeError) — a non-leaf segment existed but was not an object.
    todo!()
}
```

`path` holds **named** segments only, and stops at the first subscript. In `{{char:party[0].stats.hp}}` the host is asked for `party` with an empty path; the evaluator walks `[0].stats.hp` into the value it gets back. The split reflects what a host can plausibly push into storage: a field is addressable, a position in a returned array is not.

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

### The language

These hold in both modes. The value grammar is shared, so a literal means the same thing inside `$[cmd(...)]` as it does in a data document.

- All numbers are `f64`. Large integers above 2^53 lose precision.
- No assignment syntax. Variable mutation goes through commands the host defines.
- A bare reference (`{{n}}`) reads loop bindings only, never the host's `local` scope. Write `{{local:n}}` for that.
- Objects do not iterate and do not compare equal with `==`. `foreach` still requires an array.
- Slicing (`items[1:3]`) is not supported. A subscript selects one element.
- `{# break #}` and `{# continue #}` are not valid in item position, in either mode. Filter with `{# if ... #}` instead.
- Enum sugar covers newtype and tuple variants (`Custom(x)`, `Range(1, 10)`). A struct variant is written as the object serde reads it from: `{Custom: {a: 1}}`.
- Comments are `//` to end of line. There is no block comment form.
- Document evaluation depends on the host's `resolve_document` implementation.
- Writing an argument in the form a callable's trait does not take requires a declared signature to map through. A closure registered without `.property()` or `.param()` can only be called the way its `call` receives arguments — named for a processor, positional for a command.

### Text mode

- Trim markers are all-or-nothing: `-` removes every whitespace character on its side, so it cannot collapse a run of blank lines to exactly one newline. This is shared with Jinja and Liquid. In item position — inside a literal, in either mode — there is no surrounding text to trim, so a marker written there parses and does nothing.
- `#inputs` is a data-mode construct. In a template it is ordinary literal text, not a declaration and not an error.

### Data mode (`data` feature)

- **Every construct starter is live inside a string.** A string literal is a text-mode template, so `{{`, `[[`, `@[`, `$[` and `<trigger` are all interpreted — including in JSON that predates the template. A prompt containing `{{placeholder}}` meant for another tool, or a `[[wiki link]]`, will be evaluated and most likely error. Use `r"..."` for any string that should be taken literally.
- One `#inputs` block, at the top of the document. There is no way to share or import a set of declarations across documents.
- The `input` scope belongs to the evaluator here and to the host in text mode. The same reference resolves differently depending on which entry point parsed the document.
- Expansion produces a `Value` and stops. Nothing checks that it matches the Rust type you are about to deserialize into — that is `serde`'s job, deliberately.
- The `data` feature gates the API, the AST and evaluation, but not the grammar file — see [What the feature gates](#what-the-feature-gates) for exactly where the line falls.

## Dependencies

- [pest](https://pest.rs/) — PEG parser generator
- [thiserror](https://github.com/dtolnay/thiserror) — error derive macros
- [syn](https://github.com/dtolnay/syn), [quote](https://github.com/dtolnay/quote), [proc-macro2](https://github.com/alexcrichton/proc-macro2) — proc macro infrastructure (weaver-macros only)

## License

MIT
