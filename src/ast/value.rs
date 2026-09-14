use std::collections::BTreeMap;
use std::fmt;

/// The set of runtime value types in weaver-lang.
///
/// Expressions produce `Value`s during evaluation. When a `Value` appears
/// at the template level, it is converted to a string via
/// [`to_output_string`](Value::to_output_string). Internally, types are
/// preserved so that conditions and arithmetic operate correctly.
///
/// Conversion from common Rust types is provided via `From` impls:
///
/// ```rust
/// use weaver_lang::Value;
///
/// let s: Value = "hello".into();
/// let n: Value = 42i64.into();
/// let b: Value = true.into();
/// let a: Value = vec!["a", "b"].into();
/// let o: Value = Value::object([("hp", 10i64)]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Bool(bool),
    Array(Vec<Value>),
    /// A string-keyed map. Supplied by the host; the language can index
    /// into it with a dotted path (`{{char:alice.stats.hp}}`) but has no
    /// syntax for constructing one.
    ///
    /// Keys are stored sorted rather than in insertion order, so rendering
    /// is deterministic across runs — templates must produce byte-identical
    /// output for the same inputs.
    Object(BTreeMap<String, Value>),
    /// The absence of a value. Falsy, renders as an empty string.
    None,
}

/// One step of a path into a [`Value`].
///
/// A reference's path is static by construction: `{{c.items[0]}}` is two
/// segments, but `{{c.items[i]}}` is a reference to `c.items` wrapped in an
/// [`ExprKind::Index`](crate::ast::expr::ExprKind::Index), because `i`
/// cannot be known until evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// `.name` or `["name"]` — index an object.
    Key(String),
    /// `[0]` — index an array.
    Index(usize),
}

impl fmt::Display for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Key(k) => write!(f, "{k}"),
            PathSegment::Index(i) => write!(f, "{i}"),
        }
    }
}

/// Why a dotted path could not be walked into a [`Value`].
///
/// Distinguishes "there is no such key" (which the evaluator treats exactly
/// like an undefined variable) from "this segment is not indexable at all"
/// (a type error) — see [`Value::get_path`].
#[derive(Debug, Clone, PartialEq)]
pub struct PathError {
    /// The segment whose parent could not be indexed.
    pub segment: String,
    /// Type name of the value that was not an object.
    pub found: &'static str,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot index into {} with '{}'",
            self.found, self.segment
        )
    }
}

impl std::error::Error for PathError {}

impl Value {
    /// Convert this value to its string representation for template output.
    ///
    /// - `String` — returned as-is
    /// - `Number` — formatted without trailing `.0` for whole numbers
    /// - `Bool` — `"true"` or `"false"`
    /// - `Array` — elements joined with `", "`
    /// - `Object` — compact JSON (see [`to_json`](Value::to_json))
    /// - `None` — empty string
    ///
    /// Note the deliberate asymmetry: a top-level array joins its elements
    /// (`a, b`), but an array *inside* an object renders as JSON
    /// (`{"items":["a","b"]}`). The join predates objects and templates
    /// depend on it; JSON is only entered through an object.
    pub fn to_output_string(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Number(n) => format_number(*n),
            Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Value::Array(items) => items
                .iter()
                .map(|v| v.to_output_string())
                .collect::<Vec<_>>()
                .join(", "),
            Value::Object(_) => self.to_json(),
            Value::None => String::new(),
        }
    }

    /// Render this value as compact JSON — no spaces, keys in sorted order.
    ///
    /// This is how [`Value::Object`] reaches template output. Hosts that
    /// want pretty-printed objects can register a processor that formats
    /// the value themselves; the language only ever emits the compact form.
    ///
    /// Non-finite numbers (`NaN`, `±inf`) have no JSON representation and
    /// are emitted as `null`.
    ///
    /// ```rust
    /// use weaver_lang::Value;
    ///
    /// let v = Value::object([("name", "alice"), ("title", "the \"Bold\"")]);
    /// assert_eq!(v.to_json(), r#"{"name":"alice","title":"the \"Bold\""}"#);
    /// ```
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Value::String(s) => write_json_string(s, out),
            Value::Number(n) => {
                if n.is_finite() {
                    out.push_str(&format_number(*n));
                } else {
                    out.push_str("null");
                }
            }
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (key, val)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(key, out);
                    out.push(':');
                    val.write_json(out);
                }
                out.push('}');
            }
            Value::None => out.push_str("null"),
        }
    }

    /// Type name for diagnostic messages
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::String(_) => "string",
            Value::Number(_) => "number",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
            Value::None => "none",
        }
    }

    /// Truthiness check, used by `{# if ... #}` and `&&`/`||` operators.
    ///
    /// Falsy values: empty string, `0`, `false`, empty array, empty object,
    /// `None`. Everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::String(s) => !s.is_empty(),
            Value::Number(n) => *n != 0.0,
            Value::Bool(b) => *b,
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
            Value::None => false,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn into_array(self) -> Option<Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn into_object(self) -> Option<BTreeMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Build an object from any iterator of key/value pairs.
    ///
    /// ```rust
    /// use weaver_lang::Value;
    ///
    /// let stats = Value::object([("hp", 10i64), ("mp", 3i64)]);
    /// assert_eq!(stats.to_json(), r#"{"hp":10,"mp":3}"#);
    /// ```
    pub fn object<K, V>(entries: impl IntoIterator<Item = (K, V)>) -> Value
    where
        K: Into<String>,
        V: Into<Value>,
    {
        Value::Object(
            entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }

    /// Walk a dotted path into this value, following object keys.
    ///
    /// Every non-leaf segment must be an object — that is the whole rule.
    /// An empty path returns the value itself.
    ///
    /// - `Ok(Some(v))` — the path resolved.
    /// - `Ok(None)` — a key along the way does not exist. Callers should
    ///   treat this exactly as they treat a variable that does not exist.
    /// - `Err(_)` — a non-leaf segment was not an object, so indexing is
    ///   meaningless rather than merely unsatisfied.
    ///
    /// ```rust
    /// use weaver_lang::Value;
    ///
    /// let alice = Value::object([("stats", Value::object([("hp", 10i64)]))]);
    /// let path = ["stats".to_string(), "hp".to_string()];
    /// assert_eq!(alice.get_path(&path).unwrap(), Some(&Value::Number(10.0)));
    /// ```
    pub fn get_path(&self, path: &[String]) -> Result<Option<&Value>, PathError> {
        let mut current = self;
        for segment in path {
            let Value::Object(map) = current else {
                return Err(PathError {
                    segment: segment.clone(),
                    found: current.type_name(),
                });
            };
            match map.get(segment) {
                Some(next) => current = next,
                None => return Ok(None),
            }
        }
        Ok(Some(current))
    }

    /// Walk a reference's path into this value.
    ///
    /// The segment-aware counterpart to [`get_path`](Value::get_path): a
    /// [`Key`](PathSegment::Key) indexes an object, an
    /// [`Index`](PathSegment::Index) indexes an array.
    /// The three outcomes are the same, and an index past the end of an
    /// array is `Ok(None)` — "absent" — exactly like a missing key.
    ///
    /// ```rust
    /// use weaver_lang::{PathSegment, Value};
    ///
    /// let alice = Value::object([("gear", Value::from(vec!["sword", "shield"]))]);
    /// let path = [PathSegment::Key("gear".into()), PathSegment::Index(1)];
    /// assert_eq!(
    ///     alice.get_segments(&path).unwrap(),
    ///     Some(&Value::String("shield".into())),
    /// );
    /// ```
    pub fn get_segments(&self, path: &[PathSegment]) -> Result<Option<&Value>, PathError> {
        let mut current = self;
        for segment in path {
            let next = match (current, segment) {
                (Value::Object(map), PathSegment::Key(key)) => map.get(key),
                (Value::Array(items), PathSegment::Index(i)) => items.get(*i),
                _ => {
                    return Err(PathError {
                        segment: segment.to_string(),
                        found: current.type_name(),
                    });
                }
            };
            match next {
                Some(value) => current = value,
                None => return Ok(None),
            }
        }
        Ok(Some(current))
    }

    /// Write a value at a dotted path, creating intermediate objects.
    ///
    /// This is the write-side counterpart to [`get_path`](Value::get_path),
    /// provided so hosts implementing a `set_var`-style command can reuse
    /// the same folding rule instead of reimplementing it. The language
    /// itself never calls this — it has no assignment syntax.
    ///
    /// Missing intermediate segments are created as empty objects, as is a
    /// receiver that is currently [`Value::None`]. An intermediate segment
    /// that exists but is *not* an object is an error rather than being
    /// silently overwritten.
    ///
    /// ```rust
    /// use weaver_lang::Value;
    ///
    /// let mut alice = Value::None;
    /// let path = ["gear".to_string(), "weapon".to_string()];
    /// alice.set_path(&path, "sword".into()).unwrap();
    /// assert_eq!(alice.to_json(), r#"{"gear":{"weapon":"sword"}}"#);
    /// ```
    pub fn set_path(&mut self, path: &[String], value: Value) -> Result<(), PathError> {
        let Some((leaf, parents)) = path.split_last() else {
            *self = value;
            return Ok(());
        };

        let mut current = self;
        for segment in parents {
            current = vivify(current, segment)?
                .entry(segment.clone())
                .or_insert_with(|| Value::Object(BTreeMap::new()));
        }
        vivify(current, leaf)?.insert(leaf.clone(), value);
        Ok(())
    }
}

/// Coerce `target` into an object so a key can be written to it, treating
/// `None` as an absent object rather than a wrong type.
fn vivify<'a>(
    target: &'a mut Value,
    segment: &str,
) -> Result<&'a mut BTreeMap<String, Value>, PathError> {
    if matches!(target, Value::None) {
        *target = Value::Object(BTreeMap::new());
    }
    match target {
        Value::Object(map) => Ok(map),
        other => Err(PathError {
            segment: segment.to_string(),
            found: other.type_name(),
        }),
    }
}

/// Format a number the way templates expect: whole values lose the `.0`.
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_output_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Number(n as f64)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<BTreeMap<String, T>> for Value {
    fn from(m: BTreeMap<String, T>) -> Self {
        Value::Object(m.into_iter().map(|(k, v)| (k, v.into())).collect())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn path(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn json_escapes_the_characters_that_would_break_it() {
        let v = Value::object([("quote\"back\\slash", "tab\there\nnewline\u{1}control")]);
        assert_eq!(
            v.to_json(),
            r#"{"quote\"back\\slash":"tab\there\nnewline\u0001control"}"#
        );
    }

    #[test]
    fn json_writes_non_finite_numbers_as_null() {
        // NaN and infinity are reachable through arithmetic but have no
        // JSON form.
        let v = Value::object([("nan", f64::NAN), ("inf", f64::INFINITY)]);
        assert_eq!(v.to_json(), r#"{"inf":null,"nan":null}"#);
    }

    #[test]
    fn json_nests_arrays_and_objects() {
        let v = Value::object([
            ("items", Value::Array(vec!["a".into(), "b".into()])),
            ("nested", Value::object([("deep", true)])),
            ("nothing", Value::None),
        ]);
        assert_eq!(
            v.to_json(),
            r#"{"items":["a","b"],"nested":{"deep":true},"nothing":null}"#
        );
    }

    #[test]
    fn object_keys_render_in_sorted_order() {
        // Determinism: the same inputs must produce the same bytes.
        let a = Value::object([("z", 1i64), ("a", 2i64), ("m", 3i64)]);
        let b = Value::object([("m", 3i64), ("z", 1i64), ("a", 2i64)]);
        assert_eq!(a.to_json(), b.to_json());
        assert_eq!(a.to_json(), r#"{"a":2,"m":3,"z":1}"#);
    }

    #[test]
    fn top_level_array_still_joins() {
        // Unchanged from before objects existed — templates depend on it.
        let v = Value::Array(vec!["a".into(), "b".into()]);
        assert_eq!(v.to_output_string(), "a, b");
    }

    #[test]
    fn empty_path_returns_the_value_itself() {
        let v = Value::Number(1.0);
        assert_eq!(v.get_path(&[]).unwrap(), Some(&Value::Number(1.0)));
    }

    #[test]
    fn get_path_distinguishes_absent_from_unindexable() {
        let v = Value::object([("stats", Value::object([("hp", 10i64)]))]);

        assert_eq!(v.get_path(&path(&["stats", "luck"])).unwrap(), None);
        assert_eq!(v.get_path(&path(&["missing", "hp"])).unwrap(), None);

        let err = v.get_path(&path(&["stats", "hp", "max"])).unwrap_err();
        assert_eq!(err.found, "number");
        assert_eq!(err.segment, "max");
    }

    #[test]
    fn set_path_creates_intermediate_objects() {
        let mut v = Value::None;
        v.set_path(&path(&["a", "b", "c"]), 1i64.into()).unwrap();
        assert_eq!(v.to_json(), r#"{"a":{"b":{"c":1}}}"#);
    }

    #[test]
    fn set_path_preserves_siblings() {
        let mut v = Value::object([("stats", Value::object([("hp", 10i64)]))]);
        v.set_path(&path(&["stats", "mp"]), 3i64.into()).unwrap();
        assert_eq!(v.to_json(), r#"{"stats":{"hp":10,"mp":3}}"#);
    }

    #[test]
    fn set_path_refuses_to_clobber_a_scalar() {
        let mut v = Value::object([("hp", 10i64)]);
        let err = v.set_path(&path(&["hp", "max"]), 20i64.into()).unwrap_err();
        assert_eq!(err.found, "number");
        // The original value is untouched.
        assert_eq!(v.to_json(), r#"{"hp":10}"#);
    }

    #[test]
    fn set_path_with_empty_path_replaces_the_whole_value() {
        let mut v = Value::object([("hp", 10i64)]);
        v.set_path(&[], "gone".into()).unwrap();
        assert_eq!(v, Value::String("gone".to_string()));
    }

    #[test]
    fn empty_object_is_falsy() {
        assert!(!Value::object(Vec::<(String, Value)>::new()).is_truthy());
        assert!(Value::object([("a", 1i64)]).is_truthy());
    }
}
