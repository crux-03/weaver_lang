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
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Bool(bool),
    Array(Vec<Value>),
    /// The absence of a value. Falsy, renders as an empty string.
    None,
}

impl Value {
    /// Convert this value to its string representation for template output.
    ///
    /// - `String` — returned as-is
    /// - `Number` — formatted without trailing `.0` for whole numbers
    /// - `Bool` — `"true"` or `"false"`
    /// - `Array` — elements joined with `", "`
    /// - `None` — empty string
    pub fn to_output_string(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Number(n) => {
                if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Value::Array(items) => items
                .iter()
                .map(|v| v.to_output_string())
                .collect::<Vec<_>>()
                .join(", "),
            Value::None => String::new(),
        }
    }

    /// Type name for diagnostic messages
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::String(_) => "string",
            Value::Number(_) => "number",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::None => "none",
        }
    }

    /// Truthiness check, used by `{# if ... #}` and `&&`/`||` operators.
    ///
    /// Falsy values: empty string, `0`, `false`, empty array, `None`.
    /// Everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::String(s) => !s.is_empty(),
            Value::Number(n) => *n != 0.0,
            Value::Bool(b) => *b,
            Value::Array(a) => !a.is_empty(),
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
