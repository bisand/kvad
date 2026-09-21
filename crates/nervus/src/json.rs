//! Just enough JSON to write a checkpoint and read it back.
//!
//! Nothing in here is about neural networks. It exists because the three files
//! a saved model consists of are all JSON, or begin with it, and this crate
//! has no dependencies. It is a complete parser for the JSON grammar, because
//! half a parser is how files from other tools get misread; it is not a fast
//! one. Skip this file unless you are curious: a recursive-descent parser is
//! one function per line of the grammar, and that is all there is to see.
//!
//! An object keeps its keys in the order they were written, as a list of
//! pairs. The files are small, and a header that lists its tensors in the
//! order the model holds them is easier to read than one a hash map shuffled.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { chars: text.chars().collect(), at: 0 };
        let value = p.value()?;
        p.skip_space();
        match p.peek() {
            None => Ok(value),
            Some(c) => Err(p.error(&format!("unexpected {c:?} after the end of the value"))),
        }
    }

    /// A member of an object. `None` for a missing key, and for anything that
    /// is not an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// A whole, non-negative number: a size, an id, an offset.
    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            Json::Number(n) if n >= 0.0 && n.fract() == 0.0 && n <= 2f64.powi(53) => Some(n as usize),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(pairs) => Some(pairs),
            _ => None,
        }
    }
}

impl From<usize> for Json {
    fn from(n: usize) -> Json {
        Json::Number(n as f64)
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Json {
        Json::String(s.to_string())
    }
}

/// Build an object from `(key, value)` pairs.
pub fn object<const N: usize>(pairs: [(&str, Json); N]) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => write!(f, "null"),
            Json::Bool(b) => write!(f, "{b}"),
            // Rust prints a whole f64 without a decimal point and never uses
            // an exponent, both of which JSON accepts. JSON has no spelling
            // for infinity or NaN.
            Json::Number(n) if n.is_finite() => write!(f, "{n}"),
            Json::Number(_) => write!(f, "null"),
            Json::String(s) => write_string(f, s),
            Json::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Json::Object(pairs) => {
                write!(f, "{{")?;
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write_string(f, key)?;
                    write!(f, ":{value}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

fn write_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    write!(f, "\"")?;
    for c in s.chars() {
        match c {
            '"' => write!(f, "\\\"")?,
            '\\' => write!(f, "\\\\")?,
            '\n' => write!(f, "\\n")?,
            '\r' => write!(f, "\\r")?,
            '\t' => write!(f, "\\t")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => write!(f, "{c}")?,
        }
    }
    write!(f, "\"")
}

struct Parser {
    chars: Vec<char>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        self.at += 1;
        c
    }

    fn error(&self, what: &str) -> String {
        format!("JSON: {what} (at character {})", self.at)
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, want: char) -> Result<(), String> {
        match self.next() {
            Some(c) if c == want => Ok(()),
            Some(c) => Err(self.error(&format!("expected {want:?}, found {c:?}"))),
            None => Err(self.error(&format!("expected {want:?}, found the end of the text"))),
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_space();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => self.string().map(Json::String),
            Some('t') => self.word("true", Json::Bool(true)),
            Some('f') => self.word("false", Json::Bool(false)),
            Some('n') => self.word("null", Json::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(self.error(&format!("unexpected {c:?}"))),
            None => Err(self.error("the text ends where a value should start")),
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, String> {
        for want in word.chars() {
            self.expect(want)?;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')) {
            self.at += 1;
        }
        let text: String = self.chars[start..self.at].iter().collect();
        text.parse().map(Json::Number).map_err(|_| self.error(&format!("{text:?} is not a number")))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.next() {
                None => return Err(self.error("the text ends inside a string")),
                Some('"') => return Ok(out),
                Some('\\') => match self.next() {
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('u') => out.push(self.unicode_escape()?),
                    Some(c @ ('"' | '\\' | '/')) => out.push(c),
                    _ => return Err(self.error("unknown escape in a string")),
                },
                // The grammar forbids these raw, and stricter readers than
                // this one refuse the file. Refusing them here too means our
                // own writer cannot forget to escape one and still pass.
                Some(c) if (c as u32) < 0x20 => return Err(self.error("a raw control character in a string")),
                Some(c) => out.push(c),
            }
        }
    }

    /// `\uXXXX`. JSON escapes are UTF-16, so a character outside the basic
    /// plane arrives as two of them, a high half and then a low half.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let high = self.hex4()?;
        let code = if (0xD800..0xDC00).contains(&high) {
            self.expect('\\')?;
            self.expect('u')?;
            let low = self.hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return Err(self.error("half of a surrogate pair is missing"));
            }
            0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
        } else {
            high
        };
        char::from_u32(code).ok_or_else(|| self.error("a \\u escape that is not a character"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut code = 0;
        for _ in 0..4 {
            let digit = self.next().and_then(|c| c.to_digit(16));
            code = code * 16 + digit.ok_or_else(|| self.error("expected four hex digits after \\u"))?;
        }
        Ok(code)
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_space();
            match self.next() {
                Some(',') => {}
                Some(']') => return Ok(Json::Array(items)),
                _ => return Err(self.error("expected ',' or ']' in an array")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect('{')?;
        let mut pairs = Vec::new();
        self.skip_space();
        if self.peek() == Some('}') {
            self.at += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.expect(':')?;
            pairs.push((key, self.value()?));
            self.skip_space();
            match self.next() {
                Some(',') => {}
                Some('}') => return Ok(Json::Object(pairs)),
                _ => return Err(self.error("expected ',' or '}' in an object")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character a tokeniser's vocabulary might have to carry: the ones
    /// JSON must escape, the ones it may, and ones from well outside ASCII.
    const AWKWARD: &str = "a \"quoted\" back\\slash\nnewline\ttab \u{1} é → 😀 /";

    #[test]
    fn what_is_written_reads_back_the_same() {
        let value = object([
            ("text", AWKWARD.into()),
            ("sizes", Json::Array(vec![0.into(), 7.into(), 123_456_789_012.into()])),
            ("small", Json::Number(1e-5)),
            ("negative", Json::Number(-2.5)),
            ("flags", Json::Array(vec![Json::Bool(true), Json::Bool(false), Json::Null])),
            ("empty", Json::Array(vec![])),
            ("nested", object([(AWKWARD, Json::Object(vec![]))])),
        ]);
        assert_eq!(Json::parse(&value.to_string()), Ok(value));
    }

    /// Files from other tools are laid out differently and escape more than
    /// they need to.
    #[test]
    fn it_reads_json_it_did_not_write() {
        let text = " {\n  \"a\" : [ 1 , 2.5e1 , -3 ] ,\r\n\t\"s\": \"\\u0041\\u00e9\\ud83d\\ude00\\/\" , \"o\": { } }\n";
        let v = Json::parse(text).unwrap();
        let a = v.get("a").and_then(Json::as_array).unwrap();
        assert_eq!(a, [Json::Number(1.0), Json::Number(25.0), Json::Number(-3.0)]);
        assert_eq!(v.get("s").and_then(Json::as_str), Some("Aé😀/"));
        assert_eq!(v.get("o"), Some(&Json::Object(vec![])));
        assert_eq!(v.get("missing"), None);
    }

    #[test]
    fn sizes_must_be_whole_and_non_negative() {
        assert_eq!(Json::Number(12.0).as_usize(), Some(12));
        assert_eq!(Json::Number(12.5).as_usize(), None);
        assert_eq!(Json::Number(-1.0).as_usize(), None);
        assert_eq!(Json::String("12".into()).as_usize(), None);
    }

    #[test]
    fn broken_json_is_an_error_not_a_guess() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "{\"a\":1} x", "\"open", "tru", "\"\\q\"", "\"\\ud83d\"", "\"\\ud83d\\u0041\"", "\"a\u{1}b\"", "1.2.3", "{a:1}"] {
            assert!(Json::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }
}
