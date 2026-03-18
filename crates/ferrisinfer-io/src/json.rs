use std::collections::BTreeMap;

use ferrisinfer_core::{ErrorKind, FerrisError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonNumber {
    repr: String,
}

impl JsonNumber {
    pub fn new(repr: String) -> Self {
        Self { repr }
    }

    pub fn as_u64(&self) -> Result<u64> {
        self.repr.parse::<u64>().map_err(|_| {
            FerrisError::new(
                ErrorKind::Parse,
                format!("failed to parse JSON number '{}' as u64", self.repr),
            )
        })
    }

    pub fn as_f64(&self) -> Result<f64> {
        self.repr.parse::<f64>().map_err(|_| {
            FerrisError::new(
                ErrorKind::Parse,
                format!("failed to parse JSON number '{}' as f64", self.repr),
            )
        })
    }
}

impl JsonValue {
    pub fn as_object(&self) -> Result<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(object) => Ok(object),
            _ => Err(FerrisError::new(ErrorKind::Parse, "expected JSON object")),
        }
    }

    pub fn as_array(&self) -> Result<&[JsonValue]> {
        match self {
            Self::Array(array) => Ok(array),
            _ => Err(FerrisError::new(ErrorKind::Parse, "expected JSON array")),
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(FerrisError::new(ErrorKind::Parse, "expected JSON string")),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(FerrisError::new(ErrorKind::Parse, "expected JSON bool")),
        }
    }

    pub fn as_number(&self) -> Result<&JsonNumber> {
        match self {
            Self::Number(value) => Ok(value),
            _ => Err(FerrisError::new(ErrorKind::Parse, "expected JSON number")),
        }
    }

    pub fn get<'a>(&'a self, key: &str) -> Result<&'a JsonValue> {
        self.as_object()?
            .get(key)
            .ok_or_else(|| FerrisError::new(ErrorKind::Parse, format!("missing JSON key '{key}'")))
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue> {
    let mut parser = JsonParser::new(input);
    let value = parser.parse_value()?;
    parser.skip_whitespace();

    if !parser.is_eof() {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            "unexpected trailing JSON content",
        ));
    }

    Ok(value)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            position: 0,
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_whitespace();

        match self.peek_byte() {
            Some(b'n') => self.parse_null(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(self.error(format!(
                "unexpected JSON byte '{}' at position {}",
                other as char, self.position
            ))),
            None => Err(self.error("unexpected end of JSON input")),
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue> {
        self.expect_bytes(b"null")?;
        Ok(JsonValue::Null)
    }

    fn parse_bool(&mut self) -> Result<JsonValue> {
        if self.try_expect_bytes(b"true") {
            return Ok(JsonValue::Bool(true));
        }

        if self.try_expect_bytes(b"false") {
            return Ok(JsonValue::Bool(false));
        }

        Err(self.error("invalid JSON boolean literal"))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.position;

        if self.peek_byte() == Some(b'-') {
            self.position += 1;
        }

        match self.peek_byte() {
            Some(b'0') => {
                self.position += 1;
            }
            Some(b'1'..=b'9') => {
                self.position += 1;
                while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                    self.position += 1;
                }
            }
            _ => return Err(self.error("invalid JSON number literal")),
        }

        if self.peek_byte() == Some(b'.') {
            self.position += 1;
            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid JSON fraction"));
            }

            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }

        if matches!(self.peek_byte(), Some(b'e') | Some(b'E')) {
            self.position += 1;

            if matches!(self.peek_byte(), Some(b'+') | Some(b'-')) {
                self.position += 1;
            }

            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid JSON exponent"));
            }

            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }

        let repr = std::str::from_utf8(&self.bytes[start..self.position]).map_err(|_| {
            FerrisError::new(ErrorKind::Parse, "JSON number literal is not valid UTF-8")
        })?;

        Ok(JsonValue::Number(JsonNumber::new(repr.to_string())))
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_byte(b'"')?;

        let mut output = Vec::new();

        while let Some(byte) = self.next_byte() {
            match byte {
                b'"' => {
                    return String::from_utf8(output).map_err(|_| {
                        FerrisError::new(ErrorKind::Parse, "JSON string contains invalid UTF-8")
                    });
                }
                b'\\' => {
                    let escaped = self
                        .next_byte()
                        .ok_or_else(|| self.error("unterminated JSON escape sequence"))?;

                    match escaped {
                        b'"' | b'\\' | b'/' => output.push(escaped),
                        b'b' => output.push(0x08),
                        b'f' => output.push(0x0C),
                        b'n' => output.push(b'\n'),
                        b'r' => output.push(b'\r'),
                        b't' => output.push(b'\t'),
                        b'u' => self.push_unicode_escape(&mut output)?,
                        _ => return Err(self.error("invalid JSON escape sequence")),
                    }
                }
                0x00..=0x1F => {
                    return Err(self.error("JSON strings cannot contain control bytes"));
                }
                _ => output.push(byte),
            }
        }

        Err(self.error("unterminated JSON string literal"))
    }

    fn push_unicode_escape(&mut self, output: &mut Vec<u8>) -> Result<()> {
        let first = self.parse_hex_u16()?;

        let codepoint = if (0xD800..=0xDBFF).contains(&first) {
            self.expect_byte(b'\\')?;
            self.expect_byte(b'u')?;

            let second = self.parse_hex_u16()?;
            if !(0xDC00..=0xDFFF).contains(&second) {
                return Err(self.error("invalid JSON unicode surrogate pair"));
            }

            let high = (first as u32) - 0xD800;
            let low = (second as u32) - 0xDC00;
            0x10000 + ((high << 10) | low)
        } else if (0xDC00..=0xDFFF).contains(&first) {
            return Err(self.error("unexpected trailing JSON unicode surrogate"));
        } else {
            first as u32
        };

        let character = char::from_u32(codepoint)
            .ok_or_else(|| self.error("invalid JSON unicode codepoint"))?;
        let mut buffer = [0u8; 4];
        let encoded = character.encode_utf8(&mut buffer);
        output.extend_from_slice(encoded.as_bytes());
        Ok(())
    }

    fn parse_hex_u16(&mut self) -> Result<u16> {
        let mut value = 0u16;

        for _ in 0..4 {
            let byte = self
                .next_byte()
                .ok_or_else(|| self.error("incomplete JSON unicode escape"))?;
            let digit = match byte {
                b'0'..=b'9' => (byte - b'0') as u16,
                b'a'..=b'f' => (byte - b'a' + 10) as u16,
                b'A'..=b'F' => (byte - b'A' + 10) as u16,
                _ => return Err(self.error("invalid JSON unicode escape")),
            };
            value = (value << 4) | digit;
        }

        Ok(value)
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect_byte(b'[')?;
        self.skip_whitespace();

        let mut items = Vec::new();

        if self.peek_byte() == Some(b']') {
            self.position += 1;
            return Ok(JsonValue::Array(items));
        }

        loop {
            items.push(self.parse_value()?);
            self.skip_whitespace();

            match self.next_byte() {
                Some(b',') => self.skip_whitespace(),
                Some(b']') => return Ok(JsonValue::Array(items)),
                _ => return Err(self.error("invalid JSON array delimiter")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect_byte(b'{')?;
        self.skip_whitespace();

        let mut entries = BTreeMap::new();

        if self.peek_byte() == Some(b'}') {
            self.position += 1;
            return Ok(JsonValue::Object(entries));
        }

        loop {
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let value = self.parse_value()?;
            entries.insert(key, value);
            self.skip_whitespace();

            match self.next_byte() {
                Some(b',') => self.skip_whitespace(),
                Some(b'}') => return Ok(JsonValue::Object(entries)),
                _ => return Err(self.error("invalid JSON object delimiter")),
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek_byte(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn next_byte(&mut self) -> Option<u8> {
        let byte = self.peek_byte()?;
        self.position += 1;
        Some(byte)
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        match self.next_byte() {
            Some(actual) if actual == expected => Ok(()),
            _ => Err(self.error(format!(
                "expected JSON byte '{}' at position {}",
                expected as char, self.position
            ))),
        }
    }

    fn expect_bytes(&mut self, expected: &[u8]) -> Result<()> {
        if self.try_expect_bytes(expected) {
            Ok(())
        } else {
            Err(self.error("unexpected JSON literal"))
        }
    }

    fn try_expect_bytes(&mut self, expected: &[u8]) -> bool {
        if self
            .bytes
            .get(self.position..self.position + expected.len())
            == Some(expected)
        {
            self.position += expected.len();
            true
        } else {
            false
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.bytes.len()
    }

    fn error(&self, message: impl Into<String>) -> FerrisError {
        FerrisError::new(ErrorKind::Parse, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_handles_nested_values_and_unicode() {
        let json = r#"{
            "name": "Qwen\u0020\u4F60\u597D",
            "flag": true,
            "number": 1.5e3,
            "items": [null, false, {"value": 7}]
        }"#;

        let root = parse_json(json).unwrap();
        let object = root.as_object().unwrap();

        assert_eq!(object.get("name").unwrap().as_str().unwrap(), "Qwen 你好");
        assert!(object.get("flag").unwrap().as_bool().unwrap());
        assert_eq!(
            object
                .get("number")
                .unwrap()
                .as_number()
                .unwrap()
                .as_f64()
                .unwrap(),
            1500.0
        );
        assert_eq!(
            object.get("items").unwrap().as_array().unwrap()[2]
                .get("value")
                .unwrap()
                .as_number()
                .unwrap()
                .as_u64()
                .unwrap(),
            7
        );
    }

    #[test]
    fn parse_json_rejects_trailing_content() {
        let error = parse_json(r#"{"ok": true} nope"#).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Parse);
    }
}
