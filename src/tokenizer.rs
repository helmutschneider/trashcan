use crate::util::Error;
use crate::util::SourceLocation;
use crate::util::report_error;
use crate::util::snake_case;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Comma,
    Colon,
    Semicolon,
    OpenParenthesis,
    OpenBrace,
    CloseParenthesis,
    CloseBrace,
    FunctionKeyword,
    Identifier,
    Equals,
    VariableKeyword,
    IntegerLiteral,
    StringLiteral,
    Plus,
    Minus,
    Star,
    Slash,
    ReturnKeyword,
    DoubleEquals,
    NotEquals,
    IfKeyword,
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let maybe_literal = LITERAL_TOKENS.iter().find(|x| x.0 == *self);
        let s = match maybe_literal {
            Some((_, s)) => format!("'{}'", s),
            None => snake_case(&format!("{:?}", self)),
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub source_index: usize,
    pub value: String,
}

fn skip_while<F: Fn(u8) -> bool>(source: &str, start_at: usize, f: F) -> usize {
    let bytes = source.as_bytes();
    let mut index = start_at;
    while index < bytes.len() && f(bytes[index]) {
        index += 1;
    }
    return index;
}

const fn is_whitespace(value: u8) -> bool {
    return value.is_ascii_whitespace();
}

const fn is_digit_like(value: u8) -> bool {
    return matches!(value, b'0'..=b'9');
}

const fn is_identifier_like(value: u8) -> bool {
    return matches!(value, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_');
}

const LITERAL_TOKENS: &[(TokenKind, &'static str)] = &[
    (TokenKind::Comma, ","),
    (TokenKind::Colon, ":"),
    (TokenKind::Semicolon, ";"),
    (TokenKind::FunctionKeyword, "fun"),
    (TokenKind::OpenParenthesis, "("),
    (TokenKind::CloseParenthesis, ")"),
    (TokenKind::OpenBrace, "{"),
    (TokenKind::CloseBrace, "}"),
    (TokenKind::DoubleEquals, "=="),
    (TokenKind::NotEquals, "!="),
    (TokenKind::Equals, "="),
    (TokenKind::VariableKeyword, "var"),
    (TokenKind::Plus, "+"),
    (TokenKind::Minus, "-"),
    (TokenKind::Star, "*"),
    (TokenKind::Slash, "/"),
    (TokenKind::ReturnKeyword, "return"),
    (TokenKind::IfKeyword, "if"),
];

fn read_string_literal(source: &str, at_index: usize) -> Result<(&str, usize), Error> {
    let bytes = source.as_bytes();

    assert_eq!(b'"', bytes[at_index]);

    let start_index = at_index + 1;
    let mut is_reading_escaped_char = false;

    for i in start_index..bytes.len() {
        let ch = bytes[i];

        if ch == b'\\' {
            is_reading_escaped_char = true;
        } else if ch == b'"' && !is_reading_escaped_char {
            let start_index = at_index + 1;
            let str_bytes = &bytes[start_index..i];
            let s = match std::str::from_utf8(&str_bytes) {
                Ok(s) => s,
                Err(_) => {
                    return report_error(source, "could not parse string literal as utf-8", SourceLocation::Index(at_index));
                }
            };
            return Result::Ok((s, i + 1));
        } else {
            is_reading_escaped_char = false;
        }
    }

    return report_error(source, "reached end-of-file while reading a quoted string", SourceLocation::Index(at_index));
}

pub fn tokenize(source: &str) -> Result<Vec<Token>, Error> {
    let mut index: usize = 0;
    let mut out: Vec<Token> = Vec::new();

    while index < source.len() {
        index = skip_while(source, index, is_whitespace);

        if index >= source.len() {
            break;
        }

        let byte = source.as_bytes()[index];
        let maybe_literal: Option<&(TokenKind, &str)> = LITERAL_TOKENS.iter().find(|t| {
            let len = t.1.len();
            let end_index = std::cmp::min(source.len(), index + len);
            let chunk = &source[index..end_index];
            return chunk == t.1;
        });

        if let Some(&(kind, value)) = maybe_literal {
            out.push(Token {
                kind: kind,
                source_index: index,
                value: value.to_string(),
            });
            index += value.len();
        } else if byte == b'"' {
            let (s, next_index) = read_string_literal(source, index)?;
            let token = Token {
                kind: TokenKind::StringLiteral,
                source_index: index,
                value: s.to_string(),
            };
            out.push(token);
            index = next_index;
        } else if byte.is_ascii_digit() {
            let next_index = skip_while(source, index, is_digit_like);
            out.push(Token {
                kind: TokenKind::IntegerLiteral,
                source_index: index,
                value: source[index..next_index].to_string(),
            });
            index = next_index;
        } else if byte.is_ascii_alphabetic() {
            let next_index = skip_while(source, index, is_identifier_like);
            out.push(Token {
                kind: TokenKind::Identifier,
                source_index: index,
                value: source[index..next_index].to_string(),
            });
            index = next_index;
        } else {
            let message = format!("unknown token: {}", byte as char);
            return report_error(source, &message, SourceLocation::Index(index));
        }
    }

    return Result::Ok(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_tokenize_function() {
        let code = "fun do_thing(x: int): void {}";
        let tokens = tokenize(code).unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(
            &[
                TokenKind::FunctionKeyword,
                TokenKind::Identifier,
                TokenKind::OpenParenthesis,
                TokenKind::Identifier,
                TokenKind::Colon,
                TokenKind::Identifier,
                TokenKind::CloseParenthesis,
                TokenKind::Colon,
                TokenKind::Identifier,
                TokenKind::OpenBrace,
                TokenKind::CloseBrace,
            ],
            kinds.as_slice()
        );
        assert_eq!(11, tokens.len());
    }

    #[test]
    fn should_tokenize_assignment() {
        let code = "var x: int = 6;";
        let tokens = tokenize(code).unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(
            &[
                TokenKind::VariableKeyword,
                TokenKind::Identifier,
                TokenKind::Colon,
                TokenKind::Identifier,
                TokenKind::Equals,
                TokenKind::IntegerLiteral,
                TokenKind::Semicolon,
            ],
            kinds.as_slice()
        );
        assert_eq!(7, tokens.len());
    }

    #[test]
    fn should_tokenize_expression() {
        let prog = "6 + 5";

        let tokens = tokenize(prog).unwrap();

        assert_eq!(3, tokens.len());
        assert_eq!(
            Token {
                kind: TokenKind::IntegerLiteral,
                source_index: 0,
                value: "6".to_string()
            },
            tokens[0]
        );
        assert_eq!(
            Token {
                kind: TokenKind::Plus,
                source_index: 2,
                value: "+".to_string()
            },
            tokens[1]
        );
        assert_eq!(
            Token {
                kind: TokenKind::IntegerLiteral,
                source_index: 4,
                value: "5".to_string()
            },
            tokens[2]
        );
    }

    #[test]
    fn should_read_string_literal() {
        let code = r###"
            var x: string = "Hello there!";
            var y: string = "Hello again!";
        "###;

        let tokens = tokenize(code).unwrap();

        assert_eq!(TokenKind::StringLiteral, tokens[5].kind);
        assert_eq!("Hello there!", tokens[5].value);
    }
}
