#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Comma,
    Colon,
    OpenParenthesis,
    OpenBrace,
    CloseParenthesis,
    CloseBrace,
    Function,
    Identifier,
    Semicolon,
    Equals,
    Variable,
    Number,
    Plus,
    Minus,
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    kind: TokenKind,
    index: usize,
    value: &'a str,
}

fn skip_while<F: Fn(u8) -> bool>(chars: &[u8], start_at: usize, f: F) -> usize {
    let mut index = start_at;
    while index < chars.len() && f(chars[index]) {
        index += 1;
    }
    return index;
}

const fn is_whitespace(value: u8) -> bool {
    return value.is_ascii_whitespace();
}

const fn is_digit_like(value: u8) -> bool {
    return matches!(value, b'0'..=b'9' | b'.');
}

const fn is_identifier_like(value: u8) -> bool {
    return matches!(value, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_');
}

const LITERAL_TOKENS: &[(TokenKind, &'static str)] = &[
    (TokenKind::Function, "fun" ),
    (TokenKind::OpenParenthesis, "(" ),
    (TokenKind::CloseParenthesis, ")" ),
    (TokenKind::Colon, ":" ),
    (TokenKind::OpenBrace, "{" ),
    (TokenKind::CloseBrace, "}" ),
    (TokenKind::Semicolon, ";" ),
    (TokenKind::Equals, "=" ),
    (TokenKind::Variable, "var" ),
    (TokenKind::Plus, "+" ),
    (TokenKind::Minus, "-" ),
    (TokenKind::Return, "return"),
];

pub fn tokenize(code: &str) -> Vec<Token> {
    let mut index: usize = 0;
    let mut out: Vec<Token> = Vec::new();
    let chars = code.as_bytes();

    while index < chars.len() {
        index = skip_while(&chars, index, is_whitespace);

        if index >= chars.len() {
            break;
        }

        let maybe_literal: Option<&(TokenKind, &str)> = LITERAL_TOKENS.iter().find(|t| {
            let len = t.1.len();
            let end_index = std::cmp::min(code.len(), index + len);
            let chunk = &code[index..end_index];
            return chunk == t.1;
        });

        if let Some(&(kind, value)) = maybe_literal {
            out.push(Token {
                kind: kind,
                index: index,
                value: value,
            });
            index += value.len();
        } else if chars[index].is_ascii_digit() {
            let next_index = skip_while(&chars, index, is_digit_like);
            out.push(Token {
                kind: TokenKind::Number,
                index: index,
                value: &code[index..next_index],
            });
            index = next_index;
        } else if chars[index].is_ascii_alphabetic() {
            let next_index = skip_while(&chars, index, is_identifier_like);
            out.push(Token {
                kind: TokenKind::Identifier,
                index: index,
                value: &code[index..next_index],
            });
            index = next_index;
        } else {
            panic!("Unknown token encountered: '{}'.", &code[index..(index + 1)]);
        }
    }

    return out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_tokenize_function() {
        let code = "fun do_thing(x: int): void {}";
        let tokens = tokenize(code);
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(&[
            TokenKind::Function,
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
        ], kinds.as_slice());
        assert_eq!(11, tokens.len());
    }

    #[test]
    fn should_tokenize_assignment() {
        let code = "var x: int = 6;";
        let tokens = tokenize(code);
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(&[
            TokenKind::Variable,
            TokenKind::Identifier,
            TokenKind::Colon,
            TokenKind::Identifier,
            TokenKind::Equals,
            TokenKind::Number,
            TokenKind::Semicolon,
        ], kinds.as_slice());
        assert_eq!(7, tokens.len());
    }

    #[test]
    fn should_tokenize_expression() {
        let prog = "6.5 + 5.3";

        let tokens = tokenize(prog);

        assert_eq!(3, tokens.len());
        assert_eq!(Token { kind: TokenKind::Number, index: 0, value: "6.5" }, tokens[0]);
        assert_eq!(Token { kind: TokenKind::Plus, index: 4, value: "+" }, tokens[1]);
        assert_eq!(Token { kind: TokenKind::Number, index: 6, value: "5.3" }, tokens[2]);
    }
}
