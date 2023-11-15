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
    Integer,
    Plus,
    Minus,
    ReturnKeyword,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub source_index: usize,
    pub value: String,
}

fn skip_while<F: Fn(u8) -> bool>(code: &str, start_at: usize, f: F) -> usize {
    let mut index = start_at;
    let bytes = code.as_bytes();
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
    (TokenKind::Equals, "="),
    (TokenKind::VariableKeyword, "var"),
    (TokenKind::Plus, "+"),
    (TokenKind::Minus, "-"),
    (TokenKind::ReturnKeyword, "return"),
];

pub fn tokenize(code: &str) -> Vec<Token> {
    let mut index: usize = 0;
    let mut out: Vec<Token> = Vec::new();

    while index < code.len() {
        index = skip_while(code, index, is_whitespace);

        if index >= code.len() {
            break;
        }

        let byte = code.as_bytes()[index];
        let maybe_literal: Option<&(TokenKind, &str)> = LITERAL_TOKENS.iter().find(|t| {
            let len = t.1.len();
            let end_index = std::cmp::min(code.len(), index + len);
            let chunk = &code[index..end_index];
            return chunk == t.1;
        });

        if let Some(&(kind, value)) = maybe_literal {
            out.push(Token {
                kind: kind,
                source_index: index,
                value: value.to_string(),
            });
            index += value.len();
        } else if byte.is_ascii_digit() {
            let next_index = skip_while(code, index, is_digit_like);
            out.push(Token {
                kind: TokenKind::Integer,
                source_index: index,
                value: code[index..next_index].to_string(),
            });
            index = next_index;
        } else if byte.is_ascii_alphabetic() {
            let next_index = skip_while(code, index, is_identifier_like);
            out.push(Token {
                kind: TokenKind::Identifier,
                source_index: index,
                value: code[index..next_index].to_string(),
            });
            index = next_index;
        } else {
            panic!(
                "Unknown token encountered: '{}'.",
                &code[index..(index + 1)]
            );
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
        let tokens = tokenize(code);
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();

        assert_eq!(
            &[
                TokenKind::VariableKeyword,
                TokenKind::Identifier,
                TokenKind::Colon,
                TokenKind::Identifier,
                TokenKind::Equals,
                TokenKind::Integer,
                TokenKind::Semicolon,
            ],
            kinds.as_slice()
        );
        assert_eq!(7, tokens.len());
    }

    #[test]
    fn should_tokenize_expression() {
        let prog = "6 + 5";

        let tokens = tokenize(prog);

        assert_eq!(3, tokens.len());
        assert_eq!(
            Token {
                kind: TokenKind::Integer,
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
                kind: TokenKind::Integer,
                source_index: 4,
                value: "5".to_string()
            },
            tokens[2]
        );
    }
}
