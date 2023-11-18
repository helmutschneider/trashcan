use crate::tokenizer::Token;

pub type Error = String;

pub enum ErrorLocation<'a> {
    EOF,
    Token(&'a Token),
    Index(usize),
}

pub fn report_error<T>(source: &str, message: &str, at: ErrorLocation) -> Result<T, Error> {
    let (source_index, source_len) = match at {
        ErrorLocation::EOF => (source.len() - 1, 1),
        ErrorLocation::Token(token) => (token.source_index, token.value.len()),
        ErrorLocation::Index(index) => (index, 1),
    };

    let mut s = String::with_capacity(512);
    s.push_str(&format!("error: {}\n\n", message));
    let (line, col) = find_line_and_column(source, source_index);
    let line_with_error = source.lines().nth(line - 1).unwrap();
    s.push_str(&format!("  {}\n", line_with_error));

    // this is an ascii arrow pointing to the error.
    s.push_str(&format!(
        "  {}{}",
        " ".repeat(col - 1),
        "^".repeat(source_len)
    ));

    eprintln!("{s}");

    return Result::Err(s);
}

pub fn snake_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 4);
    let bytes = value.as_bytes();

    for i in 0..bytes.len() {
        let should_add_unserscore = bytes.get(i + 1).map(|b| b.is_ascii_uppercase()).unwrap_or(false);
        out.push(bytes[i].to_ascii_lowercase() as char);

        if should_add_unserscore {
            out.push('_');
        }
    }
    
    return out;
}

pub fn find_line_and_column(code: &str, char_index: usize) -> (usize, usize) {
    let mut line: usize = 1;
    let mut column: usize = 1;
    let bytes = code.as_bytes();

    for i in 0..bytes.len() {
        if i == char_index {
            break;
        }
        if bytes[i] == b'\n' {
            line += 1;
            column = 0;
        }
        column += 1;
    }

    return (line, column);
}

#[cfg(test)]
mod tests {
    use crate::util::snake_case;

    #[test]
    fn should_snake_case() {
        assert_eq!("i_am_snaked", snake_case("IAmSnaked"));
        assert_eq!("the_big_snaker", snake_case("the_bigSnaker"));
    }
}