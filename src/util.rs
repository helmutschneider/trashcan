use crate::{
    ast::{Expression, Statement, self, TypeName},
    tokenizer::Token, bytecode::ENTRYPOINT_NAME,
};

pub type Error = String;

#[derive(Debug, Clone, Copy)]
pub enum SourceLocation<'a> {
    None,
    Token(&'a Token),
    Index(usize),
    Expression(&'a Expression),
}

fn resolve_expression_range(expr: &Expression) -> (usize, usize) {
    return match expr {
        Expression::IntegerLiteral(x) => (x.token.source_index, x.token.value.len()),

        // the string token does not include the double quotes.
        Expression::StringLiteral(x) => (x.token.source_index, x.token.value.len() + 2),
        Expression::FunctionCall(fx) => {
            let name_token = &fx.name_token;
            let mut call_len = name_token.value.len();

            if let Some(x) = fx.arguments.last() {
                let (index, len) = resolve_expression_range(x);
                call_len = index - name_token.source_index + len;
            }

            return (name_token.source_index, call_len + 1);
        }
        Expression::BinaryExpr(bin_expr) => {
            let (left_index, _) = resolve_expression_range(&bin_expr.left);
            let (right_index, right_len) = resolve_expression_range(&bin_expr.right);

            return (left_index, (right_index - left_index + right_len));
        }
        Expression::Identifier(ident) => (ident.token.source_index, ident.name.len()),
        Expression::MemberAccess(prop) => {
            let (left, left_len) = resolve_expression_range(&prop.left);

            // add one for the '.' token.
            (left, left_len + 1 + prop.right.token.value.len())
        }
        _ => panic!("cannot resolve source location of {:?}", expr),
    };
}

pub fn report_error<T>(source: &str, message: &str, at: SourceLocation) -> Result<T, Error> {
    let (source_index, source_len) = match at {
        SourceLocation::None => (source.len() - 1, 1),
        SourceLocation::Token(token) => (token.source_index, token.value.len()),
        SourceLocation::Index(index) => (index, 1),
        SourceLocation::Expression(expr) => resolve_expression_range(expr),
    };

    let mut s = String::with_capacity(512);
    s.push_str(&format!("error: {}\n\n", message));
    let (line, col) = find_line_and_column(source, source_index);
    let line_with_error = source.lines().nth(line - 1).unwrap();
    s.push_str(&format!("  {}\n", line_with_error));

    // this is an ascii arrow pointing to the error.
    // color escape codes:
    //   https://askubuntu.com/a/1012016
    s.push_str(&format!(
        "  {}\x1B[31m{}\x1B[0m\n",
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
        let should_add_underscore = bytes
            .get(i + 1)
            .map(|b| b.is_ascii_uppercase())
            .unwrap_or(false);
        out.push(bytes[i].to_ascii_lowercase() as char);

        if should_add_underscore {
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

#[derive(Debug, Clone, Copy)]
pub enum OperatingSystemFamily {
    Linux,
    MacOS,
    Windows,
}

#[derive(Debug, Clone)]
pub struct OperatingSystem {
    pub family: OperatingSystemFamily,
    pub name: &'static str,
    pub syscall_print: i64,
    pub syscall_exit: i64,
    pub compiler_bin: &'static str,
    pub compiler_args: &'static [&'static str],
}

impl OperatingSystem {
    pub const MACOS: Self = Self {
        name: "macos",
        family: OperatingSystemFamily::MacOS,
        // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
        // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
        syscall_print: 0x2000000 + 4,
        syscall_exit: 0x2000000 + 1,
        compiler_bin: "clang",
        compiler_args: &[
            "-arch",
            "x86_64",
            "-masm=intel",
            "-x",
            "assembler",
            "-nostartfiles",
            "-nostdlib",
            "-e",
            ENTRYPOINT_NAME,
        ],
    };

    pub const LINUX: Self = Self {
        name: "linux",
        family: OperatingSystemFamily::Linux,
        // https://filippo.io/linux-syscall-table/
        syscall_print: 1,
        syscall_exit: 60,
        compiler_bin: "gcc",
        compiler_args: &[
            "-masm=intel",
            "-x",
            "assembler",
            "-nostartfiles",
            "-nolibc",
            "-e",
            ENTRYPOINT_NAME,
        ],
    };

    pub fn current() -> &'static Self {
        return Self::from_name(std::env::consts::OS);
    }

    pub fn from_name(name: &str) -> &'static Self {
        return match name {
            "macos" => &Self::MACOS,
            "linux" => &Self::LINUX,
            _ => panic!("unsupported operating system: {}", name),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Offset(pub i64);

impl Offset {
    pub const ZERO: Offset = Offset(0);

    pub fn add<T: Into<Offset>>(&self, other: T) -> Offset {
        let a = self.0;
        let b_offset: Offset = other.into();
        let b = b_offset.0;
        let res = a + b;
        return Offset(res);
    }

    pub fn operator(&self) -> &'static str {
        if self.0 == 0 {
            return "";
        }
        if self.0 > 0 {
            return "+";
        }
        return "-";
    }
}

impl Into<Offset> for i64 {
    fn into(self) -> Offset {
        return Offset(self);
    }
}

impl std::fmt::Display for Offset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self(value) => {
                f.write_str(&format!("{}{}", self.operator(), value.abs()))
            }
        }
    }
}

pub fn with_stdlib(code: &str) -> String {
    const STDLIB_COWABUNGA: &'static str = r###"
fun assert(value: bool): void {
    if value {} else {
        print(&"assertion failed.\n");
        exit(1);
    }
}
    "###;

    let mut code = code.to_string();
    code.push_str(STDLIB_COWABUNGA);

    return code;
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
