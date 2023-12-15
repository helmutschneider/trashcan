use crate::{
    ast::{Expression, Statement},
    tokenizer::Token, bytecode::ENTRYPOINT_NAME,
};

use crate::bytecode::Bytecode;
use std::io::Read;
use std::io::Write;

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
        Expression::UnaryPrefix(unary) => {
            let (expr, expr_len) = resolve_expression_range(&unary.expr);
            (unary.operator.source_index, expr_len + unary.operator.value.len())
        }
        Expression::ElementAccess(prop) => {
            let (left, left_len) = resolve_expression_range(&prop.left);
            let (right, right_len) = resolve_expression_range(&prop.right);
            (left, (right - left) + right_len)
        }
        Expression::ArrayLiteral(lit) => {
            let (left, left_len) = resolve_expression_range(lit.elements.first().unwrap());
            let (right, right_len) = resolve_expression_range(lit.elements.last().unwrap());
            (left, (right - left) + right_len)
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

fn type_name<T>(_: &T) -> &str {
    let named = std::any::type_name::<T>();

    // find the last type name after all the '::' namespaces.
    for k in 0..named.len() {
        let index = named.len() - k - 1;
        if &named[index..(index+1)] == ":" {
            return &named[(index + 1)..];
        }
    }

    return named;
}

pub fn format_stmt(stmt: &Statement, indent: i64) -> String {
    let indent_s = "  ".repeat(indent as usize);
    return match stmt {
        Statement::Function(fx) => {
            let arg_s = fx.arguments.iter()
                .map(|x| x.name_token.value.clone())
                .collect::<Vec<String>>().join(", ");

            format!("{}{} {}({})\n{}", indent_s, type_name(fx), fx.name_token.value, arg_s, format_stmt(&fx.body, indent + 1))
        }
        Statement::Variable(var) => {
            format!("{}{}({})\n{}", indent_s, type_name(var), var.name_token.value, format_expr(&var.initializer, indent + 1))
        }
        Statement::Expression(x) => {
            format!("{}{}\n{}", indent_s, type_name(x), format_expr(&x.expr, indent + 1))
        }
        Statement::Return(ret) => {
            format!("{}{}\n{}", indent_s, type_name(ret), format_expr(&ret.expr, indent + 1))
        }
        Statement::Block(block) => {
            let inner = block.statements.iter()
                .map(|s| format_stmt(&s, indent + 1))
                .collect::<Vec<String>>()
                .join("\n");

            format!("{}{}\n{}", indent_s, type_name(block), inner)
        }
        Statement::If(if_) => {
            let mut s = format!("{}{}\n{}\n{}", indent_s, type_name(if_), format_expr(&if_.condition, indent + 1), format_stmt(&if_.block, indent + 1));
            if let Some(x) = if_.else_.as_ref() {
                s.push_str(&format!("{}", format_stmt(x, indent + 1)));
            }
            return s;
        }
        Statement::While(while_) => {
            format!("{}{}\n{}\n{}", indent_s, type_name(while_), format_expr(&while_.condition, indent + 1), format_stmt(&while_.block, indent + 1))
        }
        Statement::Type(type_) => {
            format!("{}{}", indent_s, type_name(type_))
        }
    };
}

fn format_expr(expr: &Expression, indent: i64) -> String {
    let indent_s = "  ".repeat(indent as usize);
    return match expr {
        Expression::Void => "Void".to_string(),
        Expression::Identifier(ident) => {
            format!("{}{}({})", indent_s, type_name(ident), ident.name)
        },
        Expression::IntegerLiteral(x) => {
            format!("{}{}({})", indent_s, type_name(x), x.value)
        }
        Expression::StringLiteral(x) => {
            format!("{}{}(\"{}\")", indent_s, type_name(x), x.value.replace("\n", "\\n"))
        }
        Expression::FunctionCall(fx) => {
            let arg_s = fx.arguments.iter()
                .map(|x| format_expr(x, indent + 1))
                .collect::<Vec<String>>().join("\n");
            format!("{}{}({})\n{}", indent_s, type_name(fx), fx.name_token.value, arg_s)
        }
        Expression::BinaryExpr(bin_expr) => {
            let op_indent = "  ".repeat(indent as usize + 1);
            format!("{}{}\n{}\n{}{}\n{}", indent_s, type_name(bin_expr), format_expr(&bin_expr.left, indent + 1), op_indent, bin_expr.operator.value, format_expr(&bin_expr.right, indent + 1))
        }
        Expression::StructLiteral(struct_) => {
            let member_indent = "  ".repeat(indent as usize + 1);
            let members_s = struct_.members.iter()
                .map(|m| format!("{}{}\n{}", member_indent, m.field_name_token.value, format_expr(&m.value, indent + 2)))
                .collect::<Vec<String>>()
                .join("\n");
            format!("{}{}\n{}", indent_s, type_name(struct_), members_s)
        }
        Expression::MemberAccess(m) => {
            let mem_ident = "  ".repeat(indent as usize + 1);
            format!("{}{}\n{}\n{}{}", indent_s, type_name(m), format_expr(&m.left, indent + 1), mem_ident, m.right.name)
        }
        Expression::BooleanLiteral(b) => {
            format!("{}{}({})", indent_s, type_name(b), b.value)
        }
        Expression::UnaryPrefix(unary) => {
            let op_indent = "  ".repeat(indent as usize + 1);
            format!("{}{}\n{}{}\n{}", indent_s, type_name(unary), op_indent, unary.operator.value, format_expr(&unary.expr, indent + 1))
        }
        Expression::ArrayLiteral(arr) => {
            let element_s = arr.elements.iter()
                .map(|e| format_expr(e, indent + 1))
                .collect::<Vec<String>>()
                .join("\n");
            format!("{}{}\n{}", indent_s, type_name(arr), element_s)
        }
        Expression::ElementAccess(e) => {
            format!("{}{}\n{}\n{}", indent_s, type_name(e), format_expr(&e.left, indent + 1), format_expr(&e.right, indent + 1))
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub enum Arch {
    X86_64,
    ARM64,
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OS {
    Linux,
    MacOS,
}

impl std::fmt::Display for OS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Env {
    pub arch: Arch,
    pub os: OS,
    pub syscall_print: i64,
    pub syscall_exit: i64,
    pub compiler_bin: &'static str,
    pub compiler_args: &'static [&'static str],
    pub backend: fn(&Bytecode, &Self) -> Result<String, Error>,
}

impl Env {
    pub fn current() -> &'static Self {
        let arch = std::env::consts::ARCH.to_lowercase();
        let os = std::env::consts::OS.to_lowercase();
        let unsupported = || {
            panic!("unsupported platform '{}', {}", arch, os);
        };

        return match arch.as_str() {
            "x86_64" => {
                match os.as_str() {
                    "linux" => &LINUX_X86_64,
                    "macos" => &MACOS_X86_64,
                    _ => unsupported(),
                }
            },
            "aarch64" =>  {
                match os.as_str() {
                    "macos" => &MACOS_X86_64,
                    _ => unsupported(),
                }
            },
            _ => unsupported(),
        };
    }

    pub fn emit_binary(&self, out_name: &str, program: &str) -> Result<String, Error> {
        let bc = crate::bytecode::Bytecode::from_code(program)?;
        let asm = (self.backend)(&bc, self)?;
        let compiler_args = [self.compiler_args, &["-o", out_name, "-"]].concat();
    
        let mut child = std::process::Command::new(self.compiler_bin)
            .args(compiler_args)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
    
        let asm_as_string = asm.to_string();
    
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(asm_as_string.as_bytes()).unwrap();
        }
    
        let output = child.wait_with_output().unwrap();
        let str = std::str::from_utf8(&output.stdout).unwrap();

        return Ok(str.to_string());
    }
}

pub const MACOS_X86_64: Env = Env {
    arch: Arch::X86_64,
    os: OS::MacOS,

    // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
    // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
    syscall_print: 0x2000000 + 4,
    syscall_exit: 0x2000000 + 1,
    compiler_bin: "clang",
    compiler_args: &[
        "--target=x86_64-apple-darwin",
        "-masm=intel",
        "-x",
        "assembler",
        "-nostartfiles",
        "-nostdlib",
        "-e",
        ENTRYPOINT_NAME,
    ],
    backend: crate::x64::emit_assembly,
};

pub const MACOS_ARM64: Env = Env {
    arch: Arch::ARM64,
    os: OS::MacOS,

    // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
    // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
    // https://stackoverflow.com/a/56993314
    syscall_print: 0x2000000 + 4,
    syscall_exit: 0x2000000 + 1,
    compiler_bin: "clang",
    compiler_args: &[
        "--target=aarch64-apple-darwin",
        "-x",
        "assembler",
        "-nostartfiles",
        "-nostdlib",
        "-e",
        ENTRYPOINT_NAME,
    ],
    backend: crate::arm64::emit_assembly,
};

pub const LINUX_X86_64: Env = Env {
    arch: Arch::X86_64,
    os: OS::Linux,

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
    backend: crate::x64::emit_assembly,
};

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
        if self.0 < 0 {
            return "-";
        }
        return "+";
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

pub fn random_str(len: usize) -> String {
    let num_bytes = len / 2;
    let mut buf: Vec<u8> = (0..num_bytes).map(|_| 0).collect();
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut buf)
        .unwrap();

    return buf.iter().map(|x| format!("{:x?}", x)).collect::<String>();
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
