use std::collections::HashMap;

use crate::{ast, tokenizer};
use crate::tokenizer::TokenKind;
use crate::util::{Error, report_error, SourceLocation};

#[derive(Debug)]
pub struct Typer {
    pub ast: ast::AST,
    pub program: String,
    pub types: HashMap<TypeId, Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(i64);

#[derive(Debug)]
pub struct Type {
    pub id: TypeId,
    pub name: String,
    pub kind: TypeKind,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.name.fmt(f);
    }
}

impl std::cmp::PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        return self.id == other.id;
    }
}

const TYPE_ID_BOOL: TypeId = TypeId(1);
const TYPE_ID_BYTE: TypeId = TypeId(2);
const TYPE_ID_INT: TypeId = TypeId(3);
const TYPE_ID_STRING: TypeId = TypeId(4);

#[derive(Debug)]
pub struct Size(pub i64);

#[derive(Debug)]
pub enum TypeKind {
    Scalar(Size),
    Pointer(TypeId),
    Shape(HashMap<String, TypeId>),
}

fn get_builtin_types() -> Vec<Type> {
    let mut types: Vec<Type> = Vec::new();

    types.push(Type {
        id: TYPE_ID_BOOL,
        name: "bool".to_string(),
        kind: TypeKind::Scalar(Size(1)),
    });

    types.push(Type {
        id: TYPE_ID_BYTE,
        name: "byte".to_string(),
        kind: TypeKind::Scalar(Size(1)),
    });

    types.push(Type {
        id: TYPE_ID_INT,
        name: "int".to_string(),
        kind: TypeKind::Scalar(Size(8)),
    });

    let mut str_shape: HashMap<String, TypeId> = HashMap::new();
    str_shape.insert("length".to_string(), TYPE_ID_INT);

    types.push(Type {
        id: TYPE_ID_STRING,
        name: "string".to_string(),
        kind: TypeKind::Shape(str_shape),
    });

    return types;
}

impl Typer {
    fn get_symbol(&self, name: &str, kind: ast::SymbolKind) -> Option<&ast::Symbol> {
        return self.ast.get_symbol(name, kind);
    }

    fn get_function(&self, name: &str) -> Option<&ast::Function> {
        let sym = self.get_symbol(name, ast::SymbolKind::Function)?;
        return match sym.declared_at.as_ref() {
            ast::Statement::Function(fx) => Some(fx),
            _ => None,
        }
    }

    fn get_type_by_name(&self, name: &str) -> Option<&Type> {
        for (_, t) in &self.types {
            if t.name == name {
                return Some(t);
            }
        }
        return None;
    }

    fn get_inferred_type(&self, expr: &ast::Expression) -> Option<&Type> {
        return match expr {
            ast::Expression::IntegerLiteral(_) => self.types.get(&TYPE_ID_INT),
            ast::Expression::StringLiteral(_) => self.types.get(&TYPE_ID_STRING),
            ast::Expression::FunctionCall(fx_call) => {
                let fx = self.get_function(&fx_call.name)?;
                return self.get_type_by_name(&fx.return_type);
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                match &bin_expr.operator.kind {
                    TokenKind::DoubleEquals => self.types.get(&TYPE_ID_BOOL),
                    TokenKind::NotEquals => self.types.get(&TYPE_ID_BOOL),
                    TokenKind::Plus | TokenKind::Minus => self.get_inferred_type(&bin_expr.left),
                    _ => panic!()
                }
            }
            _ => None,
        };
    }

    fn maybe_report_missing_type<T>(&self, name: &str, type_: Option<T>, at: SourceLocation, errors: &mut Vec<Error>) {
        if let None = type_ {
            self.report_error(&format!("cannot find name '{}'", name), at, errors);
        }
    }

    fn maybe_report_type_mismatch(&self, given_type: Option<&Type>, declared_type: Option<&Type>, at: SourceLocation, errors: &mut Vec<Error>) {
        if let Some(given_type) = given_type {
            if let Some(declared_type) = declared_type {
                if given_type != declared_type {
                    self.report_error(&format!("type '{}' has no overlap with '{}'", given_type, declared_type), at, errors);
                }
            }
        }
    }

    fn report_error(&self, message: &str, at: SourceLocation, errors: &mut Vec<Error>) {
        let err = report_error::<()>(&self.program, message, at);
        let message = match err {
            Ok(_) => panic!(),
            Err(e) => e
        };

        errors.push(message);
    }
    
    fn check_statement(&self, stmt: &ast::Statement, errors: &mut Vec<Error>) {
        match stmt {
            ast::Statement::Variable(var) => {
                let location = SourceLocation::Token(&var.name_token);
                let declared_type = self.get_type_by_name(&var.type_);
                let given_type = self.get_inferred_type(&var.initializer);

                self.maybe_report_missing_type(&var.type_, declared_type, location, errors);
                self.maybe_report_type_mismatch(given_type, declared_type, location, errors);
                self.check_expression(&var.initializer, errors);
            },
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);
                let bool_type = self.types.get(&TYPE_ID_BOOL);
                let given_type = self.get_inferred_type(&if_expr.condition);

                self.maybe_report_type_mismatch(given_type, bool_type, location, errors);
                self.check_block(&if_expr.block, errors);
            }
            ast::Statement::Function(fx) => {
                for fx_arg in &fx.arguments {
                    let location = SourceLocation::Token(&fx_arg.type_token);
                    let declared_type = self.get_type_by_name(&fx_arg.type_);
                    self.maybe_report_missing_type(&fx_arg.type_, declared_type, location, errors);
                }

                let return_type = self.get_type_by_name(&fx.return_type);
                self.maybe_report_missing_type(&fx.return_type, return_type, SourceLocation::Token(&fx.return_type_token), errors);

                self.check_block(&fx.body, errors);
            }
            ast::Statement::Block(b) => {
                self.check_block(b, errors);
            }
            ast::Statement::Return(ret) => {
                self.check_expression(&ret.expr, errors);
            }
            ast::Statement::Expression(expr) => {
                self.check_expression(expr, errors);
            }
        }
    }

    fn check_expression(&self, expr: &ast::Expression, errors: &mut Vec<Error>) {
        match expr {
            ast::Expression::FunctionCall(fx_call) => {
                let location = SourceLocation::Token(&fx_call.name_token);
                let fx_sym = self.get_function(&fx_call.name);
                self.maybe_report_missing_type(&fx_call.name, fx_sym, location, errors);

                if let Some(fx_sym) = fx_sym {
                    let expected_len = fx_sym.arguments.len();
                    let given_len = fx_call.arguments.len();

                    if expected_len != given_len {
                        self.report_error(&format!("expected {} arguments, but got {}", expected_len, given_len), location, errors);
                    } else {
                        for i in 0..expected_len {
                            let fx_arg = &fx_sym.arguments[i];
                            let call_arg = &fx_call.arguments[i];
                            let declared_type = self.get_type_by_name(&fx_arg.type_);
                            let given_type = self.get_inferred_type(&call_arg);

                            let location = SourceLocation::Expression(call_arg);
                            self.maybe_report_type_mismatch(given_type, declared_type, location, errors)
                        }
                    }
                }
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let location = SourceLocation::Token(&bin_expr.operator);
                let left = self.get_inferred_type(&bin_expr.left);
                let right = self.get_inferred_type(&bin_expr.right);

                self.maybe_report_type_mismatch(left, right, location, errors);
            }
            _ => {}
        }
    }

    fn check_block(&self, block: &ast::Block, errors: &mut Vec<Error>) {
        for stmt in &block.statements {
            self.check_statement(stmt, errors);
        }
    }

    pub fn from_code(program: &str) -> Result<Self, Error> {
        let ast = ast::AST::from_code(program)?;
        let mut chk = Self {
            ast: ast,
            program: program.to_string(),
            types: HashMap::new(),
        };

        for t in get_builtin_types() {
            chk.types.insert(t.id, t);
        }

        return Result::Ok(chk);
    }

    fn check_with_errors(&self, errors: &mut Vec<Error>) -> Result<(), ()> {
        self.check_block(&self.ast.body, errors);

        if errors.is_empty() {
            return Ok(());
        }
        return Err(());
    }

    pub fn check(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        self.check_with_errors(&mut errors);

        if errors.is_empty() {
            return Ok(());
        }

        let err = errors.join("\n");
        return Err(err);
    }
}

#[cfg(test)]
mod tests {
    use crate::typer::Typer;

    #[test]
    fn should_reject_type_mismatch_with_literal() {
        let code = r###"
            var x: string = 1;
        "###;
        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_type_mismatch_with_function_call() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }

            var x: string = add(1, 2);
        "###;
        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_if_expr() {
        let code = r###"
        if 1 {
        }
        "###;
        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_function_call_with_wrong_argument_type() {
        let code = r###"
        fun identity(x: int): int {
            return x;
        }
        identity("hello!");
        "###;
        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_function_call_with_wrong_number_of_arguments() {
        let code = r###"
        fun identity(x: int): int {
            return x;
        }
        identity();
        "###;
        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }
}
