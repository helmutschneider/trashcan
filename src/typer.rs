use std::collections::HashMap;

use crate::ast;
use crate::tokenizer::TokenKind;
use crate::util::{Error, report_error, SourceLocation};

#[derive(Debug)]
pub struct Typer {
    ast: ast::AST,
    program: String,
    types: HashMap<TypeId, Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TypeId(i64);

#[derive(Debug)]
struct Type {
    id: TypeId,
    name: String,
    kind: TypeKind,
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
struct Size(pub i64);

#[derive(Debug)]
enum TypeKind {
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

struct ErrorContext<'a> {
    errors: &'a Vec<Error>,
    location: SourceLocation<'a>,
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
            ast::Expression::BinaryExpr(x) => {
                match &x.operator.kind {
                    TokenKind::DoubleEquals => self.types.get(&TYPE_ID_BOOL),
                    TokenKind::NotEquals => self.types.get(&TYPE_ID_BOOL),
                    _ => panic!()
                }
            }
            _ => None,
        };
    }

    fn maybe_report_missing_type(&self, name: &str, type_: Option<&Type>, at: SourceLocation, errors: &mut Vec<Error>) {
        if let None = type_ {
            self.report_error(&format!("cannot find name '{}'", name), at, errors);
        }
    }

    fn maybe_report_type_mismatch(&self, a: Option<&Type>, b: Option<&Type>, at: SourceLocation, errors: &mut Vec<Error>) {
        if let Some(a) = a {
            if let Some(b) = b {
                if a != b {
                    self.report_error(&format!("type '{}' is not assignable to '{}'", a, b), at, errors);
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
            ast::Statement::Variable(x) => {
                let location = SourceLocation::Token(&x.name_token);
                let decl_type = self.get_type_by_name(&x.type_);
                let inferred_type = self.get_inferred_type(&x.initializer);

                self.maybe_report_missing_type(&x.type_, decl_type, location, errors);
                self.maybe_report_type_mismatch(inferred_type, decl_type, location, errors);
            },
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);
                let bool_type = self.types.get(&TYPE_ID_BOOL);
                let inferred_type = self.get_inferred_type(&if_expr.condition);

                self.maybe_report_type_mismatch(inferred_type, bool_type, location, errors);
            }
            ast::Statement::Function(fx) => {
                let return_type = self.get_type_by_name(&fx.return_type);
                self.maybe_report_missing_type(&fx.return_type, return_type, SourceLocation::Token(&fx.return_type_token), errors);

                self.check_block(&fx.body, errors);
            }
            ast::Statement::Block(b) => {
                self.check_block(b, errors);
            }
            ast::Statement::Return(expr) => {
                
            }
            ast::Statement::Expression(expr) => {

            }
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
    fn should_reject_assignment_to_literal() {
        let code = r###"
            var x: string = 1;
        "###;
        let mut chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_assignment_to_function_call() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }

            var x: string = add(1, 2);
        "###;
        let mut chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_if_expr() {
        let code = r###"
        if 1 {
            var x: int = 1;
        }
        "###;
        let mut chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }
}
