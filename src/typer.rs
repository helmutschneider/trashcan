use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;

use crate::ast::{ASTLike, Expression, Function, Statement, StatementIndex};
use crate::tokenizer::TokenKind;
use crate::util::{report_error, Error, SourceLocation};
use crate::{ast, tokenizer};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Type {
    pub name: String,
    pub definition: TypeDefinition,
}

impl std::cmp::PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        if let TypeDefinition::Scalar(size_a) = self.definition {
            if let TypeDefinition::Scalar(size_b) = other.definition {
                return size_a == size_b;
            }
        }

        if let TypeDefinition::Pointer(inner_a) = &self.definition {
            if let TypeDefinition::Pointer(inner_b) = &other.definition {
                return inner_a == inner_b;
            }
        }

        if let TypeDefinition::Struct(inner_a) = &self.definition {
            if let TypeDefinition::Struct(inner_b) = &other.definition {
                let keys_a: Vec<&String> = inner_a.keys().collect();
                let keys_b: Vec<&String> = inner_a.keys().collect();

                if keys_a.len() != keys_b.len() {
                    return false;
                }

                for key in keys_a {
                    if inner_a.get(key) != inner_b.get(key) {
                        return false;
                    }
                }

                return true;
            }
        }

        // TODO: function types.

        return false;
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match &self.definition {
            TypeDefinition::Scalar(_) => self.name.clone(),
            TypeDefinition::Pointer(to_type) => format!("&{}", to_type),
            TypeDefinition::Struct(_) => self.name.clone(),
            TypeDefinition::Function(arg_types, ret_type) => {
                let arg_s = arg_types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("fun ({}): {}", arg_s, ret_type)
            }
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Scalar(i64),
    Pointer(Rc<Type>),
    Struct(HashMap<String, Rc<Type>>),
    Function(Vec<Rc<Type>>, Rc<Type>),
}

#[derive(Debug, Clone)]
pub struct TypeTable {
    types: Vec<Rc<Type>>,
}

impl TypeTable {
    fn new() -> Self {
        let mut types = TypeTable {
            types: Vec::new(),
        };

        types.add_type("void", TypeDefinition::Scalar(0));
        types.add_type("bool", TypeDefinition::Scalar(1));
        types.add_type("byte", TypeDefinition::Scalar(1));
        types.add_type("int", TypeDefinition::Scalar(8));

        let type_ptr_to_void = types.pointer_to(types.void());
        let type_string = types.add_struct_type(
            "string",
            &[("length", types.int()), ("data", type_ptr_to_void)],
        );

        return types;
    }

    pub fn void(&self) -> Rc<Type> {
        return self.get_type_by_name("void").unwrap();
    }

    pub fn bool(&self) -> Rc<Type> {
        return self.get_type_by_name("bool").unwrap();
    }

    pub fn byte(&self) -> Rc<Type> {
        return self.get_type_by_name("byte").unwrap();
    }

    pub fn int(&self) -> Rc<Type> {
        return self.get_type_by_name("int").unwrap();
    }

    pub fn string(&self) -> Rc<Type> {
        return self.get_type_by_name("string").unwrap();
    }

    pub fn pointer_to(&self, type_: Rc<Type>) -> Rc<Type> {
        return Rc::new(Type {
            name: "".to_string(),
            definition: TypeDefinition::Pointer(type_),
        });
    }

    fn add_type(&mut self, name: &str, defn: TypeDefinition) -> Rc<Type> {
        let type_ = Rc::new(Type {
            name: name.to_string(),
            definition: defn,
        });
        let index: usize = self.types.len();
        self.types.push(type_);
        return Rc::clone(&self.types[index]);
    }

    fn add_struct_type(&mut self, name: &str, fields: &[(&str, Rc<Type>)]) -> Rc<Type> {
        if name.is_empty() {
            panic!("struct types must be named.");
        }

        let mut map: HashMap<String, Rc<Type>> = HashMap::new();

        for (name, type_) in fields {
            map.insert(name.to_string(), Rc::clone(type_));
        }

        return self.add_type(name, TypeDefinition::Struct(map));
    }

    fn add_function_type(&mut self, arguments: &[Rc<Type>], return_type: Rc<Type>) -> Rc<Type> {
        let args: Vec<Rc<Type>> = arguments.iter().map(|t| Rc::clone(t)).collect();
        let defn = TypeDefinition::Function(args, return_type);
        let type_ = Rc::new(Type {
            name: "".to_string(),
            definition: defn,
        });
        let index: usize = self.types.len();
        self.types.push(type_);
        return Rc::clone(&self.types[index]);
    }

    fn get_type_by_name(&self, name: &str) -> Option<Rc<Type>> {
        return self.types.iter()
            .find(|t| t.name == name)
            .map(|t| Rc::clone(t));
    }

    fn try_resolve_type(&self, decl: &ast::TypeDeclaration) -> Result<Rc<Type>, tokenizer::Token> {
        return match decl {
            ast::TypeDeclaration::Name(tok) => {
                let found = self.get_type_by_name(&tok.value);
                
                match found {
                    Some(t) => Ok(t),
                    None => Err(tok.clone()),
                }
            },
            ast::TypeDeclaration::Pointer(inner_defn) => {
                let inner_type = self.try_resolve_type(inner_defn)?;
                let type_ = Rc::new(Type {
                    name: format!("&{}", inner_type),
                    definition: TypeDefinition::Pointer(inner_type),
                });
                return Ok(type_);
            }
        };
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Local,
    Function,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub type_: Option<Rc<Type>>,
    pub declared_at: StatementIndex,
    pub scope: StatementIndex,
}

fn create_symbols_at_statement(
    ast: &ast::AST,
    symbols: &mut Vec<Symbol>,
    types: &mut TypeTable,
    scope: StatementIndex
) {
    let block = match ast.get_statement(scope) {
        Statement::Block(b) => b,
        _ => {
            return;
        }
    };

    for child_index in &block.statements {
        let stmt = ast.get_statement(*child_index);

        match stmt {
            Statement::Function(fx) => {
                let mut fx_arg_types: Vec<Rc<Type>> = Vec::new();

                // create symbols in the function body scopes for its locals.
                for arg in &fx.arguments {
                    let arg_type = types.try_resolve_type(&arg.type_);
                    let arg_sym = Symbol {
                        name: arg.name_token.value.clone(),
                        kind: SymbolKind::Local,
                        type_: arg_type.as_ref().map(|t| Rc::clone(t)).ok(),
                        declared_at: *child_index,
                        scope: fx.body,
                    };
                    symbols.push(arg_sym);

                    if let Ok(t) = arg_type.as_ref() {
                        fx_arg_types.push(Rc::clone(t));
                    }
                }

                let fx_type: Option<Rc<Type>> = {
                    if fx_arg_types.len() == fx.arguments.len() {
                        let maybe_ret_type = types.try_resolve_type(&fx.return_type);
                        if let Ok(ret_type) = maybe_ret_type {
                            let fx_type = types.add_function_type(&fx_arg_types, ret_type);
                            Some(fx_type)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                let fn_sym = Symbol {
                    name: fx.name_token.value.clone(),
                    kind: SymbolKind::Function,
                    type_: fx_type,
                    declared_at: *child_index,
                    scope: scope,
                };
                symbols.push(fn_sym);

                create_symbols_at_statement(ast, symbols, types, fx.body);
            }
            Statement::Block(_) => {
                create_symbols_at_statement(ast, symbols, types, *child_index);
            }
            Statement::If(if_expr) => {
                create_symbols_at_statement(ast, symbols, types, if_expr.block);
            }
            Statement::Variable(v) => {
                let type_ = v.type_.as_ref()
                    .and_then(|d| types.try_resolve_type(&d).ok());
                let sym = Symbol {
                    name: v.name_token.value.clone(),
                    kind: SymbolKind::Local,
                    type_: type_,
                    declared_at: *child_index,
                    scope: scope,
                };
                symbols.push(sym);
            }
            _ => {}
        }
    }
}

fn get_parent_of_expression(ast: &ast::AST, expr: &ast::Expression) -> StatementIndex {
    return match expr {
        Expression::Identifier(ident) => ident.parent,
        Expression::FunctionCall(fn_call) => fn_call.parent,
        Expression::BinaryExpr(bin_expr) => bin_expr.parent,
        _ => panic!("cannot find parent of '{:?}'", expr),
    };
}

fn get_parent_of_statement(ast: &ast::AST, stmt: &ast::Statement) -> Option<StatementIndex> {
    return match stmt {
        Statement::Function(fx) => Some(fx.parent),
        Statement::Variable(v) => Some(v.parent),
        Statement::Expression(expr) => Some(get_parent_of_expression(ast, expr)),
        Statement::Return(ret) => Some(ret.parent),
        Statement::Block(b) => b.parent,
        Statement::If(if_expr) => Some(if_expr.parent),
    };
}

enum WalkResult<T> {
    Stop,
    ToParent,
    Ok(T),
}

fn walk_up_ast_from_statement<T, F: Fn(&ast::Statement, StatementIndex) -> WalkResult<T>>(
    ast: &ast::AST,
    at: StatementIndex,
    fx: F,
) -> Option<T> {
    let mut maybe_index = Some(at);

    while let Some(i) = maybe_index {
        let stmt = ast.get_statement(i);
        let res = fx(stmt, i);

        match res {
            WalkResult::Ok(x) => {
                return Some(x);
            }
            WalkResult::Stop => {
                return None;
            }
            WalkResult::ToParent => {}
        }

        maybe_index = get_parent_of_statement(ast, stmt);
    }

    return None;
}

fn get_enclosing_function(ast: &ast::AST, index: StatementIndex) -> Option<&ast::Function> {
    let found = walk_up_ast_from_statement(ast, index, |stmt, i| {
        if let Statement::Function(_) = stmt {
            return WalkResult::Ok(i);
        }
        return WalkResult::ToParent;
    });

    let stmt = found.map(|i| ast.get_statement(i));

    return match stmt {
        Some(Statement::Function(fx)) => Some(fx),
        _ => None,
    };
}

#[derive(Debug, Clone)]
pub struct Typer {
    pub ast: ast::AST,
    pub program: String,
    pub symbols: Vec<Symbol>,
    pub types: TypeTable,
}

impl Typer {
    pub fn try_resolve_symbol(
        &self,
        name: &str,
        kind: SymbolKind,
        at: StatementIndex,
    ) -> Option<Symbol> {
        let found = walk_up_ast_from_statement(&self.ast, at, |stmt, i| {
            if let Statement::Block(_) = stmt {
                let found = self.symbols.iter().find(|t| {
                    return t.name == name
                        && t.kind == kind
                        && t.scope == i;
                });
                if let Some(s) = found {
                    return WalkResult::Ok(s.clone());
                }
            }

            // let's not resolve locals outside the function scope.
            if let Statement::Function(_) = stmt {
                if kind == SymbolKind::Local {
                    return WalkResult::Stop;
                }
            }

            return WalkResult::ToParent;
        });

        return found;
    }

    pub fn try_resolve_symbol_type(&self, symbol: &Symbol) -> Option<Rc<Type>> {
        if let Some(type_) = &symbol.type_ {
            return Some(Rc::clone(type_));
        }

        if symbol.kind == SymbolKind::Local {
            // locals might not have a declared type, so let's infer the type.
            if let Statement::Variable(var) = self.ast.get_statement(symbol.declared_at) {
                return self.try_infer_type(&var.initializer);
            }
        }

        return None;
    }

    pub fn try_infer_type(&self, expr: &ast::Expression) -> Option<Rc<Type>> {
        return match expr {
            ast::Expression::IntegerLiteral(_) => Some(self.types.int()),
            ast::Expression::StringLiteral(_) => Some(self.types.string()),
            ast::Expression::FunctionCall(fx_call) => {
                let fx_sym = self.try_resolve_symbol(
                    &fx_call.name_token.value,
                    SymbolKind::Function,
                    fx_call.parent,
                )?;
                let fx_type = fx_sym.type_.as_ref()?;
                if let TypeDefinition::Function(_, ret_type) = &fx_type.definition {
                    return Some(Rc::clone(&ret_type));
                }
                return None;
            }
            ast::Expression::BinaryExpr(bin_expr) => match &bin_expr.operator.kind {
                TokenKind::DoubleEquals => Some(self.types.bool()),
                TokenKind::NotEquals => Some(self.types.bool()),
                TokenKind::Plus | TokenKind::Minus => {
                    let left_type = self.try_infer_type(&bin_expr.left);
                    let right_type = self.try_infer_type(&bin_expr.right);

                    if left_type.is_some() && right_type.is_some() && left_type == right_type {
                        return left_type;
                    }

                    return None;
                }
                _ => panic!(
                    "could not infer type for operator '{}'",
                    bin_expr.operator.kind
                ),
            },
            ast::Expression::Identifier(ident) => {
                let maybe_sym = self.try_resolve_symbol(&ident.name, SymbolKind::Local, ident.parent);
                
                return match maybe_sym {
                    Some(s) => self.try_resolve_symbol_type(&s),
                    None => None,
                };
            }
            ast::Expression::Void => Some(self.types.void()),
            ast::Expression::PointerExpr(to_expr) => {
                if let Some(type_) = self.try_infer_type(&to_expr) {
                    let ptr_type = self.types.pointer_to(type_);
                    return Some(ptr_type);
                }
                return None;
            }
        };
    }

    fn maybe_report_missing_type<T>(
        &self,
        name: &str,
        type_: &Option<T>,
        at: SourceLocation,
        errors: &mut Vec<Error>,
    ) {
        if let None = type_ {
            self.report_error(&format!("cannot find name '{}'", name), at, errors);
        }
    }

    fn maybe_report_type_mismatch(
        &self,
        given_type: &Option<Rc<Type>>,
        expected_type: &Option<Rc<Type>>,
        at: SourceLocation,
        errors: &mut Vec<Error>,
    ) {
        if let Some(given_type) = given_type {
            if let Some(expected_type) = expected_type {
                if given_type != expected_type {
                    self.report_error(
                        &format!(
                            "expected type '{}', but got '{}'",
                            expected_type, given_type
                        ),
                        at,
                        errors,
                    );
                }
            }
        }
    }

    fn maybe_report_no_type_overlap(
        &self,
        given_type: Option<Rc<Type>>,
        expected_type: Option<Rc<Type>>,
        at: SourceLocation,
        errors: &mut Vec<Error>,
    ) {
        if let Some(given_type) = given_type {
            if let Some(expected_type) = expected_type {
                if given_type != expected_type {
                    self.report_error(
                        &format!(
                            "type '{}' has no overlap with '{}'",
                            expected_type, given_type
                        ),
                        at,
                        errors,
                    );
                }
            }
        }
    }

    fn report_error(&self, message: &str, at: SourceLocation, errors: &mut Vec<Error>) {
        let err = report_error::<()>(&self.program, message, at);
        let message = match err {
            Ok(_) => panic!(),
            Err(e) => e,
        };

        errors.push(message);
    }

    fn check_type_declaration(&self, decl: &ast::TypeDeclaration, errors: &mut Vec<Error>) -> Option<Rc<Type>> {
        let declared_type = self.types.try_resolve_type(decl);

        return match declared_type {
            Ok(t) => Some(t),
            Err(tok) => {
                self.maybe_report_missing_type::<()>(
                    &tok.value,
                    &None,
                    SourceLocation::Token(&tok),
                    errors,
                );
                None
            }
        };
    }

    fn check_statement(&self, stmt: &ast::Statement, errors: &mut Vec<Error>) {
        match stmt {
            ast::Statement::Variable(var) => {
                let location = SourceLocation::Token(&var.name_token);

                if let Some(type_decl) = &var.type_ {
                    if let Some(type_) = self.check_type_declaration(type_decl, errors) {
                        let given_type = self.try_infer_type(&var.initializer);
                        self.maybe_report_type_mismatch(&given_type, &Some(type_), location, errors);
                    }
                }

                self.check_expression(&var.initializer, errors);
            }
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);

                self.check_expression(&if_expr.condition, errors);
                let bool_type = self.types.bool();
                let given_type = self.try_infer_type(&if_expr.condition);

                self.maybe_report_type_mismatch(&given_type, &Some(bool_type), location, errors);

                let if_block = self.ast.get_block(if_expr.block);
                self.check_block(if_block, errors);
            }
            ast::Statement::Function(fx) => {
                for fx_arg in &fx.arguments {
                    if let Ok(type_) = self.types.try_resolve_type(&fx_arg.type_) {
                        if let TypeDefinition::Struct(_) = type_.definition {
                            let loc = SourceLocation::Token(&fx_arg.name_token);
                            self.report_error(&format!("argument '{}' is a struct type and must be passed by reference", fx_arg.name_token.value), loc, errors);
                        }
                    }

                    self.check_type_declaration(&fx_arg.type_, errors);
                }

                let fx_body = self.ast.get_block(fx.body);

                if let Some(ret_type) = self.check_type_declaration(&fx.return_type, errors) {
                    let is_void_return = ret_type == self.types.void();

                    if !is_void_return {
                        let has_return_statement = fx_body.statements.iter().any(|k| {
                            let stmt = self.ast.get_statement(*k);
                            return matches!(stmt, Statement::Return(_));
                        });
    
                        if !has_return_statement {
                            // TODO: this should probably point to the declared return type,
                            //   but we don't have any way of locating a type declaration yet.
                            let location = SourceLocation::Token(&fx.name_token);
                            self.report_error(
                                "missing 'return' statement",
                                location,
                                errors,
                            );
                        }
                    }
                }

                self.check_block(fx_body, errors);
            }
            ast::Statement::Block(b) => {
                self.check_block(b, errors);
            }
            ast::Statement::Return(ret) => {
                if let Some(fx) = get_enclosing_function(&self.ast, ret.parent) {
                    let return_type = self.types.try_resolve_type(&fx.return_type).ok();
                    let given_type = self.try_infer_type(&ret.expr);
                    let location = SourceLocation::Expression(&ret.expr);
                    self.maybe_report_type_mismatch(&given_type, &return_type, location, errors);
                    self.check_expression(&ret.expr, errors);
                } else {
                    let location = SourceLocation::Token(&ret.token);
                    self.report_error(
                        "a 'return' statement can only be used within a function body",
                        location,
                        errors,
                    )
                }
            }
            ast::Statement::Expression(expr) => {
                self.check_expression(expr, errors);
            }
        }
    }

    fn check_expression(&self, expr: &ast::Expression, errors: &mut Vec<Error>) {
        match expr {
            ast::Expression::FunctionCall(fx_call) => {
                let ident_location = SourceLocation::Token(&fx_call.name_token);
                let maybe_fx_sym = self.try_resolve_symbol(
                    &fx_call.name_token.value,
                    SymbolKind::Function,
                    fx_call.parent,
                );
                self.maybe_report_missing_type(
                    &fx_call.name_token.value,
                    &maybe_fx_sym,
                    ident_location,
                    errors,
                );

                if maybe_fx_sym.is_none() {
                    return;
                }

                let fx_sym = maybe_fx_sym.unwrap();
                let maybe_fx_type = fx_sym.type_.as_ref();

                if maybe_fx_type.is_none() {
                    return;
                }

                let fx_type = maybe_fx_type.unwrap();

                if let TypeDefinition::Function(arg_types, _) = &fx_type.definition {
                    let expected_len = arg_types.len();
                    let given_len = fx_call.arguments.len();

                    if expected_len != given_len {
                        let call_location = SourceLocation::Expression(expr);
                        self.report_error(
                            &format!("expected {} arguments, but got {}", expected_len, given_len),
                            call_location,
                            errors,
                        );
                    } else {
                        for i in 0..expected_len {
                            let call_arg = &fx_call.arguments[i];
                            self.check_expression(call_arg, errors);

                            let declared_type = &arg_types[i];
                            let given_type = self.try_infer_type(&call_arg);

                            let call_arg_location = SourceLocation::Expression(call_arg);
                            self.maybe_report_type_mismatch(
                                &given_type,
                                &Some(Rc::clone(declared_type)),
                                call_arg_location,
                                errors,
                            )
                        }
                    }
                }
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                self.check_expression(&bin_expr.left, errors);
                self.check_expression(&bin_expr.right, errors);

                let location = SourceLocation::Expression(expr);
                let left = self.try_infer_type(&bin_expr.left);
                let right = self.try_infer_type(&bin_expr.right);

                self.maybe_report_no_type_overlap(right, left, location, errors);
            }
            ast::Expression::Identifier(ident) => {
                let location = SourceLocation::Token(&ident.token);
                let ident_sym = self.try_resolve_symbol(&ident.name, SymbolKind::Local, ident.parent);
                self.maybe_report_missing_type(&ident.name, &ident_sym, location, errors);
            }
            _ => {}
        }
    }

    fn check_block(&self, block: &ast::Block, errors: &mut Vec<Error>) {
        for stmt in &block.statements {
            let stmt = self.ast.get_statement(*stmt);
            self.check_statement(stmt, errors);
        }
    }

    pub fn from_code(program: &str) -> Result<Self, Error> {
        let ast = ast::AST::from_code(program)?;
        let body_index = ast.body_index;
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut types = TypeTable::new();

        let type_ptr_to_string = types.pointer_to(types.string());

        // the built-in print function.
        let type_print = types.add_function_type(&[type_ptr_to_string], types.void());
        let print_sym = Symbol {
            name: "print".to_string(),
            kind: SymbolKind::Function,
            type_: Some(type_print),
            declared_at: body_index,
            scope: body_index,
        };
        symbols.push(print_sym);

        create_symbols_at_statement(&ast, &mut symbols, &mut types, body_index);

        let typer = Self {
            ast: ast,
            program: program.to_string(),
            symbols: symbols,
            types: types,
        };

        return Result::Ok(typer);
    }

    fn check_with_errors(&self, errors: &mut Vec<Error>) -> Result<(), ()> {
        self.check_block(&self.ast.body(), errors);

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
    use crate::typer::{Typer, SymbolKind, Symbol};
    use crate::ast::ASTLike;

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

    #[test]
    fn should_fail_to_resolve_symbol() {
        let code = r###"
            fun ident(): int {
                return x;
            }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_incompatible_binary_expression() {
        let code = r###"
        var x: int = 1 + "yee!";
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_bad_return_type() {
        let code = r###"
            fun ident(): int {
                return "yee!";
            }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_identifier_of_wrong_type() {
        let code = r###"
        var x: int = 5;
        var y: string = x;
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_bad_expression_in_function_call() {
        let code = r###"
        fun ident(x: int): int {
            return x;
        }
        ident(1 + "yee!");
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_function_with_declared_return_type_without_return_statement() {
        let code = r###"
        fun ident(x: int): int {
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_bad_if_condition() {
        let code = r###"
            if 5 == "cowabunga!" {

            }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_resolve_functions_from_outer_scopes() {
        let code = r###"
            fun noop(): void {
                return;
            }
            fun main(): int {
                noop();
                return 0;
            }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_not_resolve_variables_outside_function_scope() {
        let code = r###"
        var x: int = 5;

        fun main(): int {
            var y: int = x;
            return 0;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_accept_void_return_type_without_return_statement() {
        let code = r###"
        fun main(): void {}
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_infer_variable_types() {
        let code = r###"
        fun ident(n: int): int {
            return n;
        }
        fun main(): void {
            var x = 1;
            var y = ident(x);
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_create_symbol_table() {
        let code = r###"
            fun add(x: int, y: int): int {
                var z: int = 6;
                return x + y;
            }
        "###;
        let typer = Typer::from_code(code).unwrap();
        let body_syms: Vec<&Symbol> = typer.symbols.iter()
            .filter(|s| s.scope == typer.ast.body_index)
            .collect();

        assert_eq!(2, body_syms.len());
        assert_eq!("add", body_syms[0].name);
        assert_eq!(SymbolKind::Function, body_syms[0].kind);

        assert_eq!("print", body_syms[1].name);
        assert_eq!(SymbolKind::Function, body_syms[1].kind);

        let add_fn = typer.ast.get_function("add").unwrap();
        let add_syms: Vec<&Symbol> = typer.symbols.iter().filter(|s| s.scope == add_fn.body).collect();

        assert_eq!(3, add_syms.len());
        assert_eq!("x", add_syms[0].name);
        assert_eq!(SymbolKind::Local, add_syms[0].kind);
        assert_eq!("y", add_syms[1].name);
        assert_eq!(SymbolKind::Local, add_syms[1].kind);
        assert_eq!("z", add_syms[2].name);
        assert_eq!(SymbolKind::Local, add_syms[2].kind);
    }

    #[test]
    fn should_reject_structs_passed_by_value() {
        let code = r###"
            fun takes_string(x: string): void {}
        "###;
        let typer = Typer::from_code(code).unwrap();
        let ok = typer.check().is_ok();
        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_pointer_argument_with_literal() {
        let code = r###"
            fun takes_int_ptr(x: &int): void {}
            takes_int_ptr(1);
        "###;
        let typer = Typer::from_code(code).unwrap();
        let ok = typer.check().is_ok();
        assert_eq!(false, ok);
    }

    #[test]
    fn should_accept_pointer_expressions() {
        let code = r###"
        fun takes_int_ptr(x: &int): void {}
        takes_int_ptr(&1);
        "###;
        let typer = Typer::from_code(code).unwrap();
        let ok = typer.check().is_ok();
        assert_eq!(true, ok);
    }
}
