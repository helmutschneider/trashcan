use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;

use crate::ast::Symbol;
use crate::ast::{ASTLike, Function, Statement, StatementIndex, SymbolScope};
use crate::tokenizer::TokenKind;
use crate::util::{report_error, Error, SourceLocation};
use crate::{ast, tokenizer};
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct Typer {
    pub ast: ast::AST,
    pub program: String,
    pub types: TypeTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(u64);

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

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Void,
    Bool,
    Byte,
    Int,
    Pointer(TypeId),
    Struct(HashMap<String, TypeId>),
    Function(Vec<TypeId>, TypeId),
}

#[derive(Debug)]
pub struct TypeTable {
    data: HashMap<TypeId, Type>,
}

fn get_type_id_from_name(name: &str) -> TypeId {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    return TypeId(hasher.finish());
}

impl TypeTable {
    fn new() -> Self {
        return TypeTable {
            data: HashMap::new(),
        };
    }

    fn add_type(&mut self, name: &str, kind: TypeKind) -> TypeId {
        let id = get_type_id_from_name(name);

        if self.data.contains_key(&id) {
            panic!("type '{}' already exists", name);
        }

        let type_ = Type {
            id: id,
            name: name.to_string(),
            kind: kind,
        };
        self.data.insert(id, type_);
        return id;
    }

    fn add_pointer_type(&mut self, to: TypeId) -> TypeId {
        let to_type = &self.data[&to];
        let name = format!("&{}", to_type.name);
        return self.add_type(&name, TypeKind::Pointer(to));
    }

    fn add_struct_type(&mut self, name: &str, fields: &[(&str, TypeId)]) -> TypeId {
        let mut map: HashMap<String, TypeId> = HashMap::new();

        for (name, type_) in fields {
            map.insert(name.to_string(), *type_);
        }

        return self.add_type(name, TypeKind::Struct(map));
    }

    fn add_function_type(&mut self, arguments: &[TypeId], return_type_id: TypeId) -> TypeId {
        let mut arg_names: Vec<String> = Vec::new();

        for arg in arguments {
            let type_ = &self.data[arg];
            arg_names.push(type_.name.clone());
        }

        let arg_s = arg_names.join(", ");
        let return_type = &self.data[&return_type_id];
        let name = format!("fun ({}): {}", arg_s, return_type.name);

        let arg_ids = arguments.iter().map(|s| *s).collect();
        let kind = TypeKind::Function(arg_ids, return_type.id);

        return self.add_type(&name, kind);
    }

    fn get_type_by_id(&self, id: TypeId) -> &Type {
        return &self.data[&id];
    }

    fn get_type_by_name(&self, name: &str) -> Option<&Type> {
        let id = get_type_id_from_name(name);
        return self.data.get(&id);
    }
}

const TYPE_NAME_VOID: &'static str = "void";
const TYPE_NAME_BOOL: &'static str = "bool";
const TYPE_NAME_BYTE: &'static str = "byte";
const TYPE_NAME_INT: &'static str = "int";
const TYPE_NAME_STRING: &'static str = "string";

impl Typer {
    fn try_infer_type(&self, expr: &ast::Expression) -> Option<&Type> {
        return match expr {
            ast::Expression::IntegerLiteral(_) => self.types.get_type_by_name(TYPE_NAME_INT),
            ast::Expression::StringLiteral(_) => self.types.get_type_by_name(TYPE_NAME_STRING),
            ast::Expression::FunctionCall(fx_call) => {
                let fx_sym = self.ast.get_symbol(&fx_call.name_token.value, SymbolScope::Global, fx_call.parent)?;
                let fx_type_name = fx_sym.type_.as_ref().unwrap();
                let fx_type = self.types.get_type_by_name(&fx_type_name)?;
                if let TypeKind::Function(_, ret_type_id) = fx_type.kind {
                    let t = self.types.get_type_by_id(ret_type_id);
                    return Some(t);
                }
                return None;
            }
            ast::Expression::BinaryExpr(bin_expr) => match &bin_expr.operator.kind {
                TokenKind::DoubleEquals => self.types.get_type_by_name(TYPE_NAME_BOOL),
                TokenKind::NotEquals => self.types.get_type_by_name(TYPE_NAME_BOOL),
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
                let maybe_sym = self.ast.get_symbol(&ident.name, SymbolScope::Local, ident.parent);
                if maybe_sym.is_none() {
                    return None;
                }
                let sym = maybe_sym.unwrap();

                if let Some(t) = &sym.type_ {
                    return self.types.get_type_by_name(t);
                }

                let stmt = self.ast.get_statement(sym.declared_at);
                let var_stmt = match stmt {
                    Statement::Variable(x) => x,
                    _ => panic!("expected a variable declaration, got {:?}", stmt),
                };

                return self.try_infer_type(&var_stmt.initializer);
            }
            ast::Expression::None => self.types.get_type_by_name(TYPE_NAME_VOID),
        };
    }

    fn maybe_report_missing_type<T>(
        &self,
        name: &str,
        type_: Option<T>,
        at: SourceLocation,
        errors: &mut Vec<Error>,
    ) {
        if let None = type_ {
            self.report_error(&format!("cannot find name '{}'", name), at, errors);
        }
    }

    fn maybe_report_type_mismatch(
        &self,
        given_type: Option<&Type>,
        expected_type: Option<&Type>,
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
        given_type: Option<&Type>,
        expected_type: Option<&Type>,
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

    fn check_statement(&self, stmt: &ast::Statement, errors: &mut Vec<Error>) {
        match stmt {
            ast::Statement::Variable(var) => {
                let location = SourceLocation::Token(&var.name_token);

                if let Some(type_token) = &var.type_token {
                    let declared_type = self.types.get_type_by_name(&type_token.value);
                    let given_type = self.try_infer_type(&var.initializer);
    
                    self.maybe_report_missing_type(&type_token.value, declared_type, location, errors);
                    self.maybe_report_type_mismatch(given_type, declared_type, location, errors);
                }

                self.check_expression(&var.initializer, errors);
            }
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);

                self.check_expression(&if_expr.condition, errors);
                let bool_type = self.types.get_type_by_name(TYPE_NAME_BOOL);
                let given_type = self.try_infer_type(&if_expr.condition);

                self.maybe_report_type_mismatch(given_type, bool_type, location, errors);

                let if_block = self.ast.get_block(if_expr.block);
                self.check_block(if_block, errors);
            }
            ast::Statement::Function(fx) => {
                for fx_arg in &fx.arguments {
                    let location = SourceLocation::Token(&fx_arg.type_token);
                    let declared_type = self.types.get_type_by_name(&fx_arg.type_token.value);
                    self.maybe_report_missing_type(&fx_arg.type_token.value, declared_type, location, errors);
                }

                let return_type = self.types.get_type_by_name(&fx.return_type_token.value);
                let return_type_location = SourceLocation::Token(&fx.return_type_token);
                self.maybe_report_missing_type(
                    &fx.return_type_token.value,
                    return_type,
                    return_type_location,
                    errors,
                );

                let fx_body = self.ast.get_block(fx.body);
                self.check_block(fx_body, errors);

                let is_void_return = return_type.map(|t| t.kind == TypeKind::Void).unwrap_or(false);

                if !is_void_return {
                    let has_return_statement = fx_body.statements.iter().any(|k| {
                        let stmt = self.ast.get_statement(*k);
                        return matches!(stmt, Statement::Return(_));
                    });
    
                    if !has_return_statement {
                        self.report_error("missing 'return' statement", return_type_location, errors);
                    }
                }
            }
            ast::Statement::Block(b) => {
                self.check_block(b, errors);
            }
            ast::Statement::Return(ret) => {
                if let Some(fx) = self.ast.get_enclosing_function(ret.parent) {
                    let return_type = self.types.get_type_by_name(&fx.return_type_token.value);
                    let given_type = self.try_infer_type(&ret.expr);

                    let location = SourceLocation::Expression(&ret.expr);
                    self.maybe_report_type_mismatch(given_type, return_type, location, errors);
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
                let maybe_fx_sym = self.ast.get_symbol(&fx_call.name_token.value, SymbolScope::Global, fx_call.parent);
                self.maybe_report_missing_type(&fx_call.name_token.value, maybe_fx_sym, ident_location, errors);

                if maybe_fx_sym.is_none() {
                    return;
                }

                let fx_sym = maybe_fx_sym.unwrap();
                let fx_type_name = fx_sym.type_.as_ref().unwrap();
                let maybe_fx_type = self.types.get_type_by_name(fx_type_name);

                if maybe_fx_type.is_none() {
                    return;
                }

                let fx_type = maybe_fx_type.unwrap();

                if let TypeKind::Function(arg_types, _) = &fx_type.kind {
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
    
                            let declared_type = self.types.get_type_by_id(arg_types[i]);
                            let given_type = self.try_infer_type(&call_arg);
    
                            let call_arg_location = SourceLocation::Expression(call_arg);
                            self.maybe_report_type_mismatch(
                                given_type,
                                Some(declared_type),
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
                let ident_sym = self.ast.get_symbol(&ident.name, SymbolScope::Local, ident.parent);
                self.maybe_report_missing_type(&ident.name, ident_sym, location, errors);
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
        let mut ast = ast::AST::from_code(program)?;

        let mut types = TypeTable::new();
        let type_id_void = types.add_type(TYPE_NAME_VOID, TypeKind::Void);
        let type_id_bool = types.add_type(TYPE_NAME_BOOL, TypeKind::Bool);
        let type_id_byte = types.add_type(TYPE_NAME_BYTE, TypeKind::Byte);
        let type_id_int = types.add_type(TYPE_NAME_INT, TypeKind::Int);
        let type_id_ptr_to_byte = types.add_pointer_type(type_id_byte);

        let type_id_string = types.add_struct_type(
            "string",
            &[("length", type_id_int), ("data", type_id_ptr_to_byte)],
        );

        let num_top_level_symbols = ast.body().symbols.len();

        // when the AST is constructed it doesn't contain type information for
        // the function symbols. let's iterate over the top-level function statements
        // and generate types and update the symbol table.
        for k in 0..num_top_level_symbols {
            let sym = &ast.body().symbols[k];
            if sym.scope != SymbolScope::Global {
                continue;
            }
            let fx = ast.get_function(&sym.name).unwrap();
            let mut arg_ids: Vec<TypeId> = Vec::new();

            for arg in &fx.arguments {
                let type_ = types.get_type_by_name(&arg.type_token.value);
                if let Some(t) = type_ {
                    arg_ids.push(t.id);
                }
            }

            let maybe_ret_type = types.get_type_by_name(&fx.return_type_token.value);

            if maybe_ret_type.is_some() && arg_ids.len() == fx.arguments.len() {
                let fn_type_id = types.add_function_type(&arg_ids, maybe_ret_type.unwrap().id);

                if let Statement::Block(block) = &mut ast.statements[ast.body_index.0] {
                    // this actually modifies the same value as 'sym'. it's
                    // weird that rust allows this. maybe we're cheating the
                    // borrow checker slightly?
                    block.symbols[k].type_ = Some(types.get_type_by_id(fn_type_id).name.clone());
                }
            }
        }

        // the built-in print function.
        if let Statement::Block(x) = &mut ast.statements[ast.body_index.0] {
            let type_id_print = types.add_function_type(&[type_id_string, type_id_int], type_id_void);
            let print_sym = Symbol {
                name: "print".to_string(),
                type_: Some(types.get_type_by_id(type_id_print).name.clone()),
                scope: SymbolScope::Global,
                declared_at: ast.body_index,
            };
            x.symbols.push(print_sym);
        }

        let typer = Self {
            ast: ast,
            program: program.to_string(),
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
}
