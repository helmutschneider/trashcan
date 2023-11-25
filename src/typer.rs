use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};

use crate::ast::{ASTLike, Expression, Function, Statement, StatementIndex};
use crate::tokenizer::TokenKind;
use crate::util::{report_error, Error, SourceLocation, Offset};
use crate::{ast, tokenizer};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub enum Type {
    Void,
    Bool,
    Int,
    Pointer(Box<Type>),
    Struct(String, Vec<StructMember>),
    Function(Vec<Type>, Box<Type>),
}

impl Type {
    pub fn size(&self) -> i64 {
        return self.memory_layout().iter().sum();
    }

    pub fn is_pointer(&self) -> bool {
        return matches!(self, Self::Pointer(_));
    }

    pub fn is_struct(&self) -> bool {
        return matches!(self, Self::Struct(_, _));
    }

    pub fn memory_layout(&self) -> Vec<i64> {
        if let Self::Struct(_, members) = self {
            let mut member_layout: Vec<i64> = Vec::new();

            for f in members {
                let type_ = f.type_.as_ref().unwrap();
                member_layout.extend(type_.memory_layout());
            }

            return member_layout;
        }

        return vec![8];
    }

    pub fn find_struct_member(&self, name: &str) -> Option<StructMember> {
        if let Type::Pointer(inner) = self {
            return inner.find_struct_member(name);
        }
        if let Type::Struct(_, members) = self {
            for m in members {
                if m.name == name {
                    return Some(m.clone());
                }
            }
        }
        return None;
    }

    pub fn find_struct_member_offset(&self, name: &str) -> Option<Offset> {
        if let Type::Pointer(inner) = self {
            return inner.find_struct_member_offset(name);
        }
        if let Type::Struct(_, members) = self {
            let mut offset: i64 = 0;
            for m in members {
                if m.name == name {
                    return Some(Offset::Positive(offset));
                }
                offset += m.type_.as_ref().unwrap().size();
            }
        }
        return None;
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Void => "void".to_string(),
            Self::Bool => "bool".to_string(),
            Self::Int => "int".to_string(),
            Self::Pointer(inner) => format!("{}", inner),
            Self::Struct(name, _) => name.clone(),
            Self::Function(arg_types, ret_type) => {
                let arg_s = arg_types.iter().map(|t| t.to_string()).collect::<Vec<String>>().join(", ");
                format!("fun ({}): {}", arg_s, ret_type)
            }
        };
        return s.fmt(f)
    }
}

impl std::cmp::PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        // pointers are invisible to the end-user and we don't display
        // them while formatting types. howerver, they are important for
        // code generation so we must take them into account when comparing
        // types.
        //   -johan, 2023-11-24
        if let Type::Pointer(a) = self {
            if let Type::Pointer(b) = other {
                return a == b;
            }
            return false;
        }
        return self.to_string() == other.to_string();
    }
}

#[derive(Debug, Clone)]
pub struct StructMember {
    pub name: String,
    pub type_: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct TypeTable {
    types: HashMap<String, Type>,
}

impl TypeTable {
    fn new() -> Self {
        let mut types = TypeTable {
            types: HashMap::new(),
        };
        
        // TODO: we only support 8-byte types currently as all instructions use the
        //   64-bit registers. maybe we should support smaller types? who knows.
        let type_void = types.add_type(Type::Void);
        let type_bool = types.add_type(Type::Bool);
        let type_int = types.add_type(Type::Int);

        let type_ptr_to_void = types.pointer_to(&type_void);
        let type_string = types.add_struct_type(
            "string",
            &[("length", type_int), ("data", type_ptr_to_void)],
        );

        return types;
    }

    pub fn pointer_to(&self, type_: &Type) -> Type {
        return Type::Pointer(Box::new(type_.clone()));
    }

    fn add_type(&mut self, type_: Type) -> Type {
        self.types.insert(type_.to_string(), type_.clone());
        return type_;
    }

    fn add_struct_type(&mut self, name: &str, members: &[(&str, Type)]) -> Type {
        if name.is_empty() {
            panic!("struct types must be named.");
        }

        let mut stuff: Vec<StructMember> = Vec::new();

        for (name, type_) in members {
            stuff.push(StructMember {
                name: name.to_string(),
                type_: Some(type_.clone()),
            });
        }

        let struct_type = Type::Struct(name.to_string(), stuff);

        return self.add_type(struct_type);
    }

    fn add_function_type(&mut self, arguments: &[Type], return_type: Type) -> Type {
        let args: Vec<Type> = arguments.iter().map(|t| t.clone()).collect();
        let type_ = Type::Function(args, Box::new(return_type));
        return self.add_type(type_);
    }

    pub fn get_type_by_name(&self, name: &str) -> Option<Type> {
        return self.types.get(name).map(|t| t.clone());
    }

    fn try_resolve_type(&self, decl: &ast::TypeName) -> Result<Type, tokenizer::Token> {
        let found = self.get_type_by_name(&decl.token.value);
                
        return match found {
            Some(t) => Ok(t),
            None => Err(decl.token.clone()),
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
    pub id: i64,
    pub name: String,
    pub kind: SymbolKind,
    pub type_: Option<Type>,
    pub declared_at: StatementIndex,
    pub scope: StatementIndex,
    pub is_function_argument: bool,
}

fn maybe_coerce_function_argument_to_pointer(type_: Type) -> Type {
    if type_.is_struct() {
        return Type::Pointer(Box::new(type_));
    }
    return type_;
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
                let mut fx_arg_types: Vec<Type> = Vec::new();

                // create symbols in the function body scopes for its locals.
                for arg in &fx.arguments {
                    let arg_type = types.try_resolve_type(&arg.type_)
                        .map(maybe_coerce_function_argument_to_pointer)
                        .ok();
                    let arg_sym = Symbol {
                        id: symbols.len() as i64,
                        name: arg.name_token.value.clone(),
                        kind: SymbolKind::Local,
                        type_: arg_type.clone(),
                        declared_at: *child_index,
                        scope: fx.body,
                        is_function_argument: true,
                    };
                    symbols.push(arg_sym);

                    if let Some(t) = arg_type {
                        fx_arg_types.push(t);
                    }
                }

                let fx_type: Option<Type> = {
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
                    id: symbols.len() as i64,
                    name: fx.name_token.value.clone(),
                    kind: SymbolKind::Function,
                    type_: fx_type,
                    declared_at: *child_index,
                    scope: scope,
                    is_function_argument: false,
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
                    id: symbols.len() as i64,
                    name: v.name_token.value.clone(),
                    kind: SymbolKind::Local,
                    type_: type_,
                    declared_at: *child_index,
                    scope: scope,
                    is_function_argument: false,
                };
                symbols.push(sym);
            }
            Statement::Struct(struct_) => {
                let name = &struct_.name_token.value;
                let mut members: Vec<StructMember> = Vec::new();

                for m in &struct_.members {
                    let field_type = types.try_resolve_type(&m.type_).ok();

                    members.push(StructMember {
                        name: m.field_name_token.value.clone(),
                        type_: field_type,
                    });
                }

                let type_ = Type::Struct(name.clone(), members);
                types.add_type(type_);
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
        Statement::Struct(struct_) => Some(struct_.parent),
    };
}

enum WalkResult<T> {
    Stop,
    ToParent,
    Ok(T),
}

fn walk_up_ast_from_statement<T, F: FnMut(&ast::Statement, StatementIndex) -> WalkResult<T>>(
    ast: &ast::AST,
    at: StatementIndex,
    fx: &mut F,
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
    let found = walk_up_ast_from_statement(ast, index, &mut |stmt, i| {
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
    pub fn try_find_symbol(
        &self,
        name: &str,
        kind: SymbolKind,
        at: StatementIndex,
    ) -> Option<Symbol> {
        return self.try_find_symbols(name, kind, at).first().cloned();
    }

    pub fn try_find_symbols(
        &self,
        name: &str,
        kind: SymbolKind,
        at: StatementIndex,
    ) -> Vec<Symbol> {
        let mut out: Vec<Symbol> = Vec::new();

        walk_up_ast_from_statement(&self.ast, at, &mut |stmt, i| {
            if let Statement::Block(_) = stmt {
                for sym in &self.symbols {
                    if sym.name == name && sym.kind == kind && sym.scope == i {
                        out.push(sym.clone());
                    }
                }
            }

            // let's not resolve locals outside the function scope.
            if let Statement::Function(_) = stmt {
                if kind == SymbolKind::Local {
                    return WalkResult::Stop::<()>;
                }
            }

            return WalkResult::ToParent;
        });

        return out;
    }

    pub fn try_resolve_symbol_type(&self, symbol: &Symbol) -> Option<Type> {
        if let Some(type_) = &symbol.type_ {
            return Some(type_.clone());
        }

        if symbol.kind == SymbolKind::Local {
            // locals might not have a declared type, so let's infer the type.
            if let Statement::Variable(var) = self.ast.get_statement(symbol.declared_at) {
                return self.try_infer_expression_type(&var.initializer);
            }
        }

        return None;
    }

    pub fn try_infer_expression_type(&self, expr: &ast::Expression) -> Option<Type> {
        return match expr {
            ast::Expression::IntegerLiteral(_) => Some(Type::Int),
            ast::Expression::StringLiteral(_) => Some(self.types.get_type_by_name("string").unwrap()),
            ast::Expression::FunctionCall(fx_call) => {
                let fx_sym = self.try_find_symbol(
                    &fx_call.name_token.value,
                    SymbolKind::Function,
                    fx_call.parent,
                )?;
                let fx_type = fx_sym.type_.as_ref()?;
                if let Type::Function(_, ret_type) = &fx_type {
                    return Some(*ret_type.clone());
                }
                return None;
            }
            ast::Expression::BinaryExpr(bin_expr) => match &bin_expr.operator.kind {
                TokenKind::DoubleEquals => Some(Type::Bool),
                TokenKind::NotEquals => Some(Type::Bool),
                TokenKind::Plus | TokenKind::Minus => {
                    let left_type = self.try_infer_expression_type(&bin_expr.left);
                    let right_type = self.try_infer_expression_type(&bin_expr.right);

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
                let maybe_sym = self.try_find_symbol(&ident.name, SymbolKind::Local, ident.parent);

                return match maybe_sym {
                    Some(s) => self.try_resolve_symbol_type(&s),
                    None => None,
                };
            }
            ast::Expression::Void => Some(Type::Void),
            ast::Expression::StructInitializer(s) => {
                let struct_type = self.types.get_type_by_name(&s.name_token.value);
                struct_type
            }
            ast::Expression::MemberAccess(prop_access) => {
                let left_type = self.try_infer_expression_type(&prop_access.left)?;
                let right = &prop_access.right;
                let member = left_type.find_struct_member(&right.name).and_then(|m| m.type_);

                if let Some(m) = member {
                    if left_type.is_pointer() {
                        return Some(self.types.pointer_to(&m));
                    }
                    return Some(m);
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
        given_type: &Option<Type>,
        expected_type: &Option<Type>,
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
        given_type: Option<Type>,
        expected_type: Option<Type>,
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

    fn check_type_declaration(&self, decl: &ast::TypeName, errors: &mut Vec<Error>) -> Option<Type> {
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
                        let given_type = self.try_infer_expression_type(&var.initializer);
                        self.maybe_report_type_mismatch(&given_type, &Some(type_), location, errors);
                    }
                }

                let other_symbols_in_scope = self.try_find_symbols(&var.name_token.value, SymbolKind::Local, var.parent);

                if other_symbols_in_scope.len() > 1 {
                    self.report_error(&format!("cannot redeclare block-scoped variable '{}'", var.name_token.value), location, errors);
                }

                self.check_expression(&var.initializer, errors);
            }
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);

                self.check_expression(&if_expr.condition, errors);
                let given_type = self.try_infer_expression_type(&if_expr.condition);

                self.maybe_report_type_mismatch(&given_type, &Some(Type::Bool), location, errors);

                let if_block = self.ast.get_block(if_expr.block);
                self.check_block(if_block, errors);
            }
            ast::Statement::Function(fx) => {
                for fx_arg in &fx.arguments {
                    self.check_type_declaration(&fx_arg.type_, errors);
                }

                let fx_body = self.ast.get_block(fx.body);

                if let Some(ret_type) = self.check_type_declaration(&fx.return_type, errors) {
                    let is_void_return = ret_type == Type::Void;

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
                    let given_type = self.try_infer_expression_type(&ret.expr);
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
            ast::Statement::Struct(struct_) => {
                let name = &struct_.name_token.value;
                let struct_type = self.types.get_type_by_name(name);

                for m in &struct_.members {
                    let member_name = &m.field_name_token.value;
                    let member_type = struct_type.as_ref()
                        .and_then(|s| s.find_struct_member(&member_name))
                        .and_then(|m| m.type_);
                    let loc = SourceLocation::Token(&m.type_.token);

                    self.maybe_report_missing_type(&m.type_.token.value, &member_type, loc, errors)
                }
            }
        }
    }

    fn check_expression(&self, expr: &ast::Expression, errors: &mut Vec<Error>) {
        match expr {
            ast::Expression::FunctionCall(fx_call) => {
                let ident_location = SourceLocation::Token(&fx_call.name_token);
                let maybe_fx_sym = self.try_find_symbol(
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

                if let Type::Function(arg_types, _) = &fx_type {
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

                            let declared_type = arg_types[i].clone();
                            let given_type = self.try_infer_expression_type(&call_arg)
                                .map(maybe_coerce_function_argument_to_pointer);

                            let call_arg_location = SourceLocation::Expression(call_arg);
                            self.maybe_report_type_mismatch(
                                &given_type,
                                &Some(declared_type),
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
                let left = self.try_infer_expression_type(&bin_expr.left);
                let right = self.try_infer_expression_type(&bin_expr.right);

                self.maybe_report_no_type_overlap(right, left, location, errors);
            }
            ast::Expression::Identifier(ident) => {
                let location = SourceLocation::Token(&ident.token);
                let ident_sym = self.try_find_symbol(&ident.name, SymbolKind::Local, ident.parent);
                self.maybe_report_missing_type(&ident.name, &ident_sym, location, errors);
            }
            ast::Expression::StructInitializer(s) => {
                let name = &s.name_token.value;
                let type_ = self.types.get_type_by_name(name);
                let location = SourceLocation::Token(&s.name_token);

                self.maybe_report_missing_type(name, &type_, location, errors);

                if let Some(t) = &type_ {
                    if let Type::Struct(_, members) = t {
                        let mut missing_members: HashSet<String> = members.iter().map(|m| m.name.clone()).collect();

                        for m in &s.members {
                            self.check_expression(&m.value, errors);
                            let member_name = &m.field_name_token.value;
                            let maybe_member = t.find_struct_member(&member_name);

                            missing_members.remove(member_name);
    
                            if let Some(member) = maybe_member {
                                let given_type = &self.try_infer_expression_type(&m.value);
                                let loc = SourceLocation::Token(&m.field_name_token);
                                self.maybe_report_type_mismatch(given_type, &member.type_, loc, errors)
                            } else {
                                let loc = SourceLocation::Token(&m.field_name_token);
                                self.report_error(&format!("member '{}' does not exist on type '{}'", member_name, t), loc, errors);
                            }
                        }

                        if !missing_members.is_empty() {
                            let member_s = missing_members.iter()
                                .map(|s| format!("'{}'", s))
                                .collect::<Vec<String>>()
                                .join(", ");
                            self.report_error(&format!("type '{}' is missing the following members: {}", t, member_s), location, errors);
                        }
                    } else {
                        let loc = SourceLocation::Token(&s.name_token);
                        self.report_error(&format!("type '{}' is not a struct", t), loc, errors);
                    }
                }
            }
            ast::Expression::MemberAccess(prop_access) => {
                self.check_expression(&prop_access.left, errors);

                let maybe_left_type = self.try_infer_expression_type(&prop_access.left);

                if let Some(left_type) = &maybe_left_type {
                    let right = &prop_access.right;
                    let member = left_type.find_struct_member(&right.name);
                    if member.is_none() {
                        let loc = SourceLocation::Token(&right.token);
                        self.report_error(&format!("property '{}' does not exist on type '{}'", right.name, left_type), loc, errors);
                    }
                }
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

        create_symbols_at_statement(&ast, &mut symbols, &mut types, body_index);

        let type_str = types.get_type_by_name("string").unwrap();

        // the built-in print function.
        let type_print = types.add_function_type(&[types.pointer_to(&type_str)], Type::Void);
        let print_sym = Symbol {
            id: symbols.len() as i64,
            name: "print".to_string(),
            kind: SymbolKind::Function,
            type_: Some(type_print),
            declared_at: body_index,
            scope: body_index,
            is_function_argument: false,
        };
        symbols.push(print_sym);

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
    fn should_reject_redeclaration_in_same_scope() {
        let code = r###"
        fun main(): void {
            var x = 420;
            var x = 69;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_missing_struct_type() {
        let code = r###"
        var x = person {};
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_non_struct_with_struct_initializer() {
        let code = r###"
        var x = int {};
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_struct_member_type_mismatch() {
        let code = r###"
        type person = struct { name: string };
        var x = person { name: 1 };
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_extranous_struct_member() {
        let code = r###"
        type person = struct { name: string };
        var x = person { name: "hello!", age: 5 };
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_missing_struct_members() {
        let code = r###"
        type person = struct { name: string };
        var x = person {};
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_accept_member_access_of_implicit_pointer() {
        let code = r###"
        type person = struct { name: string };
        fun takes(x: person): void {
            var y = x.name;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }
}
