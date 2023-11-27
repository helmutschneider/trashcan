use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::ast::{Expression, Function, Statement, StatementId, SymbolKind, SymbolId, UntypedSymbol};
use crate::tokenizer::TokenKind;
use crate::util::{report_error, Error, Offset, SourceLocation};
use crate::{ast, tokenizer};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

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

    pub fn is_scalar(&self) -> bool {
        return matches!(self, Self::Void | Self::Bool | Self::Int);
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
                let type_ = &f.type_;
                member_layout.extend(type_.memory_layout());
            }

            return member_layout;
        }

        return vec![8];
    }

    pub fn find_struct_member(&self, name: &str) -> Option<StructMember> {
        if let Self::Pointer(inner) = self {
            return inner.find_struct_member(name);
        }
        if let Self::Struct(_, members) = self {
            for m in members {
                if m.name == name {
                    return Some(m.clone());
                }
            }
        }
        return None;
    }

    pub fn find_struct_member_offset(&self, name: &str) -> Option<Offset> {
        if let Self::Pointer(inner) = self {
            return inner.find_struct_member_offset(name);
        }
        if let Self::Struct(_, members) = self {
            let mut offset: i64 = 0;
            for m in members {
                if m.name == name {
                    return Some(Offset::Positive(offset));
                }
                offset += m.type_.size();
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
            Self::Pointer(inner) => format!("&{}", inner),
            Self::Struct(name, _) => name.clone(),
            Self::Function(arg_types, ret_type) => {
                let arg_s = arg_types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("fun ({}): {}", arg_s, ret_type)
            }
        };
        return s.fmt(f);
    }
}

impl std::cmp::PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        // pointers are invisible to the end-user and we don't display
        // them while formatting types. howerver, they are important for
        // code generation so we must take them into account when comparing
        // types.
        //   -johan, 2023-11-24
        if let Self::Pointer(a) = self {
            if let Self::Pointer(b) = other {
                return a == b;
            }
            return false;
        }

        if let Self::Struct(a_name, _) = self {
            if let Self::Struct(b_name, _) = other {
                return a_name == b_name;
            }
            return false;
        }

        return match self {
            Self::Void => matches!(other, Self::Void),
            Self::Bool => matches!(other, Self::Bool),
            Self::Int => matches!(other, Self::Int),
            _ => false,
        };
    }
}

fn maybe_remove_pointer_if_scalar(type_: Type) -> Type {
    if let Type::Pointer(inner) = &type_ {
        if inner.is_scalar() {
            return *inner.clone();
        }
    }
    return type_.clone();
}

#[derive(Debug, Clone)]
pub struct StructMember {
    pub name: String,
    pub type_: Type,
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
                type_: type_.clone(),
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
        let maybe_type = self.get_type_by_name(&decl.token.value);

        return match maybe_type {
            Some(t) => {
                let type_ = if decl.is_pointer {
                    self.pointer_to(&t)
                } else {
                    t
                };
                Ok(type_)
            }
            None => Err(decl.token.clone()),
        };
    }
}

#[derive(Debug, Clone)]
pub struct TypedSymbol {
    pub id: SymbolId,
    pub name: String,
    pub kind: SymbolKind,
    pub type_: Type,
    pub declared_at: Rc<Statement>,
    pub scope: Rc<Statement>,
}

fn create_types_and_symbols(untyped: &[UntypedSymbol], typer: &mut Typer) {
    const MAX_YIELD_ATTEMPTS: i64 = 10;

    let mut must_yield_for: VecDeque<&UntypedSymbol> = untyped.iter().collect();
    let mut num_inference_attempts: HashMap<SymbolId, i64> = HashMap::new();

    while let Some(sym) = must_yield_for.pop_front() {
        let inferred_type: Option<Type> = match sym.kind {
            SymbolKind::Local => {
                match sym.declared_at.as_ref() {
                    Statement::Variable(var) => {
                        if let Some(t) = &var.type_ {
                            typer.types.try_resolve_type(&t).ok()
                        } else {
                            typer.try_infer_expression_type(&var.initializer)
                        }
                    }
                    Statement::Function(fx) => {
                        let fx_arg = fx.arguments.iter()
                            .find(|x| x.name_token.value == sym.name)
                            .unwrap();
                        let fx_arg_type = typer.types.try_resolve_type(&fx_arg.type_).ok();
                        fx_arg_type
                    }
                    _ => panic!("bad. local found at {}", sym.declared_at.id())
                }
            }
            SymbolKind::Function => {
                let fx = match sym.declared_at.as_ref() {
                    Statement::Function(x) => x,
                    _ => panic!("bad. function found at {}", sym.declared_at.id()),
                };
                let mut fx_arg_types: Vec<Type> = Vec::new();

                for arg in &fx.arguments {
                    let fx_type = typer.types.try_resolve_type(&arg.type_);

                    if let Ok(t) = fx_type {
                        fx_arg_types.push(t);
                    }
                }

                let ret_type = typer.types.try_resolve_type(&fx.return_type);

                if fx_arg_types.len() == fx.arguments.len() && ret_type.is_ok() {
                    let fx_type = typer.types.add_function_type(&fx_arg_types, ret_type.unwrap());
                    Some(fx_type)
                } else {
                    None
                }
            }
            SymbolKind::Type => {
                let struct_ = match sym.declared_at.as_ref() {
                    Statement::Struct(x) => x,
                    _ => panic!("bad. type found at {}", sym.declared_at.id())
                };
                let name = &struct_.name_token.value;
                let mut inferred_members: Vec<StructMember> = Vec::new();

                for m in &struct_.members {
                    let field_type = typer.types.try_resolve_type(&m.type_);

                    if let Ok(t) = field_type {
                        inferred_members.push(StructMember {
                            name: m.field_name_token.value.clone(),
                            type_: t,
                        });
                    }
                }

                if inferred_members.len() == struct_.members.len() {
                    let type_ = Type::Struct(name.clone(), inferred_members);
                    typer.types.add_type(type_.clone());
                    Some(type_)
                } else {
                    None
                }
            }
        };

        if let Some(t) = inferred_type {
            let scope = typer.ast.find_statement(sym.scope);
            let typed = TypedSymbol {
                id: sym.id,
                name: sym.name.clone(),
                kind: sym.kind,
                type_: t,
                declared_at: Rc::clone(&sym.declared_at),
                scope: Rc::clone(&scope.unwrap()),
            };
            typer.symbols.push(typed);
        } else {
            let num_attempts = num_inference_attempts.entry(sym.id).or_insert(0);
            *num_attempts += 1;
    
            // inference might fail if a symbol is referenced before it
            // is declared. that's ususally OK and we can just wait a bit
            // and try again later.
            //
            // in some cases (namely locals) it's actually a type error
            // but that is handled later in 'check_expression'.
            if *num_attempts < MAX_YIELD_ATTEMPTS {
                must_yield_for.push_back(sym);
            }
        }
    }
}

enum WalkResult<T> {
    Stop,
    ToParent,
    Ok(T),
}

fn walk_up_ast_from_statement<T, F: FnMut(&ast::Statement, StatementId) -> WalkResult<T>>(
    ast: &ast::AST,
    at: StatementId,
    fx: &mut F,
) -> Option<T> {
    let mut maybe_stmt_index = Some(at);

    while let Some(index) = maybe_stmt_index {
        let maybe_stmt = ast.find_statement(index);
        if maybe_stmt.is_none() {
            break;
        }

        let stmt = maybe_stmt.unwrap();
        let res = fx(stmt, index);

        match res {
            WalkResult::Ok(x) => {
                return Some(x);
            }
            WalkResult::Stop => {
                return None;
            }
            WalkResult::ToParent => {}
        }

        maybe_stmt_index = stmt.parent_id();
    }

    return None;
}

fn get_enclosing_function(ast: &ast::AST, at: StatementId) -> Option<&ast::Function> {
    let id = walk_up_ast_from_statement(ast, at, &mut |stmt, i| {
        if let Statement::Function(_) = stmt {
            return WalkResult::Ok(i);
        }
        return WalkResult::ToParent;
    });

    let stmt = ast.find_statement(id?)?;

    return match stmt.as_ref() {
        Statement::Function(fx) => Some(fx),
        _ => None,
    };
}

#[derive(Debug, Clone)]
pub struct Typer {
    pub ast: ast::AST,
    pub program: String,
    pub symbols: Vec<TypedSymbol>,
    pub types: TypeTable,
}

impl Typer {
    pub fn try_find_symbol(
        &self,
        name: &str,
        kind: SymbolKind,
        at: StatementId,
    ) -> Option<TypedSymbol> {
        return self.try_find_symbols(name, kind, at).first().cloned();
    }

    pub fn try_find_symbols(
        &self,
        name: &str,
        kind: SymbolKind,
        at: StatementId,
    ) -> Vec<TypedSymbol> {
        let mut out: Vec<TypedSymbol> = Vec::new();

        walk_up_ast_from_statement(&self.ast, at, &mut |stmt, i| {
            if let Statement::Block(_) = stmt {
                for sym in &self.symbols {
                    if sym.name == name && sym.kind == kind && sym.scope.as_ref() == stmt {
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

    pub fn try_infer_expression_type(&self, expr: &ast::Expression) -> Option<Type> {
        return match expr {
            ast::Expression::IntegerLiteral(_) => Some(Type::Int),
            ast::Expression::StringLiteral(_) => {
                Some(self.types.get_type_by_name("string").unwrap())
            }
            ast::Expression::FunctionCall(fx_call) => {
                let fx_sym = self.try_find_symbol(
                    &fx_call.name_token.value,
                    SymbolKind::Function,
                    fx_call.parent,
                )?;
                let fx_type = fx_sym.type_;
                if let Type::Function(_, ret_type) = &fx_type {
                    return Some(*ret_type.clone());
                }
                return None;
            }
            ast::Expression::BinaryExpr(bin_expr) => match &bin_expr.operator.kind {
                TokenKind::DoubleEquals => Some(Type::Bool),
                TokenKind::NotEquals => Some(Type::Bool),
                TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash => {
                    let left_type = self
                        .try_infer_expression_type(&bin_expr.left)
                        .map(maybe_remove_pointer_if_scalar);
                    let right_type = self
                        .try_infer_expression_type(&bin_expr.right)
                        .map(maybe_remove_pointer_if_scalar);

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
                    Some(s) => Some(s.type_),
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
                let member = left_type
                    .find_struct_member(&right.name)
                    .map(|m| m.type_);

                if let Some(m) = member {
                    if left_type.is_pointer() {
                        return Some(self.types.pointer_to(&m));
                    }
                    return Some(m);
                }

                return None;
            }
            ast::Expression::Pointer(ptr) => {
                let type_ = self.try_infer_expression_type(&ptr.expr);
                return type_.map(|t| self.types.pointer_to(&t));
            }
            ast::Expression::BooleanLiteral(_) => Some(Type::Bool)
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

    fn check_type_declaration(
        &self,
        decl: &ast::TypeName,
        errors: &mut Vec<Error>,
    ) -> Option<Type> {
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
                        let given_type = self.try_infer_expression_type(&var.initializer)
                            .map(maybe_remove_pointer_if_scalar);
                        let declared_type = maybe_remove_pointer_if_scalar(type_);
                        
                        self.maybe_report_type_mismatch(
                            &given_type,
                            &Some(declared_type),
                            location,
                            errors,
                        );
                    }
                }

                let other_symbols_in_scope =
                    self.try_find_symbols(&var.name_token.value, SymbolKind::Local, var.parent);

                if other_symbols_in_scope.len() > 1 {
                    self.report_error(
                        &format!(
                            "cannot redeclare block scoped variable '{}'",
                            var.name_token.value
                        ),
                        location,
                        errors,
                    );
                }

                self.check_expression(&var.initializer, errors);
            }
            ast::Statement::If(if_expr) => {
                let location = SourceLocation::Expression(&if_expr.condition);

                self.check_expression(&if_expr.condition, errors);
                let given_type = self.try_infer_expression_type(&if_expr.condition);

                self.maybe_report_type_mismatch(&given_type, &Some(Type::Bool), location, errors);

                let if_block = if_expr.block.as_block();
                self.check_block(if_block, errors);

                for else_ in &if_expr.else_ {
                    self.check_statement(&else_, errors);
                }
            }
            ast::Statement::Function(fx) => {
                for fx_arg in &fx.arguments {
                    let fx_arg_type = self.check_type_declaration(&fx_arg.type_, errors);

                    if let Some(t) = fx_arg_type {
                        if t.is_struct() {
                            let loc = SourceLocation::Token(&fx_arg.type_.token);
                            self.report_error(
                                &format!("type '{}' must be passed by reference", t),
                                loc,
                                errors,
                            );
                        }
                    }
                }

                let fx_body = fx.body.as_block();

                if let Some(ret_type) = self.check_type_declaration(&fx.return_type, errors) {
                    let is_void_return = ret_type == Type::Void;

                    if !is_void_return {
                        let has_return_statement = fx_body.statements.iter().any(|stmt| {
                            return matches!(stmt.as_ref(), Statement::Return(_));
                        });

                        if !has_return_statement {
                            // TODO: this should probably point to the declared return type,
                            //   but we don't have any way of locating a type declaration yet.
                            let location = SourceLocation::Token(&fx.name_token);
                            self.report_error("missing 'return' statement", location, errors);
                        }
                    }

                    if ret_type.is_struct() {
                        let loc = SourceLocation::Token(&fx.return_type.token);
                        self.report_error(
                            "only scalar values are supported the return type",
                            loc,
                            errors,
                        );
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
                    let location = if let Expression::Void = ret.expr {
                        SourceLocation::Token(&ret.token)
                    } else {
                        SourceLocation::Expression(&ret.expr)
                    };
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
                self.check_expression(&expr.expr, errors);
            }
            ast::Statement::Struct(struct_) => {
                let name = &struct_.name_token.value;

                for m in &struct_.members {
                    let member_name = &m.field_name_token.value;
                    let member_type = self.types.try_resolve_type(&m.type_).ok();
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
                let fx_type = fx_sym.type_;

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
                            let given_type = self.try_infer_expression_type(&call_arg);
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
                let mut left = self.try_infer_expression_type(&bin_expr.left);
                let mut right = self.try_infer_expression_type(&bin_expr.right);

                // the type system should allow us to compare pointers to scalars
                // with their dereferenced values.
                if let Some(Type::Pointer(inner)) = &left {
                    if inner.is_scalar() {
                        left = Some(*inner.clone());
                    }
                }

                if let Some(Type::Pointer(inner)) = &right {
                    if inner.is_scalar() {
                        right = Some(*inner.clone());
                    }
                }

                self.maybe_report_no_type_overlap(right, left, location, errors);
            }
            ast::Expression::Identifier(ident) => {
                let location = SourceLocation::Token(&ident.token);
                let ident_sym = self.try_find_symbol(&ident.name, SymbolKind::Local, ident.parent);
                self.maybe_report_missing_type(&ident.name, &ident_sym, location, errors);

                if let Some(sym) = ident_sym {
                    let sym_declared_at = sym.declared_at;

                    if sym_declared_at.id() > ident.parent {
                        self.report_error(&format!("block scoped variable '{}' used before its declaration", ident.name), location, errors)
                    }
                }
            }
            ast::Expression::StructInitializer(s) => {
                let name = &s.name_token.value;
                let type_ = self.types.get_type_by_name(name);
                let location = SourceLocation::Token(&s.name_token);

                self.maybe_report_missing_type(name, &type_, location, errors);

                if let Some(t) = &type_ {
                    if let Type::Struct(_, members) = t {
                        let mut missing_members: HashSet<String> =
                            members.iter().map(|m| m.name.clone()).collect();

                        for m in &s.members {
                            self.check_expression(&m.value, errors);
                            let member_name = &m.field_name_token.value;
                            let maybe_member = t.find_struct_member(&member_name);

                            missing_members.remove(member_name);

                            if let Some(member) = maybe_member {
                                let given_type = &self.try_infer_expression_type(&m.value);
                                let loc = SourceLocation::Token(&m.field_name_token);
                                self.maybe_report_type_mismatch(
                                    given_type,
                                    &Some(member.type_),
                                    loc,
                                    errors,
                                )
                            } else {
                                let loc = SourceLocation::Token(&m.field_name_token);
                                self.report_error(
                                    &format!(
                                        "member '{}' does not exist on type '{}'",
                                        member_name, t
                                    ),
                                    loc,
                                    errors,
                                );
                            }
                        }

                        if !missing_members.is_empty() {
                            let member_s = missing_members
                                .iter()
                                .map(|s| format!("'{}'", s))
                                .collect::<Vec<String>>()
                                .join(", ");
                            self.report_error(
                                &format!(
                                    "type '{}' is missing the following members: {}",
                                    t, member_s
                                ),
                                location,
                                errors,
                            );
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
                        self.report_error(
                            &format!(
                                "property '{}' does not exist on type '{}'",
                                right.name, left_type
                            ),
                            loc,
                            errors,
                        );
                    }
                }
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
        let root = Rc::clone(&ast.root);
        let untyped_symbols = ast.symbols.clone();
        let mut typer = Self {
            ast: ast,
            program: program.to_string(),
            symbols: Vec::new(),
            types: TypeTable::new(),
        };

        create_types_and_symbols(&untyped_symbols, &mut typer);

        let type_str = typer.types.get_type_by_name("string").unwrap();

        // the built-in print function.
        let type_print = typer.types.add_function_type(&[typer.types.pointer_to(&type_str)], Type::Void);
        let print_sym = TypedSymbol {
            id: SymbolId(typer.symbols.len() as i64),
            name: "print".to_string(),
            kind: SymbolKind::Function,
            type_: type_print,
            declared_at: Rc::clone(&root),
            scope: Rc::clone(&root),
        };
        typer.symbols.push(print_sym);

        let type_exit = typer.types.add_function_type(&[Type::Int], Type::Void);
        let print_sym = TypedSymbol {
            id: SymbolId(typer.symbols.len() as i64),
            name: "exit".to_string(),
            kind: SymbolKind::Function,
            type_: type_exit,
            declared_at: Rc::clone(&root),
            scope: Rc::clone(&root),
        };
        typer.symbols.push(print_sym);

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
    use crate::typer::{TypedSymbol, SymbolKind, Typer};

    use super::Type;

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
        let body_syms: Vec<&TypedSymbol> = typer
            .symbols
            .iter()
            .filter(|s| s.scope == typer.ast.root)
            .collect();

        assert_eq!(3, body_syms.len());
        assert_eq!("add", body_syms[0].name);
        assert_eq!(SymbolKind::Function, body_syms[0].kind);

        assert_eq!("print", body_syms[1].name);
        assert_eq!(SymbolKind::Function, body_syms[1].kind);

        assert_eq!("exit", body_syms[2].name);
        assert_eq!(SymbolKind::Function, body_syms[2].kind);

        let add_fn = typer.ast.find_function("add").unwrap();
        let add_syms: Vec<&TypedSymbol> = typer
            .symbols
            .iter()
            .filter(|s| s.scope == add_fn.body)
            .collect();

        assert_eq!(3, add_syms.len());
        assert_eq!("z", add_syms[0].name);
        assert_eq!(SymbolKind::Local, add_syms[0].kind);
        assert_eq!("x", add_syms[1].name);
        assert_eq!(SymbolKind::Local, add_syms[1].kind);
        assert_eq!("y", add_syms[2].name);
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
        type person = { name: string };
        var x = person { name: 1 };
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_extranous_struct_member() {
        let code = r###"
        type person = { name: string };
        var x = person { name: "hello!", age: 5 };
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_missing_struct_members() {
        let code = r###"
        type person = { name: string };
        var x = person {};
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_accept_member_access_of_implicit_pointer() {
        let code = r###"
        type person = { name: string };
        fun takes(x: &person): void {
            var y = x.name;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_accept_add_of_scalar_contained_in_struct() {
        let code = r###"
        type person = { age: int };
        fun takes(x: &person): int {
            return x.age + 1;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_accept_assignment_of_scalar_pointer_to_scalar() {
        let code = r###"
        type person = { age: int };
        fun takes(x: &person): void {
            var y: int = x.age;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_reject_return_of_struct() {
        let code = r###"
        type person = { age: int };
        fun takes(): person {
            var x = person { age: 5 };
            return x;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_reject_symbol_reference_with_declare_after_use() {
        let code = r###"
        fun takes(): void {
            var x = y;
            var y = 5;
        }
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(false, ok);
    }

    #[test]
    fn should_allow_types_to_be_used_before_declared() {
        let code = r###"
        fun takes(): void {
            var x = person { age: 420 };
        }
        type person = { age: int };
        "###;

        let chk = Typer::from_code(code).unwrap();
        let ok = chk.check().is_ok();

        assert_eq!(true, ok);
    }

    #[test]
    fn should_not_equate_pointers_to_scalar_types() {
        let type_int = Type::Int;
        let type_ptr_to_int = Type::Pointer(Box::new(type_int.clone()));

        assert_ne!(type_int, type_ptr_to_int);
    }

    #[test]
    fn should_reject_binary_expr_with_mismatched_types() {
        let code = r###"
        5 + 3 * (5 == 20);
        "###;

        do_test(false, code);
    }

    fn do_test(expected: bool, code: &str) {
        let typer = Typer::from_code(code).unwrap();
        let is_ok = typer.check().is_ok();

        assert_eq!(expected, is_ok);
    }
}
