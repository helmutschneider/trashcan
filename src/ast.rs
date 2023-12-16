use std::collections::HashMap;

use crate::binexpr;
use crate::tokenizer::{tokenize, Token, TokenKind};
use crate::util::report_error;
use crate::util::Error;
use crate::util::SourceLocation;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StatementId(i64);

impl std::fmt::Display for StatementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct AST {
    pub root: Rc<Statement>,
    pub statements: Vec<Rc<Statement>>,
    pub symbols: Vec<UntypedSymbol>,
}

impl std::fmt::Display for AST {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = crate::util::format_stmt(&self.root, 0);
        return f.write_str(&s);
    }
}

impl AST {
    pub fn find_statement(&self, at: StatementId) -> Option<&Rc<Statement>> {
        for stmt in &self.statements {
            let id = stmt.id();
            if id == at {
                return Some(stmt);
            }
        }
        return None;
    }

    pub fn find_function(&self, name: &str) -> Option<&Function> {
        let root = self.find_statement(StatementId(0))?;

        for stmt in &root.as_block().statements {
            if let Statement::Function(fx) = stmt.as_ref() {
                if fx.name_token.value == name {
                    return Some(fx);
                }
            }
        }

        return None;
    }

    pub fn body(&self) -> &Block {
        if let Statement::Block(b) = self.root.as_ref() {
            return b;
        }
        panic!("body is not a block");
    }

    pub fn from_code(code: &str) -> Result<Self, Error> {
        let tokens = crate::tokenizer::tokenize(code)?;
        let mut builder = ASTBuilder {
            tokens: tokens.to_vec(),
            token_index: 0,
            source: code.to_string(),
            statements: Vec::new(),
            symbols: Vec::new(),
            num_statements: 0,
        };

        let root_id = builder.get_and_increment_statement_id();
        let mut root = Block {
            id: root_id,
            statements: Vec::new(),
            parent: None,
        };

        while !builder.is_end_of_file() {
            let prev_index = builder.token_index;

            let stmt = builder.expect_statement(root_id)?;
            root.statements.push(stmt);

            if builder.token_index == prev_index {
                let token = &builder.tokens[builder.token_index];
                let loc = SourceLocation::Token(token);
                return report_error(code, "a token was not consumed while building the AST", loc);
            }
        }

        let stmt = builder.add_statement(Statement::Block(root));
        let ast = AST {
            root: stmt,
            symbols: builder.symbols,
            statements: builder.statements,
        };

        return Result::Ok(ast);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolKind {
    Local,
    Function,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub i64);

#[derive(Debug, Clone)]
pub struct UntypedSymbol {
    pub id: SymbolId,
    pub name: String,
    pub kind: SymbolKind,
    pub declared_at: Rc<Statement>,
    pub scope: StatementId,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Function(Function),
    Variable(Variable),
    Expression(ExpressionStatement),
    Return(Return),
    Block(Block),
    If(If),
    While(While),
    Type(Struct),
}

impl PartialEq for Statement {
    fn eq(&self, other: &Self) -> bool {
        return self.id() == other.id();
    }
}

impl Statement {
    pub fn id(&self) -> StatementId {
        return match self {
            Self::Function(fx) => fx.id,
            Self::Variable(var) => var.id,
            Self::Expression(expr) => expr.id,
            Self::Return(ret) => ret.id,
            Self::Block(b) => b.id,
            Self::If(if_) => if_.id,
            Self::While(while_) => while_.id,
            Self::Type(s) => s.id,
        };
    }

    pub fn parent_id(&self) -> Option<StatementId> {
        return match self {
            Self::Function(fx) => Some(fx.parent),
            Self::Variable(v) => Some(v.parent),
            Self::Expression(expr) => Some(expr.parent),
            Self::Return(ret) => Some(ret.parent),
            Self::Block(b) => b.parent,
            Self::If(if_expr) => Some(if_expr.parent),
            Self::While(while_) => Some(while_.parent),
            Self::Type(struct_) => Some(struct_.parent),
        };
    }

    pub fn as_block(&self) -> &Block {
        return match self {
            Self::Block(b) => b,
            _ => panic!("statement is not a block:\n{:?}", self),
        };
    }
}

#[derive(Debug, Clone)]
pub struct Return {
    pub id: StatementId,
    pub expr: Expression,
    pub token: Token,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Void,
    Identifier(Identifier),
    IntegerLiteral(IntegerLiteral),
    StringLiteral(StringLiteral),
    FunctionCall(FunctionCall),
    BinaryExpr(BinaryExpr),
    StructLiteral(StructLiteral),
    MemberAccess(MemberAccess),
    BooleanLiteral(BooleanLiteral),
    UnaryPrefix(UnaryPrefix),
    ArrayLiteral(ArrayLiteral),
    ElementAccess(ElementAccess),
}

#[derive(Debug, Clone)]
pub struct IntegerLiteral {
    pub value: i64,
    pub token: Token,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct StringLiteral {
    pub value: String,
    pub token: Token,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct Identifier {
    pub name: String,
    pub token: Token,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct StructLiteral {
    pub name_token: Token,
    pub members: Vec<StructLiteralMember>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct StructLiteralMember {
    pub field_name_token: Token,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct MemberAccess {
    pub left: Box<Expression>,
    pub right: Identifier,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct ElementAccess {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct FunctionArgument {
    pub name_token: Token,
    pub type_: TypeDecl,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub id: StatementId,
    pub name_token: Token,
    pub arguments: Vec<FunctionArgument>,
    pub body: Rc<Statement>,
    pub return_type: TypeDecl,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: StatementId,
    pub statements: Vec<Rc<Statement>>,
    pub parent: Option<StatementId>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub id: StatementId,
    pub name_token: Token,
    pub type_: Option<TypeDecl>,
    pub initializer: Expression,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name_token: Token,
    pub arguments: Vec<Expression>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct If {
    pub id: StatementId,
    pub condition: Expression,
    pub block: Rc<Statement>,
    pub else_: Option<Rc<Statement>>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub kind: TypeDeclKind,
    pub parent: StatementId,
}

impl TypeDecl {
    pub fn identifier_token(&self) -> Token {
        return match &self.kind {
            TypeDeclKind::Name(tok) => tok.clone(),
            TypeDeclKind::Pointer(inner) => inner.identifier_token(),
            TypeDeclKind::Array(elem, _) => elem.identifier_token(),
        };
    }
}

#[derive(Debug, Clone)]
pub enum TypeDeclKind {
    Name(Token),
    Pointer(Box<TypeDecl>),
    Array(Box<TypeDecl>, Option<i64>),
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub id: StatementId,
    pub name_token: Token,
    pub members: Vec<StructMember>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct StructMember {
    pub field_name_token: Token,
    pub type_: TypeDecl,
}

#[derive(Debug, Clone)]
pub struct BooleanLiteral {
    pub value: bool,
    pub token: Token,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct ExpressionStatement {
    pub id: StatementId,
    pub expr: Expression,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct While {
    pub id: StatementId,
    pub condition: Expression,
    pub block: Rc<Statement>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct UnaryPrefix {
    pub operator: Token,
    pub expr: Box<Expression>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct ArrayLiteral {
    pub elements: Vec<Expression>,
    pub parent: StatementId,
}

pub struct ASTBuilder {
    pub tokens: Vec<Token>,
    pub token_index: usize,
    source: String,
    statements: Vec<Rc<Statement>>,
    symbols: Vec<UntypedSymbol>,
    num_statements: i64,
}

impl ASTBuilder {
    pub fn peek(&self) -> Result<TokenKind, Error> {
        return self.peek_at(0);
    }

    fn peek_at(&self, offset: usize) -> Result<TokenKind, Error> {
        let tok = self.tokens.get(self.token_index + offset);

        return match tok {
            Some(tok) => Ok(tok.kind),
            None => Err("attempted to peek beyond the end of the file".to_string()),
        };
    }

    fn is_end_of_file(&self) -> bool {
        return self.tokens.get(self.token_index).is_none();
    }

    fn get_and_increment_statement_id(&mut self) -> StatementId {
        let id = StatementId(self.num_statements);
        self.num_statements += 1;
        return id;
    }

    fn get_next_symbol_id(&mut self) -> SymbolId {
        let id = SymbolId(self.symbols.len() as i64);
        return id;
    }

    fn add_statement(&mut self, stmt: Statement) -> Rc<Statement> {
        let rced = Rc::new(stmt);
        self.statements.push(Rc::clone(&rced));
        return rced;
    }

    fn expect(&mut self, expected_kind: TokenKind) -> Result<Token, Error> {
        if let Some(token) = self.tokens.get(self.token_index) {
            if expected_kind != token.kind {
                let message = format!("expected {}, found {}", expected_kind, token.kind);
                return report_error(&self.source, &message, SourceLocation::Token(token));
            }

            self.token_index += 1;
            return Result::Ok(token.clone());
        }

        return report_error(
            &self.source,
            &format!("expected {}, found end of file", expected_kind),
            SourceLocation::None,
        );
    }

    pub fn consume_one_token(&mut self) -> Result<Token, Error> {
        if let Some(token) = self.tokens.get(self.token_index) {
            self.token_index += 1;
            return Result::Ok(token.clone());
        }

        return report_error(
            &self.source,
            "expected a token, found end of file",
            SourceLocation::None,
        );
    }

    fn expect_function_argument(&mut self, parent: StatementId) -> Result<FunctionArgument, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_ = self.expect_type_decl(parent)?;
        let arg = FunctionArgument {
            name_token: name_token,
            type_: type_,
            parent: parent,
        };
        return Result::Ok(arg);
    }

    fn expect_type_decl(&mut self, parent: StatementId) -> Result<TypeDecl, Error> {
        let kind = self.peek()?;
        let type_ = match kind {
            TokenKind::Identifier => {
                let tok = self.consume_one_token()?;
                TypeDecl {
                    kind: TypeDeclKind::Name(tok.clone()),
                    parent: parent,
                }
            }
            TokenKind::Ampersand => {
                self.consume_one_token()?;
                let inner = self.expect_type_decl(parent)?;
                TypeDecl {
                    kind: TypeDeclKind::Pointer(Box::new(inner)),
                    parent: parent,
                }
            }
            TokenKind::OpenBracket => {
                self.consume_one_token()?;
                let elem_type = self.expect_type_decl(parent)?;
                let mut array_size: Option<i64> = None;

                if self.peek()? == TokenKind::Semicolon {
                    self.consume_one_token()?;
                    let tok = self.expect(TokenKind::IntegerLiteral)?;
                    let parsed: i64 = tok.value.parse().unwrap();
                    array_size = Some(parsed);
                }

                self.expect(TokenKind::CloseBracket)?;

                TypeDecl {
                    kind: TypeDeclKind::Array(Box::new(elem_type), array_size),
                    parent: parent,
                }
            }
            _ => {
                let message = format!("expected type identifier, got '{}'", kind);
                let tok = &self.tokens[self.token_index];
                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };
        return Ok(type_);
    }

    fn expect_block(&mut self, parent: StatementId) -> Result<Rc<Statement>, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let id = self.get_and_increment_statement_id();
        let mut block = Block {
            id: id,
            statements: Vec::new(),
            parent: Some(parent),
        };

        while self.peek()? != TokenKind::CloseBrace {
            let stmt = self.expect_statement(id)?;
            block.statements.push(stmt);
        }

        self.expect(TokenKind::CloseBrace)?;
        let stmt = self.add_statement(Statement::Block(block));

        return Result::Ok(stmt);
    }

    fn expect_statement(&mut self, parent: StatementId) -> Result<Rc<Statement>, Error> {
        let stmt_index = match self.peek()? {
            TokenKind::FunctionKeyword => {
                let stmt = self.expect_function(parent)?;
                stmt
            }
            TokenKind::VariableKeyword => {
                let stmt = self.expect_variable(parent)?;
                self.expect(TokenKind::Semicolon)?;
                stmt
            }
            TokenKind::ReturnKeyword => {
                let token = self.expect(TokenKind::ReturnKeyword)?;
                let id = self.get_and_increment_statement_id();
                let expr = match self.peek()? {
                    TokenKind::Semicolon => Expression::Void,
                    _ => self.expect_expression(id, false, false)?,
                };
                self.expect(TokenKind::Semicolon)?;

                let ret = Return {
                    id: id,
                    expr: expr,
                    token: token,
                    parent: parent,
                };
                let stmt = self.add_statement(Statement::Return(ret));

                stmt
            }
            TokenKind::IfKeyword => {
                let if_id = self.get_and_increment_statement_id();
                let if_token = self.consume_one_token()?;

                let condition = self.expect_expression(if_id, false, true)?;
                let block = self.expect_block(if_id)?;
                let mut else_: Option<Rc<Statement>> = None;

                if self.peek() == Ok(TokenKind::ElseKeyword) && self.peek_at(1) == Ok(TokenKind::IfKeyword) {
                    self.consume_one_token()?;
                    else_ = Some(self.expect_statement(if_id)?);
                }

                if self.peek() == Ok(TokenKind::ElseKeyword) && self.peek_at(1) == Ok(TokenKind::OpenBrace) {
                    self.consume_one_token()?;
                    else_ = Some(self.expect_block(if_id)?);
                }

                let if_ = If {
                    id: if_id,
                    condition: condition,
                    block: block,
                    else_: else_,
                    parent: parent,
                };

                let stmt = self.add_statement(Statement::If(if_));
                stmt
            }
            TokenKind::WhileKeyword => {
                let while_id = self.get_and_increment_statement_id();
                let while_token = self.consume_one_token()?;

                let condition = self.expect_expression(while_id, false, true)?;
                let block = self.expect_block(while_id)?;

                let while_ = While {
                    id: while_id,
                    condition: condition,
                    block: block,
                    parent: parent,
                };

                let stmt = self.add_statement(Statement::While(while_));
                stmt
            }
            TokenKind::OpenBrace => {
                let stmt = self.expect_block(parent)?;
                stmt
            }
            TokenKind::TypeKeyword => {
                let struct_index = self.expect_type(parent)?;
                struct_index
            }
            _ => {
                let expr_id = self.get_and_increment_statement_id();
                let expr = self.expect_expression(expr_id, false, false)?;
                self.expect(TokenKind::Semicolon)?;
                let expr_stmt = ExpressionStatement {
                    id: expr_id,
                    expr: expr,
                    parent: parent,
                };
                let stmt = self.add_statement(Statement::Expression(expr_stmt));
                stmt
            }
        };

        // eat extranous semicolons between statements, if any.
        while !self.is_end_of_file() {
            let is_semi = self.peek() == Ok(TokenKind::Semicolon);
            if !is_semi {
                break;
            }
            self.consume_one_token()?;
        }

        return Result::Ok(stmt_index);
    }

    pub fn expect_expression(&mut self, parent: StatementId, is_reading_binary_expr: bool, is_reading_control_flow_condition: bool) -> Result<Expression, Error> {
        let kind = self.peek()?;
        let index_at_start_of_expression = self.token_index;

        if binexpr::token_is_prefix_operator_or_paren(kind) {
            // this happens if we're starting an expression with a unary operator.
            return binexpr::do_shunting_yard(self, parent, None, index_at_start_of_expression, is_reading_control_flow_condition);
        }

        let expr = match kind {
            TokenKind::Identifier => {
                let next_plus_one = self.peek_at(1)?;

                match next_plus_one {
                    // the struct literal collides with the if-statement, because they
                    // both accept an identifier followed by an opening brace. we would need
                    // some kind of contextual parsing to resolve that. the workaround for now
                    // is to just not allow struct literals in control flow contitions.
                    //   -johan, 2023-12-02
                    TokenKind::OpenBrace if !is_reading_control_flow_condition => {
                        let struct_init = self.expect_struct_literal(parent)?;
                        Expression::StructLiteral(struct_init)
                    }
                    _ => {
                        let ident = self.expect_identifier(parent)?;
                        Expression::Identifier(ident)
                    }
                }
            }
            TokenKind::IntegerLiteral => {
                let token = self.consume_one_token()?;
                let parsed: i64 = token.value.parse().unwrap();
                Expression::IntegerLiteral(IntegerLiteral {
                    value: parsed,
                    token: token,
                    parent: parent,
                })
            }
            TokenKind::StringLiteral => {
                let token = self.consume_one_token()?;
                Expression::StringLiteral(StringLiteral {
                    value: token.value.clone(),
                    token: token,
                    parent: parent,
                })
            }
            TokenKind::TrueKeyword => {
                let token = self.consume_one_token()?;
                let expr = BooleanLiteral {
                    value: true,
                    parent: parent,
                    token: token,
                };
                Expression::BooleanLiteral(expr)
            }
            TokenKind::FalseKeyword => {
                let token = self.consume_one_token()?;
                let expr = BooleanLiteral {
                    value: false,
                    parent: parent, 
                    token: token,
                };
                Expression::BooleanLiteral(expr)
            }
            TokenKind::OpenBracket => {
                let array_literal = self.expect_array_literal(parent)?;
                Expression::ArrayLiteral(array_literal)
            }
            _ => {
                let message = format!("expected expression, got token '{}'", kind);
                let tok = &self.tokens[self.token_index];

                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };

        let mut actual_expr = expr;

        if !is_reading_binary_expr && !self.is_end_of_file() && binexpr::token_is_binary_or_postfix_operator_or_paren(self.peek()?) {
            actual_expr = binexpr::do_shunting_yard(self, parent, Some(actual_expr), index_at_start_of_expression, is_reading_control_flow_condition)?;
        }

        return Result::Ok(actual_expr);
    }

    fn expect_identifier(&mut self, parent: StatementId) -> Result<Identifier, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        let ident = Identifier {
            name: name_token.value.clone(),
            token: name_token,
            parent: parent,
        };
        return Ok(ident);
    }

    fn expect_variable(&mut self, parent: StatementId) -> Result<Rc<Statement>, Error> {
        self.expect(TokenKind::VariableKeyword)?;
        let var_id = self.get_and_increment_statement_id();
        let name_token = self.expect(TokenKind::Identifier)?;
        let type_ = match self.peek()? {
            TokenKind::Colon => {
                self.consume_one_token()?;
                let t = self.expect_type_decl(var_id)?;
                Some(t)
            }
            _ => None,
        };

        self.expect(TokenKind::Equals)?;

        let init_expr = self.expect_expression(var_id, false, false)?;

        let var = Variable {
            id: var_id,
            name_token: name_token.clone(),
            type_: type_,
            initializer: init_expr,
            parent: parent,
        };

        let stmt = self.add_statement(Statement::Variable(var));
        let sym = UntypedSymbol {
            id: self.get_next_symbol_id(),
            name: name_token.value.clone(),
            kind: SymbolKind::Local,
            declared_at: Rc::clone(&stmt),
            scope: parent,
        };
        self.symbols.push(sym);

        return Result::Ok(stmt);
    }

    fn expect_function(&mut self, parent: StatementId) -> Result<Rc<Statement>, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let fx_id = self.get_and_increment_statement_id();
        let mut arguments: Vec<FunctionArgument> = Vec::new();

        while self.peek()? != TokenKind::CloseParenthesis {
            let arg = self.expect_function_argument(fx_id)?;
            arguments.push(arg);

            if self.peek()? != TokenKind::CloseParenthesis {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        self.expect(TokenKind::Colon)?;
        let return_type = self.expect_type_decl(fx_id)?;
        let body = self.expect_block(fx_id)?;
        let body_id = body.id();

        let fx = Function {
            id: fx_id,
            name_token: name_token.clone(),
            arguments: arguments.clone(),
            body: Rc::clone(&body),
            return_type: return_type,
            parent: parent,
        };

        let stmt = self.add_statement(Statement::Function(fx));

        let sym = UntypedSymbol {
            id: self.get_next_symbol_id(),
            name: name_token.value.clone(),
            kind: SymbolKind::Function,
            declared_at: Rc::clone(&stmt),
            scope: parent,
        };
        self.symbols.push(sym);

        for fx_arg in arguments {
            let arg_sym = UntypedSymbol {
                id: self.get_next_symbol_id(),
                name: fx_arg.name_token.value,
                kind: SymbolKind::Local,
                declared_at: Rc::clone(&stmt),
                scope: body_id,
            };
            self.symbols.push(arg_sym);
        }

        return Ok(stmt);
    }

    fn expect_type(&mut self, parent: StatementId) -> Result<Rc<Statement>, Error> {
        assert_eq!(TokenKind::TypeKeyword, self.peek()?);

        let struct_id = self.get_and_increment_statement_id();

        self.consume_one_token()?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Equals)?;

        // TODO: allow other kinds of types... like enums.
        self.expect(TokenKind::OpenBrace)?;

        let mut members: Vec<StructMember> = Vec::new();

        while self.peek()? != TokenKind::CloseBrace {
            let field_name = self.expect(TokenKind::Identifier)?;
            self.expect(TokenKind::Colon)?;
            let type_ = self.expect_type_decl(struct_id)?;

            members.push(StructMember {
                field_name_token: field_name,
                type_: type_,
            });

            if self.peek()? != TokenKind::CloseBrace {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseBrace)?;
        let struct_ = Struct {
            id: struct_id,
            name_token: name_token.clone(),
            members: members,
            parent: parent,
        };
        let stmt = self.add_statement(Statement::Type(struct_));
        let sym = UntypedSymbol {
            id: self.get_next_symbol_id(),
            name: name_token.value.clone(),
            kind: SymbolKind::Type,
            declared_at: Rc::clone(&stmt),
            scope: parent,
        };
        self.symbols.push(sym);

        return Ok(stmt);
    }

    fn expect_struct_literal(&mut self, parent: StatementId) -> Result<StructLiteral, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenBrace)?;

        let mut member_inits: Vec<StructLiteralMember> = Vec::new();

        while self.peek()? != TokenKind::CloseBrace {
            let field_name_token = self.expect(TokenKind::Identifier)?;
            self.expect(TokenKind::Colon)?;

            let init_expr = self.expect_expression(parent, false, false)?;

            member_inits.push(StructLiteralMember {
                field_name_token: field_name_token,
                value: init_expr,
            });

            if self.peek()? != TokenKind::CloseBrace {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseBrace)?;

        let struct_ = StructLiteral {
            name_token: name_token,
            members: member_inits,
            parent: parent,
        };
        return Ok(struct_);
    }

    fn expect_array_literal(&mut self, parent: StatementId) -> Result<ArrayLiteral, Error> {
        self.expect(TokenKind::OpenBracket)?;

        let mut array_elems: Vec<Expression> = Vec::new();

        while self.peek()? != TokenKind::CloseBracket {
            let elem = self.expect_expression(parent, false, false)?;

            array_elems.push(elem);

            if self.peek()? != TokenKind::CloseBracket {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseBracket)?;

        let array_literal = ArrayLiteral {
            elements: array_elems,
            parent: parent
        };

        return Ok(array_literal);
    }
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use crate::ast::*;
    use crate::tokenizer::*;

    #[test]
    fn should_create_ast_from_function_with_empty_body() {
        let code = "fun do_thing(x: int, y: double): void {}";
        let ast = AST::from_code(&code).unwrap();
        let fx = ast.find_function("do_thing").unwrap();

        assert_eq!(2, fx.arguments.len());
        assert_eq!("x", fx.arguments[0].name_token.value);

        let type_ = &fx.arguments[0].type_;

        if let TypeDeclKind::Name(tok) = &type_.kind {
            assert_eq!("int", tok.value);
        } else {
            panic!();
        }

        assert_eq!("y", fx.arguments[1].name_token.value);

        let type_ = &fx.arguments[1].type_;

        if let TypeDeclKind::Name(tok) = &type_.kind {
            assert_eq!("double", tok.value);
        } else {
            panic!();
        }

        let type_ = &fx.return_type;

        if let TypeDeclKind::Name(tok) = &type_.kind {
            assert_eq!("void", tok.value);
        } else {
            panic!();
        }

        let fx_body = fx.body.as_block();
        assert_eq!(0, fx_body.statements.len());
    }

    #[test]
    fn should_create_ast_from_variable_list() {
        let code = r###"
        var x: int = 1;
        var y: double = 2;
        "###;

        let ast = AST::from_code(code).unwrap();

        assert_eq!(2, ast.body().statements.len());

        if let Statement::Variable(x) = ast.root.as_block().statements[0].as_ref() {
            assert_eq!("x", x.name_token.value);

            let type_ = &x.type_.as_ref().unwrap();

            if let TypeDeclKind::Name(tok) = &type_.kind {
                assert_eq!("int", tok.value);
            } else {
                panic!();
            }

            if let Expression::IntegerLiteral(init) = &x.initializer {
                assert_eq!(1, init.value);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }

        if let Statement::Variable(x) = ast.root.as_block().statements[1].as_ref() {
            assert_eq!("y", x.name_token.value);

            let type_ = &x.type_.as_ref().unwrap();

            if let TypeDeclKind::Name(tok) = &type_.kind {
                assert_eq!("double", tok.value);
            } else {
                panic!();
            }

            if let Expression::IntegerLiteral(init) = &x.initializer {
                assert_eq!(2, init.value);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_create_ast_from_call_expression() {
        let code = "call_me_maybe(5, thing);";
        let ast = AST::from_code(code).unwrap();

        dbg!("{:?}", &ast);

        if let Statement::Expression(expr) = ast.root.as_block().statements[0].as_ref() {
            if let Expression::FunctionCall(call) = &expr.expr {
                assert_eq!("call_me_maybe", call.name_token.value);
                assert_eq!(2, call.arguments.len());
            } else {
                panic!();
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_create_ast_from_main() {
        let code = r###"
            fun main(): void {
                var x: int = 5;
                var y: int = x;
                println(x, y, 42);
            }
        "###;

        let ast = AST::from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let fx = ast.find_function("main").unwrap();
        assert_eq!("main", fx.name_token.value);

        assert_eq!(3, fx.body.as_block().statements.len());
    }

    #[test]
    fn should_create_ast_from_binary_expr() {
        let code = r###"
            var x: int = 1 + do_thing(420, 69);
        "###;

        let ast = AST::from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let stmt = ast.body().statements[0].as_ref();

        if let Statement::Variable(v) = stmt {
            if let Expression::BinaryExpr(expr) = &v.initializer {
                assert_eq!(TokenKind::Plus, expr.operator.kind);
                assert_eq!(true, matches!(*expr.left, Expression::IntegerLiteral(_)));
                assert_eq!(true, matches!(*expr.right, Expression::FunctionCall(_)));
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_create_ast_from_two_binary_exprs_in_sequence() {
        let code = r###"
            var x: int = 1 + 2 + 3;
        "###;

        let ast = AST::from_code(code).unwrap();
        let stmt = ast.body().statements[0].as_ref();

        if let Statement::Variable(v) = stmt {
            if let Expression::BinaryExpr(expr) = &v.initializer {
                assert_eq!(true, matches!(*expr.left, Expression::BinaryExpr(_)));
                assert_eq!(true, matches!(*expr.right, Expression::IntegerLiteral(_)));
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_parse_if() {
        let code = r###"
        if 1 == 2 {
            var x: int = 1;
        }
        "###;
        let ast = AST::from_code(code).unwrap();

        let if_stmt = match ast.body().statements[0].as_ref() {
            Statement::If(x) => x,
            _ => {
                assert!(false);
                panic!();
            }
        };

        let bin_expr = match &if_stmt.condition {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::EqualsEquals, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            assert!(false);
        }

        if let Expression::IntegerLiteral(x) = bin_expr.right.as_ref() {
            assert_eq!(2, x.value);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_report_error_for_missing_type_declaration() {
        let code = "fun add(x: int, y) {}";
        let ast = AST::from_code(code);

        assert!(ast.is_err());
    }

    #[test]
    fn should_allow_if_statement_without_boolean_expr() {
        let code = "if 1 {}";
        let ast = AST::from_code(code);

        assert_eq!(true, ast.is_ok());
    }

    #[test]
    fn should_parse_struct_type() {
        let code = r###"
        type person = {
            name: string,
            age: int,
        }

        fun thing(x: person): void {}
        "###;
        let ast = AST::from_code(code).unwrap();
        let struct_ = match ast.root.as_block().statements[0].as_ref() {
            Statement::Type(s) => s,
            _ => panic!(),
        };
        assert_eq!("person", struct_.name_token.value);
        assert_eq!(2, struct_.members.len());

        assert_eq!("name", struct_.members[0].field_name_token.value);
        if let TypeDeclKind::Name(tok) = &struct_.members[0].type_.kind {
            assert_eq!("string", tok.value);
        } else {
            panic!();
        }

        assert_eq!("age", struct_.members[1].field_name_token.value);
        if let TypeDeclKind::Name(tok) = &struct_.members[1].type_.kind {
            assert_eq!("int", tok.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_struct_initializer() {
        let code = r###"
        var x = person {
            name: "yee",
            age: 5,
        };
        "###;
        let ast = AST::from_code(code).unwrap();
        let x = match ast.root.as_block().statements[0].as_ref() {
            Statement::Variable(x) => x,
            _ => panic!(),
        };
        let struct_init = match &x.initializer {
            Expression::StructLiteral(s) => s,
            _ => panic!(),
        };
        assert_eq!("person", struct_init.name_token.value);
        assert_eq!(2, struct_init.members.len());
        assert_eq!("name", struct_init.members[0].field_name_token.value);

        if let Expression::StringLiteral(s) = &struct_init.members[0].value {
            assert_eq!("yee", s.value);
        } else {
            panic!();
        }

        assert_eq!("age", struct_init.members[1].field_name_token.value);

        if let Expression::IntegerLiteral(i) = &struct_init.members[1].value {
            assert_eq!(5, i.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_allow_extra_semicolons_between_statements() {
        let code = r###"
            var x = 1;;;;
            var y = 5;;;
            fun thing(): void {};
        "###;
        let ast = AST::from_code(code);

        assert_eq!(true, ast.is_ok());
    }

    #[test]
    fn should_parse_property_access() {
        let code = r###"
        var x = box { value: 1 };
        var y = x.value;
        "###;

        let ast = AST::from_code(code).unwrap();

        if let Statement::Variable(x) = ast.root.as_block().statements[1].as_ref() {
            if let Expression::MemberAccess(y) = &x.initializer {
                if let Expression::Identifier(left) = y.left.as_ref() {
                    assert_eq!("x", left.name);
                } else {
                    panic!();
                }
                assert_eq!("value", y.right.name);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_deep_property_access() {
        let code = r###"
        type A = { value: int };
        type B = { a: A };
        var x = B { a: A { value: 5 } };
        var y = x.a.value;
        "###;

        let ast = AST::from_code(code).unwrap();

        if let Statement::Variable(a) = ast.root.as_block().statements[3].as_ref() {
            if let Expression::MemberAccess(b) = &a.initializer {
                assert_eq!("value", b.right.name);

                if let Expression::MemberAccess(c) = b.left.as_ref() {
                    assert_eq!("a", c.right.name);

                    if let Expression::Identifier(d) = c.left.as_ref() {
                        assert_eq!("x", d.name);
                    } else {
                        panic!();
                    }
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_if_else() {
        let code = r###"
        if 1 {

        } else if 2 {

        }
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.body();

        assert_eq!(1, body.statements.len());

        let maybe_if = ast.root.as_block().statements[0].as_ref();

        if let Statement::If(if_) = maybe_if {
            if let Expression::IntegerLiteral(cond) = &if_.condition {
                assert_eq!(1, cond.value);
            } else {
                panic!();
            }

            let else_ = &if_.else_;

            if let Statement::If(else_if) = else_.as_ref().unwrap().as_ref() {
                if let Expression::IntegerLiteral(cond) = &else_if.condition {
                    assert_eq!(2, cond.value);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
    
    #[test]
    fn should_parse_boolean_literals() {
        let code = r###"
        var x = true;
        var y = false;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.body();

        assert_eq!(2, body.statements.len());

        if let Statement::Variable(x) = ast.root.as_block().statements[0].as_ref() {
            if let Expression::BooleanLiteral(y) = &x.initializer {
                assert_eq!(true, y.value);
            } else {
                panic!();
            }
        } else {
            panic!();
        }

        if let Statement::Variable(x) = ast.root.as_block().statements[1].as_ref() {
            if let Expression::BooleanLiteral(y) = &x.initializer {
                assert_eq!(false, y.value);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_1() {
        let code = r###"
        1 + 2 == 3;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::EqualsEquals, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.right.as_ref() {
            assert_eq!(3, x.value);
        } else {
            panic!();
        }

        let left = match bin_expr.left.as_ref() {
            Expression::BinaryExpr(x) => x,
            _ => panic!()
        };

        assert_eq!(TokenKind::Plus, left.operator.kind);
        
        if let Expression::IntegerLiteral(x) = left.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            panic!();
        }

        if let Expression::IntegerLiteral(x) = left.right.as_ref() {
            assert_eq!(2, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_2() {
        let code = r###"
        1 * 2 + 3;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::Plus, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.right.as_ref() {
            assert_eq!(3, x.value);
        } else {
            panic!();
        }

        let left = match bin_expr.left.as_ref() {
            Expression::BinaryExpr(x) => x,
            _ => panic!()
        };

        assert_eq!(TokenKind::Asterisk, left.operator.kind);
        
        if let Expression::IntegerLiteral(x) = left.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            panic!();
        }

        if let Expression::IntegerLiteral(x) = left.right.as_ref() {
            assert_eq!(2, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_3() {
        let code = r###"
        (1 + 2) * 3;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::Asterisk, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.right.as_ref() {
            assert_eq!(3, x.value);
        } else {
            panic!();
        }

        let left = match bin_expr.left.as_ref() {
            Expression::BinaryExpr(x) => x,
            _ => panic!()
        };

        assert_eq!(TokenKind::Plus, left.operator.kind);
        
        if let Expression::IntegerLiteral(x) = left.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            panic!();
        }

        if let Expression::IntegerLiteral(x) = left.right.as_ref() {
            assert_eq!(2, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_4() {
        let code = r###"
        1 * (2 + 3);
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::Asterisk, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            panic!();
        }

        let left = match bin_expr.right.as_ref() {
            Expression::BinaryExpr(x) => x,
            _ => panic!()
        };

        assert_eq!(TokenKind::Plus, left.operator.kind);
        
        if let Expression::IntegerLiteral(x) = left.left.as_ref() {
            assert_eq!(2, x.value);
        } else {
            panic!();
        }

        if let Expression::IntegerLiteral(x) = left.right.as_ref() {
            assert_eq!(3, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_5() {
        let code = r###"
        5 + 3 * 5 == 20;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::EqualsEquals, bin_expr.operator.kind);
    }

    #[test]
    fn should_respect_operator_precedence_6() {
        let code = r###"
        x.name + 5;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::Plus, bin_expr.operator.kind);

        if let Expression::MemberAccess(x) = bin_expr.left.as_ref() {
            assert_eq!("name", x.right.name);
        } else {
            panic!();
        }

        if let Expression::IntegerLiteral(x) = bin_expr.right.as_ref() {
            assert_eq!(5, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_7() {
        let code = r###"
        6 * -7;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        assert_eq!(TokenKind::Asterisk, bin_expr.operator.kind);

        if let Expression::IntegerLiteral(x) = bin_expr.left.as_ref() {
            assert_eq!(6, x.value);
        } else {
            panic!();
        }

        if let Expression::UnaryPrefix(unary) = bin_expr.right.as_ref() {
            assert_eq!(TokenKind::Minus, unary.operator.kind);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_8() {
        let code = r###"
        a(7);
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let fn_ = match expr {
            Expression::FunctionCall(x) => x,
            _ => panic!(),
        };

        assert_eq!("a", fn_.name_token.value);
        assert_eq!(1, fn_.arguments.len());

        if let Expression::IntegerLiteral(x) = &fn_.arguments[0] {
            assert_eq!(7, x.value);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_9() {
        let code = r###"
        takes_str(&"yee!");
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let fn_ = match expr {
            Expression::FunctionCall(x) => x,
            _ => panic!(),
        };

        assert_eq!("takes_str", fn_.name_token.value);
        assert_eq!(1, fn_.arguments.len());

        if let Expression::UnaryPrefix(x) = &fn_.arguments[0] {
            assert_eq!(TokenKind::Ampersand, x.operator.kind);
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_10() {
        let code = r###"
        a[420 + 69];
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let elem_access = match expr {
            Expression::ElementAccess(x) => x,
            _ => panic!(),
        };

        if let Expression::Identifier(x) = elem_access.left.as_ref() {
            assert_eq!("a", x.name);
        } else {
            panic!();
        }

        if let Expression::BinaryExpr(x) = elem_access.right.as_ref() {
            if let Expression::IntegerLiteral(y) = x.left.as_ref() {
                assert_eq!(420, y.value);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_respect_operator_precedence_11() {
        let code = r###"
        1 + do_thing(420, 69);
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        if let Expression::IntegerLiteral(x) = bin_expr.left.as_ref() {
            assert_eq!(1, x.value);
        } else {
            panic!();
        }

        if let Expression::FunctionCall(fx) = bin_expr.right.as_ref() {
            assert_eq!("do_thing", fx.name_token.value);
            assert_eq!(2, fx.arguments.len());
        }
    }


    #[test]
    fn should_respect_operator_precedence_12() {
        let code = r###"
        return *x.age + 1;
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Return(x) => &x.expr,
            _ => panic!(),
        };
        let bin_expr = match expr {
            Expression::BinaryExpr(x) => x,
            _ => panic!(),
        };

        if let Expression::UnaryPrefix(unary) = bin_expr.left.as_ref() {
            assert_eq!(TokenKind::Asterisk, unary.operator.kind);
        }
    }

    #[test]
    fn should_respect_operator_precedence_13() {
        let code = r###"
        a();
        "###;

        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        assert_eq!(1, body.statements.len());
        
        let expr = match body.statements[0].as_ref() {
            Statement::Expression(x) => &x.expr,
            _ => panic!(),
        };
        let fx_call = match expr {
            Expression::FunctionCall(x) => x,
            _ => panic!(),
        };

        assert_eq!("a", fx_call.name_token.value);
        assert_eq!(0, fx_call.arguments.len());
    }

    #[test]
    fn should_require_commas_between_struct_members() {
        let code = r###"
        type X = { a: int, b: int };
        var x = X {
            a: 420
            b: 69
        };
        "###;

        let ast = AST::from_code(code);

        assert_eq!(false, ast.is_ok());
    }

    #[test]
    fn should_parse_unary_expression() {
        let code = r###"
        var x = -420 + 3;
        var y = -x;
        "###;

        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        assert_eq!(2, root.statements.len());

        if let Statement::Variable(var) = root.statements[0].as_ref() {
            if let Expression::BinaryExpr(expr) = &var.initializer {
                if let Expression::UnaryPrefix(unary) = expr.left.as_ref() {
                    assert_eq!(TokenKind::Minus, unary.operator.kind);
                    if let Expression::IntegerLiteral(lit) = unary.expr.as_ref() {
                        assert_eq!(420, lit.value);
                    }
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }

        if let Statement::Variable(var) = root.statements[1].as_ref() {
            if let Expression::UnaryPrefix(unary) = &var.initializer {
                if let Expression::Identifier(ident) = unary.expr.as_ref() {
                    assert_eq!("x", ident.name);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_array_type() {
        let code = r###"
        var x: [int] = false;
        var y: [int; 5] = false;
        "###;
        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        if let Statement::Variable(var) = &root.statements[0].as_ref() {
            if let Some(type_) = &var.type_ {
                if let TypeDeclKind::Array(elem, size) = &type_.kind {
                    assert_eq!("int", elem.identifier_token().value);
                    assert!(matches!(size, None));
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }

        if let Statement::Variable(var) = &root.statements[1].as_ref() {
            if let Some(type_) = &var.type_ {
                if let TypeDeclKind::Array(elem, size) = &type_.kind {
                    assert_eq!("int", elem.identifier_token().value);
                    assert_eq!(Some(5), *size);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_initialize_struct_in_member_access() {
        let code = r###"
        type B = {};
        type A = { b: B };
        var x = A { b: B {} };
        x.b = B {};
        "###;
        let ast = AST::from_code(code);

        assert_eq!(true, ast.is_ok())
    }

    #[test]
    fn should_parse_array_initialzer() {
        let code = r###"
        var x = [1, 2, 3];
        "###;
        let ast = AST::from_code(code).unwrap();
        let body = ast.root.as_block();

        if let Statement::Variable(var) = body.statements[0].as_ref() {
            if let Expression::ArrayLiteral(arr) = &var.initializer {
                assert_eq!(3, arr.elements.len());

                for (index, value) in [1, 2, 3].iter().enumerate() {
                    if let Expression::IntegerLiteral(x) = &arr.elements[index] {
                        assert_eq!(*value, x.value);
                    } else {
                        panic!();
                    }
                }
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_not_parse_struct_in_control_flow() {
        let code = r###"
        var x = 1;
        while x {}
        "###;
        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        if let Statement::While(while_) = root.statements[1].as_ref() {
            if let Expression::Identifier(ident) = &while_.condition {
                assert_eq!("x", ident.name);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_not_parse_struct_in_binary_expr_in_control_flow() {
        let code = r###"
        var x = 1;
        while 0 == x {}
        "###;
        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        if let Statement::While(while_) = root.statements[1].as_ref() {
            if let Expression::BinaryExpr(bin_expr) = &while_.condition {
                if let Expression::Identifier(ident) = bin_expr.right.as_ref() {
                    assert_eq!("x", ident.name);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_element_access() {
        let code = r###"
        var x = [1, 2, 3];
        var y = x[0][5];
        "###;
        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        if let Statement::Variable(var) = root.statements[1].as_ref() {
            if let Expression::ElementAccess(elem_access) = &var.initializer {
                if let Expression::ElementAccess(x) = elem_access.left.as_ref() {
                    if let Expression::Identifier(ident) = x.left.as_ref() {
                        assert_eq!("x", ident.name);
                    } else {
                        panic!();
                    }
                    if let Expression::IntegerLiteral(x) = x.right.as_ref() {
                        assert_eq!(0, x.value);
                    } else {
                        panic!();
                    }
                }
                if let Expression::IntegerLiteral(x) = elem_access.right.as_ref() {
                    assert_eq!(5, x.value);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn should_parse_element_access_with_expression() {
        let code = r###"
        var x = [1, 2, 3];
        var y = x[1 + 2];
        "###;
        let ast = AST::from_code(code).unwrap();
        let root = ast.root.as_block();

        if let Statement::Variable(var) = root.statements[1].as_ref() {
            if let Expression::ElementAccess(elem_access) = &var.initializer {
                if let Expression::Identifier(ident) = elem_access.left.as_ref() {
                    assert_eq!("x", ident.name);
                } else {
                    panic!();
                }
                if let Expression::BinaryExpr(_) = elem_access.right.as_ref() {
                    assert!(true);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}
