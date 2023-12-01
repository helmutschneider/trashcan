use std::collections::HashMap;

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
}

impl AST {
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
    StructInitializer(StructInitializer),
    MemberAccess(MemberAccess),
    Pointer(Pointer),
    BooleanLiteral(BooleanLiteral),
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
pub struct StructInitializer {
    pub name_token: Token,
    pub members: Vec<StructMemberInitializer>,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct StructMemberInitializer {
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
pub struct FunctionArgument {
    pub name_token: Token,
    pub type_: TypeName,
    pub parent: StatementId,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub id: StatementId,
    pub name_token: Token,
    pub arguments: Vec<FunctionArgument>,
    pub body: Rc<Statement>,
    pub return_type: TypeName,
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
    pub type_: Option<TypeName>,
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
pub struct TypeName {
    pub token: Token,
    pub is_pointer: bool,
    pub parent: StatementId,
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
    pub type_: TypeName,
}

#[derive(Debug, Clone)]
pub struct Pointer {
    pub expr: Box<Expression>,
    pub parent: StatementId,
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

struct ASTBuilder {
    tokens: Vec<Token>,
    token_index: usize,
    source: String,
    statements: Vec<Rc<Statement>>,
    symbols: Vec<UntypedSymbol>,
    num_statements: i64,
}

fn read_member_access_right_to_left(idents: &[Identifier], parent: StatementId) -> MemberAccess {
    // the member access pattern is one of very few nodes where we
    // actually parse the code right to left, eg. we want the right
    // hand side of the expression to always be a plain identifier.
    // for example, consider a property access 'a.b.c'. it would be modeled
    // like so:
    //
    //   PropertyAccess:
    //     left: PropertyAccess:
    //       left: a
    //       right: b
    //     right: c
    //
    //   -johan, 2023-11-25
    //
    if idents.len() == 2 {
        let lhs = Box::new(Expression::Identifier(idents[0].clone()));
        let rhs = idents[1].clone();

        return MemberAccess {
            left: lhs,
            right: rhs,
            parent: parent,
        };
    }

    let ident = idents.last().unwrap();
    let len = idents.len();

    let next = read_member_access_right_to_left(&idents[0..(len - 1)], parent);

    return MemberAccess {
        left: Box::new(Expression::MemberAccess(next)),
        right: ident.clone(),
        parent: parent,
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinaryOperatorAssociativity {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
struct BinaryOperator {
    kind: TokenKind,
    precedence: i64,
    associativity: BinaryOperatorAssociativity,
}

impl BinaryOperator {
    const fn new(kind: TokenKind, precedence: i64, associativity: BinaryOperatorAssociativity) -> Self {
        return Self {
            kind: kind,
            precedence: precedence,
            associativity: associativity,
        };
    }
}

const BINARY_OPERATORS: &[BinaryOperator] = &[
    BinaryOperator::new(TokenKind::OpenParenthesis, 15, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::CloseParenthesis, 15, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::Star, 13, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::Slash, 13, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::Plus, 12, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::Minus, 12, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::DoubleEquals, 9, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::NotEquals, 9, BinaryOperatorAssociativity::Left),
    BinaryOperator::new(TokenKind::Equals, 2, BinaryOperatorAssociativity::Right),
];

fn is_binary_operator(kind: TokenKind) -> bool {
    return find_binary_operator(kind).is_some();
}

fn find_binary_operator(kind: TokenKind) -> Option<BinaryOperator> {
    return BINARY_OPERATORS.iter().find(|op| op.kind == kind).cloned();
}

impl ASTBuilder {
    fn peek(&self) -> Result<TokenKind, Error> {
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

    fn consume_one_token(&mut self) -> Result<Token, Error> {
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
        let type_ = self.expect_type_name(parent)?;
        let arg = FunctionArgument {
            name_token: name_token,
            type_: type_,
            parent: parent,
        };
        return Result::Ok(arg);
    }

    fn expect_type_name(&mut self, parent: StatementId) -> Result<TypeName, Error> {
        let kind = self.peek()?;
        let type_ = match kind {
            TokenKind::Identifier => {
                let name_token = self.consume_one_token()?;
                TypeName {
                    token: name_token,
                    is_pointer: false,
                    parent: parent,
                }
            }
            TokenKind::Ampersand => {
                self.consume_one_token()?;
                let name_token = self.expect(TokenKind::Identifier)?;
                TypeName {
                    token: name_token,
                    is_pointer: true,
                    parent: parent,
                }
            }
            _ => {
                let message = format!("expected type identifier, got '{}'", kind);
                let tok = &self.tokens[self.token_index];
                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };
        return Result::Ok(type_);
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
                let mut ret = Return {
                    id: id,
                    expr: Expression::Void,
                    token: token,
                    parent: parent,
                };
                
                let expr = match self.peek()? {
                    TokenKind::Semicolon => Expression::Void,
                    _ => self.expect_expression(id)?,
                };
                self.expect(TokenKind::Semicolon)?;

                ret.expr = expr;
                let stmt = self.add_statement(Statement::Return(ret));

                stmt
            }
            TokenKind::IfKeyword => {
                let if_id = self.get_and_increment_statement_id();
                let if_token = self.consume_one_token()?;

                let condition = self.expect_expression(if_id)?;
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

                let condition = self.expect_expression(while_id)?;
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
                let expr = self.expect_expression(expr_id)?;
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

    fn expect_function_call(&mut self, parent: StatementId) -> Result<FunctionCall, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut args: Vec<Expression> = Vec::new();

        while self.peek()? != TokenKind::CloseParenthesis {
            let expr = self.expect_expression(parent)?;
            args.push(expr);

            if self.peek()? != TokenKind::CloseParenthesis {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;

        let expr = FunctionCall {
            name_token: name_token,
            arguments: args,
            parent: parent,
        };

        return Result::Ok(expr);
    }

    fn expect_expression(&mut self, parent: StatementId) -> Result<Expression, Error> {
        return self.expect_expression_and_maybe_read_binary_expression(parent, false);
    }

    fn expect_expression_and_maybe_read_binary_expression(&mut self, parent: StatementId, is_reading_binary_expr: bool) -> Result<Expression, Error> {
        let kind = self.peek()?;
        let expr = match kind {
            TokenKind::Identifier => {
                let next_plus_one = self.peek_at(1)?;

                match next_plus_one {
                    TokenKind::OpenParenthesis => {
                        let fx = self.expect_function_call(parent)?;
                        Expression::FunctionCall(fx)
                    }
                    TokenKind::Dot => {
                        let mut idents: Vec<Identifier> = Vec::new();

                        loop {
                            let ident = self.expect_identifier(parent)?;
                            idents.push(ident);
                            if self.peek() != Ok(TokenKind::Dot) {
                                break;
                            }
                            self.consume_one_token()?;
                        }

                        let prop_access = read_member_access_right_to_left(&idents, parent);
                        Expression::MemberAccess(prop_access)
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
            TokenKind::OpenParenthesis => {
                self.consume_one_token()?;
                let inner = self.expect_expression(parent)?;
                self.expect(TokenKind::CloseParenthesis)?;
                inner
            }
            TokenKind::Ampersand => {
                self.consume_one_token()?;
                let inner = self.expect_expression(parent)?;
                let ptr = Pointer {
                    expr: Box::new(inner),
                    parent: parent,
                };
                Expression::Pointer(ptr)
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
            TokenKind::Minus => {
                self.consume_one_token()?;
                let token = self.expect(TokenKind::IntegerLiteral)?;
                let parsed: i64 = token.value.parse().unwrap();
                Expression::IntegerLiteral(IntegerLiteral {
                    value: -parsed,
                    token: token,
                    parent: parent,
                })
            }
            _ => {
                let message = format!("expected expression, got token '{}'", kind);
                let tok = &self.tokens[self.token_index];

                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };

        let mut actual_expr = expr;

        if !is_reading_binary_expr && !self.is_end_of_file() && is_binary_operator(self.peek()?) {
            actual_expr = self.do_shunting_yard(actual_expr, parent)?;
        }

        return Result::Ok(actual_expr);
    }

    fn do_shunting_yard(&mut self, first_lhs: Expression, parent: StatementId) -> Result<Expression, Error> {
        // a shunting yard implementation, slightly adapted from the wikipedia
        // example. since we already have a starting left hand when we get here
        // the next token must always be an operator. instead of evaluating
        // the expression we construct AST nodes.
        //
        //   https://en.wikipedia.org/wiki/Shunting_yard_algorithm
        //
        //   -johan, 2023-11-27

        let mut out_queue: Vec<Expression> = vec![first_lhs];
        let mut operator_stack: Vec<Token> = Vec::new();

        fn pop_expression(parent: StatementId, operator_stack: &mut Vec<Token>, out_queue: &mut Vec<Expression>) {
            assert!(out_queue.len() >= 2);
            assert!(operator_stack.len() >= 1);

            let operator = operator_stack.pop().unwrap();
            let rhs = out_queue.pop().unwrap();
            let lhs = out_queue.pop().unwrap();
            let bin_expr = BinaryExpr {
                left: Box::new(lhs),
                operator: operator,
                right: Box::new(rhs),
                parent: parent,
            };
            out_queue.push(Expression::BinaryExpr(bin_expr));
        }

        fn should_keep_shunting(kind: TokenKind, operator_stack: &[Token]) -> bool {
            // the parentheses are not really binary operators but we
            // need to be able to read them to keep track of precedence.
            // a problem with that is that we might read too far into
            // another statement.
            //
            // if we encounter a close parenthesis make sure that we're
            // actually closing something.
            //   -johan, 2023-11-27
            if kind == TokenKind::CloseParenthesis {
                return operator_stack.iter().any(|tok| tok.kind == TokenKind::OpenParenthesis);
            }
            return is_binary_operator(kind);
        }

        fn has_operator_on_stack_with_greater_precedence(kind: TokenKind, operator_stack: &[Token]) -> bool {
            let my_op = find_binary_operator(kind).unwrap();

            return operator_stack.last().map(|op| {
                let other_op = find_binary_operator(op.kind).unwrap();

                return (other_op.precedence >= my_op.precedence)
                    || (other_op.precedence == my_op.precedence 
                        && my_op.associativity == BinaryOperatorAssociativity::Left);
            })
            .unwrap_or(false);
        }

        while should_keep_shunting(self.peek()?, &operator_stack) {
            let op_token = self.consume_one_token()?;

            if op_token.kind == TokenKind::OpenParenthesis {
                operator_stack.push(op_token);
            } else if op_token.kind == TokenKind::CloseParenthesis {
                while !operator_stack.is_empty() && operator_stack.last().unwrap().kind != TokenKind::OpenParenthesis {
                    pop_expression(parent, &mut operator_stack, &mut out_queue);
                }
                
                // pop the open parenthesis
                operator_stack.pop().unwrap();
            } else {
                while has_operator_on_stack_with_greater_precedence(op_token.kind, &operator_stack) {
                    pop_expression(parent, &mut operator_stack, &mut out_queue);
                }
                operator_stack.push(op_token);
            }

            let maybe_rhs = self.expect_expression_and_maybe_read_binary_expression(parent, true)?;
            out_queue.push(maybe_rhs);
        }

        while !operator_stack.is_empty() {
            pop_expression(parent, &mut operator_stack, &mut out_queue);
        }

        assert_eq!(1, out_queue.len());
        let expr = &out_queue[0];
        return Ok(expr.clone());
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
                let t = self.expect_type_name(var_id)?;
                Some(t)
            }
            _ => None,
        };

        self.expect(TokenKind::Equals)?;

        let init_expr: Expression;

        if self.peek()? == TokenKind::Identifier && self.peek_at(1)? == TokenKind::OpenBrace {
            // the struct initializer collides with the if-statement, because they
            // both accept an identifier followed by an opening brace. we would need
            // some kind of contextual parsing to resolve that. the workaround for now
            // is to just allow the struct initializer on variable assignment.
            //   -johan, 2023-11-24
            let struct_init = self.expect_struct_initializer(var_id)?;
            init_expr = Expression::StructInitializer(struct_init);
        } else {
            init_expr = self.expect_expression(var_id)?;
        }

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
        let return_type = self.expect_type_name(fx_id)?;
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
            let type_ = self.expect_type_name(struct_id)?;

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

    fn expect_struct_initializer(&mut self, parent: StatementId) -> Result<StructInitializer, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenBrace)?;

        let mut member_inits: Vec<StructMemberInitializer> = Vec::new();

        while self.peek()? != TokenKind::CloseBrace {
            let field_name_token = self.expect(TokenKind::Identifier)?;
            self.expect(TokenKind::Colon)?;

            let init_expr: Expression;

            if self.peek()? == TokenKind::Identifier && self.peek_at(1)? == TokenKind::OpenBrace {
                // the struct initializer collides with the if-statement, because they
                // both accept an identifier followed by an opening brace. we would need
                // some kind of contextual parsing to resolve that. the workaround for now
                // is to just allow the struct initializer on variable assignment.
                //   -johan, 2023-11-24
                let struct_init = self.expect_struct_initializer(parent)?;
                init_expr = Expression::StructInitializer(struct_init);
            } else {
                init_expr = self.expect_expression(parent)?;
            }

            member_inits.push(StructMemberInitializer {
                field_name_token: field_name_token,
                value: init_expr,
            });

            if self.peek()? != TokenKind::CloseBrace {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::CloseBrace)?;

        let struct_ = StructInitializer {
            name_token: name_token,
            members: member_inits,
            parent: parent,
        };
        return Ok(struct_);
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

        assert_eq!("int", type_.token.value);
        assert_eq!("y", fx.arguments[1].name_token.value);

        let type_ = &fx.arguments[1].type_;

        assert_eq!("double", type_.token.value);

        let type_ = &fx.return_type;

        assert_eq!("void", type_.token.value);

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

            let type_ = &x.type_;

            assert_eq!("int", type_.as_ref().unwrap().token.value);

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

            let type_ = &x.type_.as_ref().unwrap().token.value;

            assert_eq!("double", type_);

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

        assert_eq!(TokenKind::DoubleEquals, bin_expr.operator.kind);

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
        assert_eq!("string", struct_.members[0].type_.token.value);
        assert_eq!("age", struct_.members[1].field_name_token.value);
        assert_eq!("int", struct_.members[1].type_.token.value);
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
            Expression::StructInitializer(s) => s,
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
    fn should_respect_operator_presedence_1() {
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

        assert_eq!(TokenKind::DoubleEquals, bin_expr.operator.kind);

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
    fn should_respect_operator_presedence_2() {
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

        assert_eq!(TokenKind::Star, left.operator.kind);
        
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
    fn should_respect_operator_presedence_3() {
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

        assert_eq!(TokenKind::Star, bin_expr.operator.kind);

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
    fn should_respect_operator_presedence_4() {
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

        assert_eq!(TokenKind::Star, bin_expr.operator.kind);

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
    fn should_respect_operator_presedence_5() {
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

        assert_eq!(TokenKind::DoubleEquals, bin_expr.operator.kind);
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
}
