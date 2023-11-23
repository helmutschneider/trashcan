use crate::tokenizer::{tokenize, Token, TokenKind};
use crate::util::report_error;
use crate::util::Error;
use crate::util::SourceLocation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatementIndex(pub usize);

impl std::fmt::Display for StatementIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct AST {
    pub body_index: StatementIndex,
    pub statements: Vec<Statement>,
}

trait HasStatements {
    fn get_statements(&self) -> &[Statement];
}

impl HasStatements for AST {
    fn get_statements(&self) -> &[Statement] {
        return &self.statements;
    }
}

pub trait ASTLike {
    fn get_statement(&self, index: StatementIndex) -> &Statement;
    fn get_block(&self, index: StatementIndex) -> &Block;
    fn get_function(&self, name: &str) -> Option<&Function>;
}

impl<T: HasStatements> ASTLike for T {
    fn get_statement(&self, at: StatementIndex) -> &Statement {
        if let Some(s) = self.get_statements().get(at.0) {
            return s;
        }
        panic!("statement {} does not exist", at);
    }

    fn get_block(&self, at: StatementIndex) -> &Block {
        if let Statement::Block(b) = self.get_statement(at) {
            return b;
        }
        panic!("statement {} is not a block", at);
    }

    fn get_function(&self, name: &str) -> Option<&Function> {
        let body_index = StatementIndex(0);
        let block = self.get_block(body_index);

        for index in &block.statements {
            if let Statement::Function(fx) = self.get_statement(*index) {
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
        return self.get_block(self.body_index);
    }

    pub fn from_code(code: &str) -> Result<Self, Error> {
        let tokens = crate::tokenizer::tokenize(code)?;
        let mut builder = ASTBuilder {
            tokens: tokens.to_vec(),
            token_index: 0,
            source: code.to_string(),
            statements: Vec::new(),
        };

        let body = Block {
            statements: Vec::new(),
            parent: None,
        };
        let body_index = builder.add_statement(Statement::Block(body));

        while builder.token_index < tokens.len() {
            let stmt = builder.expect_statement(body_index)?;
            let body = match builder.get_statement_mut(body_index) {
                Statement::Block(x) => x,
                _ => panic!(),
            };
            body.statements.push(stmt);
        }

        let ast = AST {
            body_index: body_index,
            statements: builder.statements,
        };

        return Result::Ok(ast);
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    Function(Function),
    Variable(Variable),
    Expression(Expression),
    Return(Return),
    Block(Block),
    If(If),
}

#[derive(Debug, Clone)]
pub struct Return {
    pub expr: Expression,
    pub token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Void,
    Identifier(Identifier),
    IntegerLiteral(IntegerLiteral),
    StringLiteral(StringLiteral),
    FunctionCall(FunctionCall),
    BinaryExpr(BinaryExpr),
}

#[derive(Debug, Clone)]
pub struct IntegerLiteral {
    pub value: i64,
    pub token: Token,
}

#[derive(Debug, Clone)]
pub struct StringLiteral {
    pub value: String,
    pub token: Token,
}

#[derive(Debug, Clone)]
pub struct Identifier {
    pub name: String,
    pub token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct FunctionArgument {
    pub name_token: Token,
    pub type_: TypeDeclaration,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name_token: Token,
    pub arguments: Vec<FunctionArgument>,
    pub body: StatementIndex,
    pub return_type: TypeDeclaration,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<StatementIndex>,
    pub parent: Option<StatementIndex>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name_token: Token,
    pub type_: Option<TypeDeclaration>,
    pub initializer: Expression,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name_token: Token,
    pub arguments: Vec<Expression>,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct If {
    pub condition: Expression,
    pub block: StatementIndex,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub enum TypeDeclaration {
    Name(Token),
}

struct ASTBuilder {
    tokens: Vec<Token>,
    token_index: usize,
    source: String,
    statements: Vec<Statement>,
}

impl HasStatements for ASTBuilder {
    fn get_statements(&self) -> &[Statement] {
        return &self.statements;
    }
}

impl ASTBuilder {
    fn peek(&self) -> TokenKind {
        return self.peek_at(0);
    }

    fn peek_at(&self, offset: usize) -> TokenKind {
        return self
            .tokens
            .get(self.token_index + offset)
            .map(|t| t.kind)
            .expect("End-of-file reached.");
    }

    fn add_statement(&mut self, stmt: Statement) -> StatementIndex {
        let id = self.statements.len();
        self.statements.push(stmt);
        return StatementIndex(id);
    }

    fn get_statement_mut(&mut self, index: StatementIndex) -> &mut Statement {
        return self.statements.get_mut(index.0).unwrap();
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

    fn expect_function_argument(&mut self) -> Result<FunctionArgument, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        let type_ = self.expect_type_declaration()?;
        let arg = FunctionArgument {
            name_token: name_token,
            type_: type_,
        };
        return Result::Ok(arg);
    }

    fn expect_type_declaration(&mut self) -> Result<TypeDeclaration, Error> {
        self.expect(TokenKind::Colon)?;
        return self.expect_type();
    }

    fn expect_type(&mut self) -> Result<TypeDeclaration, Error> {
        let type_ = match self.peek() {
            TokenKind::Identifier => {
                let name_token = self.consume_one_token()?;
                TypeDeclaration::Name(name_token)
            }
            _ => {
                let message = format!("expected type identifier, got '{}'", self.peek());
                let tok = &self.tokens[self.token_index];
                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };
        return Result::Ok(type_);
    }

    fn expect_block(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let block = Block {
            statements: Vec::new(),
            parent: Some(parent),
        };

        let block_index = self.add_statement(Statement::Block(block));

        while self.peek() != TokenKind::CloseBrace {
            let stmt = self.expect_statement(block_index)?;

            if let Statement::Block(b) = self.get_statement_mut(block_index) {
                b.statements.push(stmt);
            }
        }

        self.expect(TokenKind::CloseBrace)?;

        return Result::Ok(block_index);
    }

    fn expect_statement(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        let stmt_index = match self.peek() {
            TokenKind::FunctionKeyword => {
                let stmt_index = self.expect_function(parent)?;
                stmt_index
            }
            TokenKind::VariableKeyword => {
                let stmt_index = self.expect_variable(parent)?;
                self.expect(TokenKind::Semicolon)?;
                stmt_index
            }
            TokenKind::ReturnKeyword => {
                let token = self.expect(TokenKind::ReturnKeyword)?;
                let ret = Return {
                    expr: Expression::Void,
                    token: token,
                    parent: parent,
                };
                let ret_stmt_index = self.add_statement(Statement::Return(ret));
                let expr = match self.peek() {
                    TokenKind::Semicolon => Expression::Void,
                    _ => self.expect_expression(ret_stmt_index)?,
                };
                self.expect(TokenKind::Semicolon)?;

                if let Statement::Return(ret_stmt) = self.get_statement_mut(ret_stmt_index) {
                    ret_stmt.expr = expr;
                }

                ret_stmt_index
            }
            TokenKind::IfKeyword => {
                let if_token = self.consume_one_token()?;
                let if_stmt = If {
                    condition: Expression::Void,
                    block: StatementIndex(0),
                    parent: parent,
                };
                let if_stmt_index = self.add_statement(Statement::If(if_stmt));
                let condition = self.expect_expression(if_stmt_index)?;
                let block_index = self.expect_block(if_stmt_index)?;

                if let Statement::If(if_stmt) = self.get_statement_mut(if_stmt_index) {
                    if_stmt.condition = condition;
                    if_stmt.block = block_index;
                }

                if_stmt_index
            }
            TokenKind::OpenBrace => {
                let block_index = self.expect_block(parent)?;
                block_index
            }
            _ => {
                let expr = self.expect_expression(parent)?;
                self.expect(TokenKind::Semicolon)?;
                let stmt = self.add_statement(Statement::Expression(expr));
                stmt
            }
        };

        return Result::Ok(stmt_index);
    }

    fn expect_function_call(&mut self, parent: StatementIndex) -> Result<FunctionCall, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut args: Vec<Expression> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let expr = self.expect_expression(parent)?;
            args.push(expr);

            if self.peek() == TokenKind::Comma {
                self.consume_one_token()?;
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

    fn expect_expression(&mut self, parent: StatementIndex) -> Result<Expression, Error> {
        let expr = match self.peek() {
            TokenKind::Identifier => {
                if self.peek_at(1) == TokenKind::OpenParenthesis {
                    let fx = self.expect_function_call(parent)?;
                    Expression::FunctionCall(fx)
                } else {
                    let name_token = self.expect(TokenKind::Identifier)?;
                    let ident = Identifier {
                        name: name_token.value.clone(),
                        token: name_token,
                        parent: parent,
                    };
                    Expression::Identifier(ident)
                }
            }
            TokenKind::IntegerLiteral => {
                let token = self.consume_one_token()?;
                let parsed: i64 = token.value.parse().unwrap();
                Expression::IntegerLiteral(IntegerLiteral {
                    value: parsed,
                    token: token,
                })
            }
            TokenKind::StringLiteral => {
                let token = self.consume_one_token()?;
                Expression::StringLiteral(StringLiteral {
                    value: token.value.clone(),
                    token: token,
                })
            }
            TokenKind::OpenParenthesis => {
                self.consume_one_token()?;
                let inner = self.expect_expression(parent)?;
                self.expect(TokenKind::CloseParenthesis)?;
                inner
            }
            _ => {
                let message = format!("invalid token for expression: {}", self.peek());
                let tok = &self.tokens[self.token_index];

                return report_error(&self.source, &message, SourceLocation::Token(tok));
            }
        };

        let actual_expr = match self.peek() {
            TokenKind::Plus | TokenKind::Minus | TokenKind::DoubleEquals | TokenKind::NotEquals => {
                let op = self.consume_one_token()?;
                let rhs = self.expect_expression(parent)?;
                let bin_expr = BinaryExpr {
                    left: Box::new(expr),
                    operator: op,
                    right: Box::new(rhs),
                    parent: parent,
                };
                Expression::BinaryExpr(bin_expr)
            }
            _ => expr,
        };

        return Result::Ok(actual_expr);
    }

    fn expect_variable(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::VariableKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        let type_ = match self.peek() {
            TokenKind::Colon => self.expect_type_declaration().ok(),
            _ => None,
        };
        let var = Variable {
            name_token: name_token.clone(),
            type_: type_,
            initializer: Expression::Void,
            parent: parent,
        };
        let var_stmt_index = self.add_statement(Statement::Variable(var));

        self.expect(TokenKind::Equals)?;
        let init_expr = self.expect_expression(var_stmt_index)?;

        if let Statement::Variable(var) = self.get_statement_mut(var_stmt_index) {
            var.initializer = init_expr;
        }

        return Result::Ok(var_stmt_index);
    }

    fn expect_function(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let fx = Function {
            name_token: name_token.clone(),
            arguments: Vec::new(),
            body: StatementIndex(0),
            return_type: TypeDeclaration::Name(Token {
                kind: TokenKind::Identifier,
                source_index: 0,
                value: String::new(),
            }),
            parent: parent,
        };
        let fx_stmt_index = self.add_statement(Statement::Function(fx));

        let mut arguments: Vec<FunctionArgument> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg = self.expect_function_argument()?;
            arguments.push(arg);

            if self.peek() == TokenKind::Comma {
                self.consume_one_token()?;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        let return_type = self.expect_type_declaration()?;
        let body_block_index = self.expect_block(fx_stmt_index)?;

        if let Statement::Function(fx) = self.get_statement_mut(fx_stmt_index) {
            fx.arguments = arguments.clone();
            fx.body = body_block_index;
            fx.return_type = return_type;
        }

        return Result::Ok(fx_stmt_index);
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::tokenizer::*;

    #[test]
    fn should_create_ast_from_function_with_empty_body() {
        let code = "fun do_thing(x: int, y: double): void {}";
        let ast = AST::from_code(&code).unwrap();
        let fx = ast.get_function("do_thing").unwrap();

        assert_eq!(2, fx.arguments.len());
        assert_eq!("x", fx.arguments[0].name_token.value);

        let type_ = match &fx.arguments[0].type_ {
            TypeDeclaration::Name(t) => &t.value,
            _ => panic!(),
        };

        assert_eq!("int", type_);
        assert_eq!("y", fx.arguments[1].name_token.value);

        let type_ = match &fx.arguments[1].type_ {
            TypeDeclaration::Name(t) => &t.value,
            _ => panic!(),
        };

        assert_eq!("double", type_);

        let type_ = match &fx.return_type {
            TypeDeclaration::Name(t) => &t.value,
            _ => panic!(),
        };

        assert_eq!("void", type_);

        let fx_body = ast.get_block(fx.body);
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

        if let Statement::Variable(x) = ast.get_statement(ast.body().statements[0]) {
            assert_eq!("x", x.name_token.value);

            let type_ = match &x.type_ {
                Some(TypeDeclaration::Name(t)) => &t.value,
                _ => panic!(),
            };

            assert_eq!("int", type_);

            if let Expression::IntegerLiteral(init) = &x.initializer {
                assert_eq!(1, init.value);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }

        if let Statement::Variable(x) = ast.get_statement(ast.body().statements[1]) {
            assert_eq!("y", x.name_token.value);

            let type_ = match &x.type_ {
                Some(TypeDeclaration::Name(t)) => &t.value,
                _ => panic!(),
            };

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

        if let Statement::Expression(Expression::FunctionCall(call)) =
            ast.get_statement(ast.body().statements[0])
        {
            assert_eq!("call_me_maybe", call.name_token.value);
            assert_eq!(2, call.arguments.len());
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

        let fx = ast.get_function("main").unwrap();
        assert_eq!("main", fx.name_token.value);

        let fx_body = ast.get_block(fx.body);
        assert_eq!(3, fx_body.statements.len());
    }

    #[test]
    fn should_create_ast_from_binary_expr() {
        let code = r###"
            var x: int = 1 + do_thing(420, 69);
        "###;

        let ast = AST::from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let stmt = ast.get_statement(ast.body().statements[0]);

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
        dbg!("{:?}", &ast);

        let stmt = ast.get_statement(ast.body().statements[0]);

        if let Statement::Variable(v) = stmt {
            if let Expression::BinaryExpr(expr) = &v.initializer {
                assert_eq!(true, matches!(*expr.left, Expression::IntegerLiteral(_)));
                assert_eq!(true, matches!(*expr.right, Expression::BinaryExpr(_)));
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

        let if_stmt = match ast.get_statement(ast.body().statements[0]) {
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
    fn should_report_error_for_missing_type() {
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
}
