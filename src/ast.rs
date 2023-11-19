use crate::tokenizer::{tokenize, Token, TokenKind};
use crate::util::report_error;
use crate::util::Error;
use crate::util::SourceLocation;

#[derive(Debug, Clone, Copy)]
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
    fn get_symbol(&self, name: &str, at: StatementIndex) -> Option<&Symbol>;
    fn get_function(&self, name: &str) -> Option<&Function>;
    fn get_enclosing_block(&self, index: StatementIndex) -> Option<StatementIndex>;
    fn get_enclosing_function(&self, index: StatementIndex) -> Option<&Function>;
}

fn get_parent_index(ast: &impl ASTLike, index: StatementIndex) -> Option<StatementIndex> {
    let stmt = ast.get_statement(index);
    let maybe_parent_index = match stmt {
        Statement::Function(fx) => Some(fx.parent),
        Statement::Variable(var) => Some(var.parent),
        Statement::Expression(expr) => match expr {
            Expression::BinaryExpr(bin_expr) => Some(bin_expr.parent),
            Expression::FunctionCall(fx_call) => Some(fx_call.parent),
            Expression::Identifier(ident) => Some(ident.parent),
            _ => panic!("cannot find parent of {:?}", expr),
        },
        Statement::Return(ret) => Some(ret.parent),
        Statement::Block(b) => b.parent,
        Statement::If(if_stmt) => Some(if_stmt.parent),
    };
    return maybe_parent_index;
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

    fn get_symbol(&self, name: &str, at: StatementIndex) -> Option<&Symbol> {
        let block_index = self.get_enclosing_block(at);

        if let Some(i) = block_index {
            if let Statement::Block(b) = self.get_statement(i) {
                return b.symbols.iter().find(|s| s.name == name);
            }
        }

        return None;
    }

    fn get_function(&self, name: &str) -> Option<&Function> {
        let body_index = StatementIndex(0);

        return match self.get_symbol(name, body_index) {
            Some(sym) => {
                if let Statement::Function(fx) = self.get_statement(sym.declared_at) {
                    return Some(fx);
                }
                return None;
            }
            None => None,
        };
    }

    fn get_enclosing_block(&self, at_stmt: StatementIndex) -> Option<StatementIndex> {
        let mut maybe_index = Some(at_stmt);

        while let Some(index) = maybe_index {
            let stmt = self.get_statement(index);
            if let Statement::Block(_) = stmt {
                return Some(index);
            }
            maybe_index = get_parent_index(self, index);
        }

        return None;
    }

    fn get_enclosing_function(&self, at_stmt: StatementIndex) -> Option<&Function> {
        let mut maybe_index = Some(at_stmt);

        while let Some(index) = maybe_index {
            let stmt = self.get_statement(index);
            if let Statement::Function(fx) = stmt {
                return Some(fx);
            }
            maybe_index = get_parent_index(self, index);
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
            symbols: Vec::new(),
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

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Variable,
    Function,
    FunctionArgument,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub type_: String,
    pub kind: SymbolKind,
    pub declared_at: StatementIndex,
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
    pub parent: StatementIndex,
    pub token: Token,
}

#[derive(Debug, Clone)]
pub enum Expression {
    None,
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
    pub name: String,
    pub type_: String,
    pub name_token: Token,
    pub type_token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<FunctionArgument>,
    pub body: StatementIndex,
    pub return_type: String,
    pub name_token: Token,
    pub return_type_token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<StatementIndex>,
    pub symbols: Vec<Symbol>,
    pub parent: Option<StatementIndex>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: String,
    pub initializer: Expression,
    pub name_token: Token,
    pub type_token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Vec<Expression>,
    pub name_token: Token,
    pub parent: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct If {
    pub condition: Expression,
    pub block: StatementIndex,
    pub parent: StatementIndex,
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

    fn add_symbol(&mut self, symbol: Symbol) {
        if let Some(i) = self.get_enclosing_block(symbol.declared_at) {
            if let Statement::Block(b) = self.get_statement_mut(i) {
                b.symbols.push(symbol);
            }
        }
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

    fn expect_function_argument(
        &mut self,
        parent: StatementIndex,
    ) -> Result<FunctionArgument, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_token = self.expect(TokenKind::Identifier)?;
        let arg = FunctionArgument {
            name: name_token.value.clone(),
            type_: type_token.value.clone(),
            name_token: name_token,
            type_token: type_token,
            parent: parent,
        };
        return Result::Ok(arg);
    }

    fn expect_block(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let block = Block {
            statements: Vec::new(),
            symbols: Vec::new(),
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
                    expr: Expression::None,
                    parent: parent,
                    token: token,
                };
                let ret_stmt_index = self.add_statement(Statement::Return(ret));
                let expr = match self.peek() {
                    TokenKind::Semicolon => Expression::None,
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
                    condition: Expression::None,
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
            name: name_token.value.clone(),
            arguments: args,
            name_token: name_token,
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
        self.expect(TokenKind::Colon)?;
        let type_token = self.expect(TokenKind::Identifier)?;

        let var = Variable {
            name: name_token.value.clone(),
            type_: type_token.value.clone(),
            initializer: Expression::None,
            name_token: name_token.clone(),
            type_token: type_token.clone(),
            parent: parent,
        };
        let var_stmt_index = self.add_statement(Statement::Variable(var));

        self.expect(TokenKind::Equals)?;
        let init_expr = self.expect_expression(var_stmt_index)?;

        if let Statement::Variable(var) = self.get_statement_mut(var_stmt_index) {
            var.initializer = init_expr;
        }

        let var_sym = Symbol {
            name: name_token.value,
            type_: type_token.value,
            kind: SymbolKind::Variable,
            declared_at: var_stmt_index,
        };
        self.add_symbol(var_sym);

        return Result::Ok(var_stmt_index);
    }

    fn expect_function(&mut self, parent: StatementIndex) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let fx = Function {
            name: name_token.value.clone(),
            arguments: Vec::new(),
            body: StatementIndex(0),
            return_type: String::new(),
            return_type_token: Token {
                kind: TokenKind::Identifier,
                source_index: 0,
                value: String::new(),
            },
            name_token: name_token.clone(),
            parent: parent,
        };
        let fx_stmt_index = self.add_statement(Statement::Function(fx));

        let mut arguments: Vec<FunctionArgument> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg = self.expect_function_argument(fx_stmt_index)?;
            arguments.push(arg);

            if self.peek() == TokenKind::Comma {
                self.consume_one_token()?;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        self.expect(TokenKind::Colon)?;
        let return_type_token = self.expect(TokenKind::Identifier)?;
        let body_block_index = self.expect_block(fx_stmt_index)?;

        if let Statement::Function(fx) = self.get_statement_mut(fx_stmt_index) {
            fx.arguments = arguments.clone();
            fx.body = body_block_index;
            fx.return_type = return_type_token.value.clone();
            fx.return_type_token = return_type_token;
        }

        for arg in &arguments {
            let fx_arg_sym = Symbol {
                name: arg.name.clone(),
                type_: arg.type_.clone(),
                kind: SymbolKind::FunctionArgument,
                declared_at: body_block_index,
            };
            self.add_symbol(fx_arg_sym);
        }

        let fx_sym = Symbol {
            name: name_token.value,
            type_: String::new(),
            kind: SymbolKind::Function,
            declared_at: fx_stmt_index,
        };
        self.add_symbol(fx_sym);

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
        assert_eq!("x", fx.arguments[0].name);
        assert_eq!("int", fx.arguments[0].type_);
        assert_eq!("y", fx.arguments[1].name);
        assert_eq!("double", fx.arguments[1].type_);
        assert_eq!("void", fx.return_type);

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
            assert_eq!("x", x.name);
            assert_eq!("int", x.type_);

            if let Expression::IntegerLiteral(init) = &x.initializer {
                assert_eq!(1, init.value);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }

        if let Statement::Variable(x) = ast.get_statement(ast.body().statements[1]) {
            assert_eq!("y", x.name);
            assert_eq!("double", x.type_);

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
            assert_eq!("call_me_maybe", call.name);
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
        assert_eq!("main", fx.name);

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
    fn should_create_symbol_table() {
        let code = r###"
            fun add(x: int, y: int): int {
                var z: int = 6;
                return x + y;
            }
        "###;
        let ast = AST::from_code(code).unwrap();
        let body = ast.body();
        let body_syms = &body.symbols;

        assert_eq!(1, body_syms.len());
        assert_eq!("add", body_syms[0].name);
        assert_eq!(SymbolKind::Function, body_syms[0].kind);

        let add_fn = ast.get_function("add").unwrap();
        let add_fn_body = ast.get_block(add_fn.body);
        let add_syms = &add_fn_body.symbols;

        assert_eq!(3, add_syms.len());
        assert_eq!("z", add_syms[0].name);
        assert_eq!(SymbolKind::Variable, add_syms[0].kind);
        assert_eq!("x", add_syms[1].name);
        assert_eq!(SymbolKind::FunctionArgument, add_syms[1].kind);
        assert_eq!("y", add_syms[2].name);
        assert_eq!(SymbolKind::FunctionArgument, add_syms[2].kind);
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
