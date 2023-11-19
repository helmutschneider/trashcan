use crate::tokenizer::{tokenize, Token, TokenKind};
use crate::util::report_error;
use crate::util::Error;
use crate::util::SourceLocation;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct AST {
    pub body: Block,
    pub symbols: Vec<Symbol>,
}

impl AST {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let tokens = crate::tokenizer::tokenize(code)?;
        let mut builder = ASTBuilder {
            tokens: tokens.to_vec(),
            token_index: 0,
            source: code.to_string(),
            statements: Vec::new(),
            symbols: Vec::new(),
        };

        let mut body = Block {
            statements: Vec::new(),
        };

        while builder.token_index < tokens.len() {
            let stmt = builder.expect_statement()?;
            body.statements.push(stmt);
        }

        let ast = AST {
            body: body,
            symbols: builder.symbols,
        };

        return Result::Ok(ast);
    }

    pub fn get_symbol(&self, name: &str, kind: SymbolKind) -> Option<&Symbol> {
        let fn_sym = self
            .symbols
            .iter()
            .find(|s| s.kind == kind && s.name == name);
        return fn_sym;
    }

    pub fn get_function(&self, name: &str) -> Option<&Function> {
        return match self.get_symbol(name, SymbolKind::Function) {
            Some(sym) => {
                if let Statement::Function(fx) = sym.declared_at.as_ref() {
                    return Some(fx);
                }
                return None;
            },
            None => None
        };
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
    pub kind: SymbolKind,
    pub declared_at: Rc<Statement>,
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
}

#[derive(Debug, Clone)]
pub enum Expression {
    Empty,
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
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct FunctionArgument {
    pub name: String,
    pub type_: String,
    pub name_token: Token,
    pub type_token: Token,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<FunctionArgument>,
    pub body: Block,
    pub return_type: String,
    pub name_token: Token,
    pub return_type_token: Token,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Rc<Statement>>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: String,
    pub initializer: Expression,
    pub name_token: Token,
    pub type_token: Token,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Vec<Expression>,
    pub name_token: Token,
}

#[derive(Debug, Clone)]
pub struct If {
    pub condition: Expression,
    pub block: Block,
}

struct ASTBuilder {
    tokens: Vec<Token>,
    token_index: usize,
    source: String,
    statements: Vec<Rc<Statement>>,
    symbols: Vec<Symbol>,
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

    fn add_statement(&mut self, stmt: Statement) -> Rc<Statement> {
        let rced = Rc::new(stmt);
        self.statements.push(rced);
        let last = self.statements.last().unwrap();
        return Rc::clone(last);
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
        self.expect(TokenKind::Colon)?;
        let type_token = self.expect(TokenKind::Identifier)?;
        let arg = FunctionArgument {
            name: name_token.value.clone(),
            type_: type_token.value.clone(),
            name_token: name_token,
            type_token: type_token,
        };
        return Result::Ok(arg);
    }

    fn expect_block(&mut self) -> Result<Block, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let mut block = Block {
            statements: Vec::new(),
        };

        while self.peek() != TokenKind::CloseBrace {
            let stmt = self.expect_statement()?;
            block.statements.push(stmt);
        }

        self.expect(TokenKind::CloseBrace)?;

        return Result::Ok(block);
    }

    fn expect_statement(&mut self) -> Result<Rc<Statement>, Error> {
        let stmt = match self.peek() {
            TokenKind::FunctionKeyword => {
                let stmt = self.expect_function()?;
                stmt
            }
            TokenKind::VariableKeyword => {
                let stmt = self.expect_variable()?;
                self.expect(TokenKind::Semicolon)?;
                stmt
            }
            TokenKind::ReturnKeyword => {
                self.expect(TokenKind::ReturnKeyword)?;
                let expr = match self.peek() {
                    TokenKind::Semicolon => Expression::Empty,
                    _ => self.expect_expression()?,
                };
                self.expect(TokenKind::Semicolon)?;
                let ret_stmt = Return { expr: expr };
                let stmt = self.add_statement(Statement::Return(ret_stmt));
                stmt
            }
            TokenKind::IfKeyword => {
                let if_token = self.consume_one_token()?;
                let expr = self.expect_expression()?;
                let block = self.expect_block()?;
                let if_stmt = If {
                    condition: expr,
                    block: block,
                };
                let stmt = self.add_statement(Statement::If(if_stmt));
                stmt
            }
            _ => {
                let expr = self.expect_expression()?;
                self.expect(TokenKind::Semicolon)?;
                let stmt = self.add_statement(Statement::Expression(expr));
                stmt
            }
        };

        return Result::Ok(stmt);
    }

    fn expect_function_call(&mut self) -> Result<FunctionCall, Error> {
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut args: Vec<Expression> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let expr = self.expect_expression()?;
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
        };

        return Result::Ok(expr);
    }

    fn expect_expression(&mut self) -> Result<Expression, Error> {
        let expr = match self.peek() {
            TokenKind::Identifier => {
                if self.peek_at(1) == TokenKind::OpenParenthesis {
                    let fx = self.expect_function_call()?;
                    Expression::FunctionCall(fx)
                } else {
                    let name_token = self.expect(TokenKind::Identifier)?;
                    let ident = Identifier {
                        name: name_token.value.clone(),
                        token: name_token,
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
                let inner = self.expect_expression()?;
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
                let rhs = self.expect_expression()?;
                let bin_expr = BinaryExpr {
                    left: Box::new(expr),
                    operator: op,
                    right: Box::new(rhs),
                };
                Expression::BinaryExpr(bin_expr)
            }
            _ => expr,
        };

        return Result::Ok(actual_expr);
    }

    fn expect_variable(&mut self) -> Result<Rc<Statement>, Error> {
        self.expect(TokenKind::VariableKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Equals)?;
        let expr = self.expect_expression()?;

        let var = Variable {
            name: name_token.value.clone(),
            type_: type_token.value.clone(),
            initializer: expr,
            name_token: name_token.clone(),
            type_token: type_token.clone(),
        };
        let var_stmt = self.add_statement(Statement::Variable(var));
        let var_sym = Symbol {
            name: name_token.value,
            kind: SymbolKind::Variable,
            declared_at: Rc::clone(&var_stmt),
        };
        self.symbols.push(var_sym);

        return Result::Ok(var_stmt);
    }

    fn expect_function(&mut self) -> Result<Rc<Statement>, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut arguments: Vec<FunctionArgument> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg = self.expect_function_argument()?;
            arguments.push(arg);

            if self.peek() == TokenKind::Comma {
                self.consume_one_token()?;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        self.expect(TokenKind::Colon)?;
        let return_type_token = self.expect(TokenKind::Identifier)?;

        let body_block = self.expect_block()?;

        let fx = Function {
            name: name_token.value.clone(),
            arguments: arguments,
            body: body_block,
            return_type: return_type_token.value.clone(),
            return_type_token: return_type_token,
            name_token: name_token.clone(),
        };
        let fn_stmt = self.add_statement(Statement::Function(fx));
        let fx_sym = Symbol {
            name: name_token.value,
            kind: SymbolKind::Function,
            declared_at: Rc::clone(&fn_stmt),
        };
        self.symbols.push(fx_sym);

        if let Statement::Function(fx) = fn_stmt.as_ref() {
            for arg in &fx.arguments {
                let fx_arg_sym = Symbol {
                    name: arg.name.clone(),
                    kind: SymbolKind::FunctionArgument,
                    declared_at: Rc::clone(&fn_stmt),
                };
                self.symbols.push(fx_arg_sym);
            }
        }

        return Result::Ok(fn_stmt);
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
        assert_eq!(0, fx.body.statements.len());
    }

    #[test]
    fn should_create_ast_from_variable_list() {
        let code = r###"
        var x: int = 1;
        var y: double = 2;
        "###;

        let ast = AST::from_code(code).unwrap();

        assert_eq!(2, ast.body.statements.len());

        if let Statement::Variable(x) = ast.body.statements[0].as_ref() {
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

        if let Statement::Variable(x) = ast.body.statements[1].as_ref() {
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
            ast.body.statements[0].as_ref()
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

        let fx_body = &fx.body;
        assert_eq!(3, fx_body.statements.len());
    }

    #[test]
    fn should_create_ast_from_binary_expr() {
        let code = r###"
            var x: int = 1 + do_thing(420, 69);
        "###;

        let ast = AST::from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let stmt = ast.body.statements[0].as_ref();

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

        let stmt = ast.body.statements[0].as_ref();

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

        assert_eq!(4, ast.symbols.len());
        assert_eq!("z", ast.symbols[0].name);
        assert_eq!(SymbolKind::Variable, ast.symbols[0].kind);
        assert_eq!("add", ast.symbols[1].name);
        assert_eq!(SymbolKind::Function, ast.symbols[1].kind);
        assert_eq!("x", ast.symbols[2].name);
        assert_eq!(SymbolKind::FunctionArgument, ast.symbols[2].kind);
        assert_eq!("y", ast.symbols[3].name);
        assert_eq!(SymbolKind::FunctionArgument, ast.symbols[3].kind);
    }

    #[test]
    fn should_parse_if() {
        let code = r###"
        if 1 == 2 {
            var x: int = 1;
        }
        "###;
        let ast = AST::from_code(code).unwrap();

        let if_stmt = match ast.body.statements[0].as_ref() {
            Statement::If(x) => x,
            _ => {
                assert!(false);
                panic!();
            }
        };

        let bin_expr = match &if_stmt.condition {
            Expression::BinaryExpr(x) => x,
            _=> panic!(),
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
