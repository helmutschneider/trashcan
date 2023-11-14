use crate::tokenizer::{Token, TokenKind};

#[derive(Debug, Clone)]
pub struct Ast {
    pub body: Block,
    statements: Vec<Statement>,
}

impl Ast {
    pub fn get_statement(&self, index: &StatementIndex) -> &Statement {
        return &self.statements[index.0];
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StatementIndex(pub usize);

#[derive(Debug, Clone)]
pub enum SymbolKind {
    Global,
    Local,
    Function,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
    index: StatementIndex,
    kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Function(Function),
    FunctionArgument(Argument),
    Variable(Variable),
    Expression(Expression),
    Return(Expression),
}

#[derive(Debug, Clone)]
pub enum Expression {
    Empty,
    Identifier(Token),
    Literal(Token),
    FunctionCall(FunctionCall),
    BinaryExpr(BinaryExpr),
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: Token,
    pub type_: Token,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Token,
    pub arguments: Vec<StatementIndex>,
    pub body: Block,
    pub return_type: Token,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<StatementIndex>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: Token,
    pub type_: Token,
    pub initializer: Expression,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: Token,
    pub arguments: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct Error {
    message: String,
}

struct AstBuilder {
    tokens: Vec<Token>,
    index: usize,
    statements: Vec<Statement>,
}

impl AstBuilder {
    fn peek(&self) -> TokenKind {
        return self.peek_at(0);
    }

    fn peek_at(&self, offset: usize) -> TokenKind {
        return self.tokens.get(self.index + offset).map(|t| t.kind).expect("End-of-file reached.");
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, Error> {
        if self.index >= self.tokens.len() {
            let err = Error {
                message: format!("Syntax error: expected {:?}, found end of file.", kind),
            };
            return Result::Err(err);
        }

        let token = &self.tokens[self.index];

        if kind != token.kind {
            let err = Error {
                message: format!("Syntax error: expected {:?}, found '{:?}' at index {}.", kind, token.kind, token.source_index),
            };
            return Result::Err(err);
        }

        self.index += 1;

        return Result::Ok(token.clone());
    }

    fn expect_argument(&mut self) -> Result<Argument, Error> {
        let arg_name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_name = self.expect(TokenKind::Identifier)?;

        let arg = Argument {
            name: arg_name,
            type_: type_name,
        };

        return Result::Ok(arg);
    }

    fn expect_block(&mut self) -> Result<Block, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let mut block = Block {
            statements: Vec::new(),
        };

        while self.peek() != TokenKind::CloseBrace {
            let stmt_index = self.expect_statement()?;
            block.statements.push(stmt_index);
        }

        self.expect(TokenKind::CloseBrace)?;

        return Result::Ok(block);
    }

    fn expect_statement(&mut self) -> Result<StatementIndex, Error> {
        let stmt = match self.peek() {
            TokenKind::FunctionKeyword => {
                let fx = self.expect_function()?;
                Statement::Function(fx)
            },
            TokenKind::VariableKeyword => {
                let var = self.expect_variable()?;
                self.expect(TokenKind::Semicolon)?;
                Statement::Variable(var)
            },
            TokenKind::ReturnKeyword => {
                self.expect(TokenKind::ReturnKeyword)?;
                let expr = match self.peek() {
                    TokenKind::Semicolon => Expression::Empty,
                    _ => self.expect_expression()?,
                };
                self.expect(TokenKind::Semicolon)?;
                Statement::Return(expr)
            },
            _ => {
                let expr = self.expect_expression()?;
                self.expect(TokenKind::Semicolon)?;
                Statement::Expression(expr)
            }
        };

        let stmt_index = StatementIndex(self.statements.len());
        self.statements.push(stmt);

        return Result::Ok(stmt_index);
    }

    fn expect_function_call(&mut self) -> Result<FunctionCall, Error> {
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut args: Vec<Expression> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let expr = self.expect_expression()?;
            args.push(expr);

            if self.peek() == TokenKind::Comma {
                self.index += 1;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        
        let expr = FunctionCall {
            name: name,
            arguments: args,
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
                    let ident_name = self.expect(TokenKind::Identifier)?;
                    Expression::Identifier(ident_name)
                }
            },
            TokenKind::Integer => {
                let token = self.expect(TokenKind::Integer)?;
                Expression::Literal(token)
            },
            _ => panic!("Invalid expression."),
        };

        let actual_expr = match self.peek() {
            // TODO: these branches are identical, but we don't have a method
            //   to eat an arbitrary token at the moment.
            TokenKind::Plus => {
                let op = self.expect(TokenKind::Plus)?;
                let right_hand = self.expect_expression()?;
                let bin_expr = BinaryExpr {
                    left: Box::new(expr),
                    operator: op,
                    right: Box::new(right_hand),
                };
                Expression::BinaryExpr(bin_expr)
            }
            TokenKind::Minus => {
                let op = self.expect(TokenKind::Minus)?;
                let right_hand = self.expect_expression()?;
                let bin_expr = BinaryExpr {
                    left: Box::new(expr),
                    operator: op,
                    right: Box::new(right_hand),
                };
                Expression::BinaryExpr(bin_expr)
            },
            _ => expr,
        };

        return Result::Ok(actual_expr);
    }

    fn expect_variable(&mut self) -> Result<Variable, Error> {
        self.expect(TokenKind::VariableKeyword)?;
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Equals)?;
        let expr = self.expect_expression()?;

        let var = Variable {
            name: name,
            type_: type_name,
            initializer: expr,
        };

        return Result::Ok(var);
    }

    fn expect_function(&mut self) -> Result<Function, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut arguments: Vec<StatementIndex> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg = self.expect_argument()?;
            let stmt_index = StatementIndex(self.statements.len());
            self.statements.push(Statement::FunctionArgument(arg));

            arguments.push(stmt_index);

            if self.peek() == TokenKind::Comma {
                self.index += 1;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        self.expect(TokenKind::Colon)?;
        let return_type = self.expect(TokenKind::Identifier)?;

        let body = self.expect_block()?;

        let fx = Function {
            name: name,
            arguments: arguments,
            body: body,
            return_type: return_type,
        };

        return Result::Ok(fx);
    }
}

pub fn from_code(code: &str) -> Result<Ast, Error> {
    let tokens = crate::tokenizer::tokenize(code);
    return from_tokens(&tokens);
}

pub fn from_tokens(tokens: &[Token]) -> Result<Ast, Error> {
    let mut builder = AstBuilder {
        tokens: tokens.to_vec(),
        index: 0,
        statements: Vec::new(),
    };

    let mut ast = Ast {
        body: Block {
            statements: Vec::new(),
        },
        statements: Vec::new(),
    };

    while builder.index < tokens.len() {
        let stmt_index = builder.expect_statement()?;
        ast.body.statements.push(stmt_index);
    }

    ast.statements = builder.statements;

    return Result::Ok(ast);
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::*;
    use crate::ast::*;

    use super::from_code;

    #[test]
    fn should_create_ast_from_function_with_empty_body() {
        let code = "fun do_thing(x: int, y: double): void {}";
        let ast = from_code(&code).unwrap();
        let fun_index = ast.body.statements[0].0;
        let stmt = &ast.statements[fun_index];
        
        if let Statement::Function(fx) = stmt {
            assert_eq!(2, fx.arguments.len());

            let arg0_index = fx.arguments[0].0;
            let arg1_index = fx.arguments[1].0;

            if let Statement::FunctionArgument(arg) = &ast.statements[arg0_index] {
                assert_eq!("x", arg.name.value);
                assert_eq!("int", arg.type_.value);
            } else {
                panic!();
            }

            if let Statement::FunctionArgument(arg) = &ast.statements[arg1_index] {
                assert_eq!("y", arg.name.value);
                assert_eq!("double", arg.type_.value);
            } else {
                panic!();
            }

            assert_eq!("void", fx.return_type.value);
            assert_eq!(0, fx.body.statements.len());
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_create_ast_from_variable_list() {
        let code = r###"
        var x: int = 1;
        var y: double = 2;
        "###;

        let ast = from_code(code).unwrap();
        
        assert_eq!(2, ast.statements.len());

        if let Statement::Variable(x) = &ast.statements[0] {
            assert_eq!("x", x.name.value);
            assert_eq!("int", x.type_.value);

            if let Expression::Literal(init) = &x.initializer {
                assert_eq!("1", init.value);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }

        if let Statement::Variable(x) = &ast.statements[1] {
            assert_eq!("y", x.name.value);
            assert_eq!("double", x.type_.value);

            if let Expression::Literal(init) = &x.initializer {
                assert_eq!("2", init.value);
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
        let ast = from_code(code).unwrap();

        dbg!("{:?}", &ast);

        if let Statement::Expression(Expression::FunctionCall(call)) = &ast.statements[0] {
            assert_eq!("call_me_maybe", call.name.value);
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

        let ast = from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let main_idx = ast.body.statements[0].0;

        if let Statement::Function(fx) = &ast.statements[main_idx] {
            assert_eq!("main", fx.name.value);
            assert_eq!(3, fx.body.statements.len());
        } else {
            assert!(false);
        }
    }

    #[test]
    fn should_create_ast_from_binary_expr() {
        let code = r###"
            var x: int = 1 + do_thing(420, 69);
        "###;

        let ast = from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let stmt = &ast.statements[0];

        if let Statement::Variable(v) = stmt {
            if let Expression::BinaryExpr(expr) = &v.initializer {
                assert_eq!(TokenKind::Plus, expr.operator.kind);
                assert_eq!(true, matches!(*expr.left, Expression::Literal(_)));
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

        let ast = from_code(code).unwrap();
        dbg!("{:?}", &ast);

        let stmt = &ast.statements[0];

        if let Statement::Variable(v) = stmt {
            if let Expression::BinaryExpr(expr) = &v.initializer {
                assert_eq!(true, matches!(*expr.left, Expression::Literal(_)));
                assert_eq!(true, matches!(*expr.right, Expression::BinaryExpr(_)));
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }
}
