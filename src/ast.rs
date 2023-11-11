use crate::tokenizer::{Token, TokenKind};

#[derive(Debug, Clone)]
pub struct Source {
    statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Function(Function),
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
    name: Token,
    type_: Token,
}

#[derive(Debug, Clone)]
pub struct Function {
    name: Token,
    arguments: Vec<Argument>,
    body: Block,
    return_type: Token,
}

#[derive(Debug, Clone)]
pub struct Block {
    statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    name: Token,
    type_name: Token,
    initializer: Expression,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    name: Token,
    arguments: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct Error {
    message: String,
}

struct Builder {
    tokens: Vec<Token>,
    index: usize,
}

impl Builder {
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
            let stmt = self.expect_statement()?;
            block.statements.push(stmt);
        }

        self.expect(TokenKind::CloseBrace)?;

        return Result::Ok(block);
    }

    fn expect_statement(&mut self) -> Result<Statement, Error> {
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
            TokenKind::Identifier => {
                let expr = self.expect_expression()?;
                self.expect(TokenKind::Semicolon)?;
                Statement::Expression(expr)
            }
            _ => {
                panic!("Invalid statement.");
            }
        };
        return Result::Ok(stmt);
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
            type_name: type_name,
            initializer: expr,
        };

        return Result::Ok(var);
    }

    fn expect_function(&mut self) -> Result<Function, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut arguments: Vec<Argument> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg = self.expect_argument()?;

            arguments.push(arg);

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

pub fn create_ast(tokens: &[Token]) -> Result<Source, Error> {
    let mut builder = Builder {
        tokens: tokens.to_vec(),
        index: 0,
    };

    let mut block = Source {
        statements: Vec::new(),
    };

    while builder.index < tokens.len() {
        let stmt = builder.expect_statement()?;
        block.statements.push(stmt);
    }

    return Result::Ok(block);
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::*;
    use crate::ast::*;

    use super::create_ast;

    #[test]
    fn should_create_ast_from_function_with_empty_body() {
        let code = "fun do_thing(x: int, y: double): void {}";
        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();
        let stmt = &ast.statements[0];
        
        if let Statement::Function(fx) = stmt {
            assert_eq!(2, fx.arguments.len());
            assert_eq!("x", fx.arguments[0].name.value);
            assert_eq!("int", fx.arguments[0].type_.value);
            assert_eq!("y", fx.arguments[1].name.value);
            assert_eq!("double", fx.arguments[1].type_.value);
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

        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();
        
        assert_eq!(2, ast.statements.len());

        if let Statement::Variable(x) = &ast.statements[0] {
            assert_eq!("x", x.name.value);
            assert_eq!("int", x.type_name.value);

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
            assert_eq!("double", x.type_name.value);

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
        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();

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

        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();
        dbg!("{:?}", &ast);

        if let Statement::Function(fx) = &ast.statements[0] {
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

        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();
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

        let tokens = tokenize(code);
        let ast = create_ast(&tokens).unwrap();
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
