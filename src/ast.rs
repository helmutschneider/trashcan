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
    Identifier(Token),
    Literal(Token),
    FunctionCall(FunctionCall),
}

#[derive(Debug, Clone)]
pub struct Argument {
    name: Token,
    type_name: Token,
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

pub struct Builder {
    tokens: Vec<Token>,
    index: usize,
}

impl Builder {
    fn peek(&self) -> TokenKind {
        return self.tokens[self.index].kind;
    }

    fn expect<const N: usize>(&mut self, kinds: [TokenKind; N]) -> Result<Token, Error> {
        if self.index >= self.tokens.len() {
            let kind_names = kinds.iter().map(|k| format!("'{:?}'", k)).collect::<Vec<String>>().join(", ");
            let err = Error {
                message: format!("Syntax error: expected {}, found end of file.", kind_names),
            };
            return Result::Err(err);
        }

        let token = &self.tokens[self.index];

        if !kinds.contains(&token.kind) {
            let kind_names = kinds.iter().map(|k| format!("'{:?}'", k)).collect::<Vec<String>>().join(", ");
            let err = Error {
                message: format!("Syntax error: expected {}, found '{:?}' at index {}.", kind_names, token.kind, token.source_index),
            };
            return Result::Err(err);
        }

        self.index += 1;

        return Result::Ok(token.clone());
    }

    fn maybe_expect_argument(&mut self) -> Result<Option<Argument>, Error> {
        if self.peek() != TokenKind::Identifier {
            return Result::Ok(None);
        }

        let arg_name = self.expect([TokenKind::Identifier])?;
        self.expect([TokenKind::Colon])?;
        let type_name = self.expect([TokenKind::Identifier])?;

        if self.peek() == TokenKind::Comma {
            self.index += 1;
        }

        let arg = Argument {
            name: arg_name,
            type_name: type_name,
        };

        return Result::Ok(Some(arg));
    }

    fn expect_block(&mut self) -> Result<Block, Error> {
        self.expect([TokenKind::OpenBrace])?;

        let mut block = Block {
            statements: Vec::new(),
        };

        while self.peek() != TokenKind::CloseBrace {
            let stmt = self.expect_statement()?;
            block.statements.push(stmt);
        }

        self.expect([TokenKind::CloseBrace])?;

        return Result::Ok(block);
    }

    fn expect_statement(&mut self) -> Result<Statement, Error> {
        let token = &self.tokens[self.index];
        let stmt = match token.kind {
            TokenKind::Function => {
                let fx = self.expect_function()?;
                Statement::Function(fx)
            },
            TokenKind::Variable => {
                let var = self.expect_variable()?;
                self.expect([TokenKind::Semicolon])?;
                Statement::Variable(var)
            },
            TokenKind::Identifier => {
                let expr = self.expect_expression()?;
                self.expect([TokenKind::Semicolon])?;
                Statement::Expression(expr)
            }
            _ => {
                self.expect([TokenKind::Function, TokenKind::Variable])?;
                panic!("");
            }
        };
        return Result::Ok(stmt);
    }

    fn expect_function_call(&mut self, name: Token) -> Result<FunctionCall, Error> {
        self.expect([TokenKind::OpenParenthesis])?;

        let mut args: Vec<Expression> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let expr = self.expect_expression()?;
            args.push(expr);

            if self.peek() == TokenKind::Comma {
                self.index += 1;
            }
        }

        self.expect([TokenKind::CloseParenthesis])?;
        
        let expr = FunctionCall {
            name: name,
            arguments: args,
        };

        return Result::Ok(expr);
    }

    fn expect_expression(&mut self) -> Result<Expression, Error> {
        let value = self.expect([TokenKind::Integer, TokenKind::Identifier])?;

        if self.peek() == TokenKind::OpenParenthesis {
            let call = self.expect_function_call(value)?;
            let expr = Expression::FunctionCall(call);

            return Result::Ok(expr);
        }

        let expr = match value.kind {
            TokenKind::Identifier => Expression::Identifier(value),
            TokenKind::Integer => Expression::Literal(value),
            _ => panic!(),
        };

        return Result::Ok(expr);
    }

    fn expect_variable(&mut self) -> Result<Variable, Error> {
        self.expect([TokenKind::Variable])?;
        let name = self.expect([TokenKind::Identifier])?;
        self.expect([TokenKind::Colon])?;
        let type_name = self.expect([TokenKind::Identifier])?;
        self.expect([TokenKind::Equals])?;
        let expr = self.expect_expression()?;

        let var = Variable {
            name: name,
            type_name: type_name,
            initializer: expr,
        };

        return Result::Ok(var);
    }

    fn expect_function(&mut self) -> Result<Function, Error> {
        self.expect([TokenKind::Function])?;
        let name = self.expect([TokenKind::Identifier])?;
        self.expect([TokenKind::OpenParenthesis])?;

        let mut arguments: Vec<Argument> = Vec::new();

        while let Some(arg) = self.maybe_expect_argument()? {
            arguments.push(arg);
        }

        self.expect([TokenKind::CloseParenthesis])?;
        self.expect([TokenKind::Colon])?;
        let return_type = self.expect([TokenKind::Identifier])?;

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
            assert_eq!("int", fx.arguments[0].type_name.value);
            assert_eq!("y", fx.arguments[1].name.value);
            assert_eq!("double", fx.arguments[1].type_name.value);
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
}
