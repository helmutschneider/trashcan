use crate::tokenizer::{Token, TokenKind};

#[derive(Debug, Clone)]
pub struct Ast {
    pub body: Block,
    pub symbols: Vec<Symbol>,
    statements: Vec<Statement>,
}

impl Ast {
    pub fn get_statement(&self, index: &StatementIndex) -> &Statement {
        return &self.statements[index.0];
    }

    pub fn get_block(&self, index: &StatementIndex) -> &Block {
        if let Statement::Block(block) = &self.statements[index.0] {
            return block;
        }
        panic!();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StatementIndex(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Global,
    Local,
    Function,
    FunctionArgument,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub index: StatementIndex,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Function(Function),
    FunctionArgument(FunctionArgument),
    Variable(Variable),
    Expression(Expression),
    Return(Expression),
    Block(Block),
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
pub struct FunctionArgument {
    pub name: Token,
    pub type_: Token,
    pub parent_index: StatementIndex,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Token,
    pub arguments: Vec<StatementIndex>,
    pub body: StatementIndex,
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
    pub parent_index: StatementIndex,
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
    symbols: Vec<Symbol>,
}

impl AstBuilder {
    fn peek(&self) -> TokenKind {
        return self.peek_at(0);
    }

    fn peek_at(&self, offset: usize) -> TokenKind {
        return self.tokens.get(self.index + offset).map(|t| t.kind).expect("End-of-file reached.");
    }

    fn add_statement(&mut self, stmt: Statement) -> StatementIndex {
        let stmt_index = StatementIndex(self.statements.len());
        self.statements.push(stmt);
        return stmt_index;
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

    fn expect_function_argument(&mut self) -> Result<StatementIndex, Error> {
        let arg_name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_name = self.expect(TokenKind::Identifier)?;

        let arg = FunctionArgument {
            name: arg_name,
            type_: type_name,
            parent_index: StatementIndex(0), // patched later.
        };
        let stmt_index = self.add_statement(Statement::FunctionArgument(arg));

        return Result::Ok(stmt_index);
    }

    fn expect_block(&mut self) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::OpenBrace)?;

        let mut statements: Vec<StatementIndex> = Vec::new();
        let mut block = Block {
            statements: Vec::new(),
        };

        while self.peek() != TokenKind::CloseBrace {
            let stmt_index = self.expect_statement()?;
            statements.push(stmt_index);
        }

        block.statements = statements.clone();

        self.expect(TokenKind::CloseBrace)?;

        let block_index = self.add_statement(Statement::Block(block));

        for stmt_index in statements {
            if let Statement::Variable(var) = &mut self.statements[stmt_index.0] {
                var.parent_index = block_index;
            }
        }

        return Result::Ok(block_index);
    }

    fn expect_statement(&mut self) -> Result<StatementIndex, Error> {
        let stmt_index = match self.peek() {
            TokenKind::FunctionKeyword => {
                let stmt_index = self.expect_function()?;
                stmt_index
            },
            TokenKind::VariableKeyword => {
                let stmt_index = self.expect_variable()?;
                self.expect(TokenKind::Semicolon)?;
                stmt_index
            },
            TokenKind::ReturnKeyword => {
                self.expect(TokenKind::ReturnKeyword)?;
                let expr = match self.peek() {
                    TokenKind::Semicolon => Expression::Empty,
                    _ => self.expect_expression()?,
                };
                self.expect(TokenKind::Semicolon)?;
                let stmt_index = self.add_statement(Statement::Return(expr));
                stmt_index
            },
            _ => {
                let expr = self.expect_expression()?;
                self.expect(TokenKind::Semicolon)?;
                let stmt_index = self.add_statement(Statement::Expression(expr));
                stmt_index
            }
        };

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

    fn expect_variable(&mut self) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::VariableKeyword)?;
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Colon)?;
        let type_name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::Equals)?;
        let expr = self.expect_expression()?;

        let var = Variable {
            name: name.clone(),
            type_: type_name,
            initializer: expr,
            parent_index: StatementIndex(0), // patched later.
        };
        let stmt_index = self.add_statement(Statement::Variable(var));
        let var_sym = Symbol {
            name: name.value.clone(),
            index: stmt_index,
            kind: SymbolKind::Local,
        };
        self.symbols.push(var_sym);

        return Result::Ok(stmt_index);
    }

    fn expect_function(&mut self) -> Result<StatementIndex, Error> {
        self.expect(TokenKind::FunctionKeyword)?;
        let name = self.expect(TokenKind::Identifier)?;
        self.expect(TokenKind::OpenParenthesis)?;

        let mut arguments: Vec<StatementIndex> = Vec::new();

        while self.peek() != TokenKind::CloseParenthesis {
            let arg_index = self.expect_function_argument()?;
            let arg = {
                if let Statement::FunctionArgument(arg) = &self.statements[arg_index.0] {
                    arg
                } else {
                    panic!();
                }
            };
            let arg_name = arg.name.value.clone();
            let arg_sym = Symbol {
                name: arg_name,
                index: arg_index,
                kind: SymbolKind::FunctionArgument,
            };
            self.symbols.push(arg_sym);

            arguments.push(arg_index);

            if self.peek() == TokenKind::Comma {
                self.index += 1;
            }
        }

        self.expect(TokenKind::CloseParenthesis)?;
        self.expect(TokenKind::Colon)?;
        let return_type = self.expect(TokenKind::Identifier)?;

        let body = self.expect_block()?;

        let fx = Function {
            name: name.clone(),
            arguments: arguments.clone(),
            body: body,
            return_type: return_type,
        };

        let stmt_index = self.add_statement(Statement::Function(fx));
        let fx_sym = Symbol {
            name: name.value,
            index: stmt_index,
            kind: SymbolKind::Function,
        };
        self.symbols.push(fx_sym);

        for arg_index in arguments {
            if let Statement::FunctionArgument(arg) = &mut self.statements[arg_index.0] {
                arg.parent_index = stmt_index;
            }
        }

        return Result::Ok(stmt_index);
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
        symbols: Vec::new(),
    };

    let mut ast = Ast {
        body: Block {
            statements: Vec::new(),
        },
        statements: Vec::new(),
        symbols: Vec::new(),
    };

    while builder.index < tokens.len() {
        let stmt_index = builder.expect_statement()?;
        ast.body.statements.push(stmt_index);
    }

    ast.statements = builder.statements;
    ast.symbols = builder.symbols;

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

        println!("{:?}", ast.symbols);
        // assert!(false);
        
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

            let body = ast.get_block(&fx.body);

            assert_eq!("void", fx.return_type.value);
            assert_eq!(0, body.statements.len());
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
            let body = ast.get_block(&fx.body);

            assert_eq!("main", fx.name.value);
            assert_eq!(3, body.statements.len());
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

    #[test]
    fn should_create_symbol_table() {
        let code = r###"
            fun add(x: int, y: int): int {
                var z: int = 6;
                return x + y;
            }
        "###;
        let ast = from_code(code).unwrap();

        assert_eq!(4, ast.symbols.len());
        assert_eq!("x", ast.symbols[0].name);
        assert_eq!(SymbolKind::FunctionArgument, ast.symbols[0].kind);
        assert_eq!("y", ast.symbols[1].name);
        assert_eq!(SymbolKind::FunctionArgument, ast.symbols[1].kind);
        assert_eq!("z", ast.symbols[2].name);
        assert_eq!(SymbolKind::Local, ast.symbols[2].kind);
        assert_eq!("add", ast.symbols[3].name);
        assert_eq!(SymbolKind::Function, ast.symbols[3].kind);
    }
}
