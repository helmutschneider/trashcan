use crate::{
    ast::{self, SymbolKind},
    tokenizer::TokenKind, util::Error, typer,
};

use crate::ast::ASTLike;

#[derive(Debug, Clone)]
pub struct Variable(pub String);

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.0.fmt(f);
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Variable(Variable),
    Integer(i64),
    String(String),
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Variable(r) => r.fmt(f),
            Self::Integer(i) => i.fmt(f),
            Self::String(s) => f.write_str(&format!("\"{}\"", s)),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Label(String),
    Copy(Variable, Argument),
    Ret(Argument),
    Add(Variable, Argument, Argument),
    Sub(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
    Eq(Variable, Argument, Argument),
    Jne(String, Argument, Argument),
    Noop,
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(name, args) => {
                let args_s = args
                    .iter()
                    .map(|v| v.0.clone())
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", name, args_s)
            }
            Self::Label(name) => {
                format!("{}:", name)
            }
            Self::Copy(dest_var, source) => {
                format!("  {} = copy {}", dest_var, source)
            }
            Self::Ret(value) => {
                format!("  ret {}", value)
            }
            Self::Add(dest_var, x, y) => {
                format!("  {} = add {}, {}", dest_var, x, y)
            }
            Self::Sub(dest_var, x, y) => {
                format!("  {} = sub {}, {}", dest_var, x, y)
            }
            Self::Call(dest_var, name, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  {} = call {}({})", dest_var, name, arg_s)
            }
            Self::Eq(dest_var, a, b) => {
                format!("  {} = eq {}, {}", dest_var, a, b)
            }
            Self::Jne(to_label, a, b) => {
                format!("  jne {}, {}, {}", to_label, a, b)
            }
            Self::Noop => format!("  noop"),
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: usize,
    pub temporaries: usize,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = typer::Typer::from_code(code)?;
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
            temporaries: 0,
        };
    
        compile_block(&mut bc, &typer.ast, typer.ast.body());
    
        return Ok(bc);
    }

    fn add_temporary(&mut self) -> Variable {
        let temp = Variable(format!("%{}", self.temporaries));
        self.temporaries += 1;
        return temp;
    }

    fn add_label(&mut self) -> String {
        let label = format!(".LB{}", self.labels);
        self.labels += 1;
        return label;
    }
}

impl std::fmt::Display for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instr in &self.instructions {
            instr.fmt(f)?;
            f.write_str("\n")?;
        }
        return std::fmt::Result::Ok(());
    }
}

fn maybe_add_temp_variable(bc: &mut Bytecode, dest_var: Option<&Variable>) -> Variable {
    return match dest_var {
        Some(v) => v.clone(),
        None => bc.add_temporary(),
    };
}

fn compile_expression(
    bc: &mut Bytecode,
    ast: &ast::AST,
    expr: &ast::Expression,
    dest_var: Option<&Variable>,
) -> Argument {
    let value = match expr {
        ast::Expression::IntegerLiteral(x) => {
            Argument::Integer(x.value)
        }
        ast::Expression::StringLiteral(s) => {
            Argument::String(s.value.clone())
        }
        ast::Expression::BinaryExpr(bin_expr) => {
            let lhs = compile_expression(bc, ast, &bin_expr.left, None);
            let rhs = compile_expression(bc, ast, &bin_expr.right, None);
            let dest_ref = maybe_add_temp_variable(bc, dest_var);
            let instr = match bin_expr.operator.kind {
                TokenKind::Plus => Instruction::Add(dest_ref.clone(), lhs, rhs),
                TokenKind::Minus => Instruction::Sub(dest_ref.clone(), lhs, rhs),
                TokenKind::DoubleEquals => Instruction::Eq(dest_ref.clone(), lhs, rhs),
                _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
            };
            bc.instructions.push(instr);

            Argument::Variable(dest_ref)
        }
        ast::Expression::Identifier(ident) => {
            Argument::Variable(Variable(ident.name.clone()))
        }
        ast::Expression::FunctionCall(call) => {
            let args: Vec<Argument> = call
                .arguments
                .iter()
                .map(|x| compile_expression(bc, ast, x, None))
                .collect();
            let dest_ref = maybe_add_temp_variable(bc, dest_var);
            bc.instructions.push(Instruction::Call(
                dest_ref.clone(),
                call.name.clone(),
                args,
            ));
            Argument::Variable(dest_ref)
        }
        _ => panic!(),
    };
    return value;
}

fn compile_function(bc: &mut Bytecode, ast: &ast::AST, fx: &ast::Function) {
    let temps_prev = bc.temporaries;
    bc.temporaries = 0;

    let arg_vars: Vec<Variable> = fx
        .arguments
        .iter()
        .map(|fx_arg| {
            Variable(fx_arg.name.clone())
        })
        .collect();

    bc.instructions
        .push(Instruction::Function(fx.name.clone(), arg_vars));

    let fx_body = ast.get_block(fx.body);

    compile_block(bc, ast, fx_body);

    // add an implicit return statement if the function doesn't have one.
    if !matches!(bc.instructions.last().unwrap(), Instruction::Ret(_)) {
        bc.instructions
            .push(Instruction::Ret(Argument::Integer(0)));
    }

    bc.temporaries = temps_prev;
}

fn compile_block(bc: &mut Bytecode, ast: &ast::AST, block: &ast::Block) {
    for stmt in &block.statements {
        let stmt = ast.get_statement(*stmt);
        compile_statement(bc, ast, stmt);
    }
}

fn compile_statement(bc: &mut Bytecode, ast: &ast::AST, stmt: &ast::Statement) {
    match stmt {
        ast::Statement::Function(fx) => {
            compile_function(bc, ast, fx);
        }
        ast::Statement::Block(block) => {
            compile_block(bc, ast, block);
        }
        ast::Statement::Variable(var) => {
            let var_ref = Variable(var.name.clone());
            let init_arg = compile_expression(bc, ast, &var.initializer, Some(&var_ref));

            // literals and identifier expressions don't emit any stack
            // variables, so we need an implicit copy here.
            if matches!(
                var.initializer,
                ast::Expression::IntegerLiteral(_) | ast::Expression::StringLiteral(_) | ast::Expression::Identifier(_)
            ) {
                bc.instructions.push(Instruction::Copy(var_ref, init_arg));
            }
        }
        ast::Statement::Return(ret) => {
            let ret_reg = compile_expression(bc, ast, &ret.expr, None);
            bc.instructions.push(Instruction::Ret(ret_reg));
        }
        ast::Statement::If(if_stmt) => {
            let label_after_block = bc.add_label();
            let bin_expr = match &if_stmt.condition {
                ast::Expression::BinaryExpr(x) => x,
                _ => panic!(),
            };
            let left_arg = compile_expression(bc, ast, &bin_expr.left, None);
            let right_arg = compile_expression(bc, ast, &bin_expr.right, None);

            // jump over the true-block if the result isn't truthy.
            bc.instructions.push(Instruction::Jne(
                label_after_block.clone(),
                left_arg,
                right_arg,
            ));

            let if_block = ast.get_block(if_stmt.block);

            compile_block(bc, ast, if_block);
            bc.instructions.push(Instruction::Label(label_after_block));
        }
        ast::Statement::Expression(expr) => {
            compile_expression(bc, ast, expr, None);
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode::*;

    #[test]
    fn should_compile_assignment() {
        let code = r###"
            var x: int = 6;
        "###;

        let bc = Bytecode::from_code(code).unwrap();

        let expected = r###"
        x = copy 6
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        x = copy 6
        y = copy x
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            x = add 1, 2
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_function() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            add(x, y):
                %0 = add x, y
                ret %0
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_double_equals() {
        let code = r###"
        var x: int = 2;
        var y: bool = (x + 1) == 3;
    "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            x = copy 2
            %0 = add x, 1
            y = eq %0, 3
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_if_with_jump() {
        let code = r###"
        var x: bool = 1 == 2;
        if x {
            var y: int = 42;
        }
        var z: int = 3;
    "###;
        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            x = eq 1, 2
            jne .LB0, x, 7
            y = copy 42
        .LB0:
            z = copy 3
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    fn assert_bytecode_matches(expected: &str, bc: &crate::bytecode::Bytecode) {
        let expected_lines: Vec<&str> = expected.trim().lines().map(|l| l.trim()).collect();
        let bc_s = bc.to_string();
        let bc_lines: Vec<&str> = bc_s.trim().lines().map(|l| l.trim()).collect();

        assert_eq!(expected_lines.len(), bc_lines.len());

        for i in 0..expected_lines.len() {
            assert_eq!(expected_lines[i], bc_lines[i]);
        }
    }
}
