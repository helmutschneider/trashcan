use crate::ast::{self, StatementIndex, SymbolKind};

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
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Variable(r) => r.fmt(f),
            Self::Integer(i) => i.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Label(String),
    Copy(Variable, Argument),
    Return(Argument),
    Add(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
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
                format!("{}:\n", name)
            }
            Self::Copy(dest, source) => {
                format!("  {} = copy {}", dest, source)
            }
            Self::Return(value) => {
                format!("  return {}", value)
            }
            Self::Add(reg, x, y) => {
                format!("  {} = add {}, {}", reg, x, y)
            }
            Self::Call(reg, name, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  {} = call {}({})", reg, name, arg_s)
            }
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub temporaries: usize,
}

impl Bytecode {
    fn add_temporary(&mut self) -> Variable {
        let temp = Variable(format!("%{}", self.temporaries));
        self.temporaries += 1;
        return temp;
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
    ast: &ast::Ast,
    expr: &ast::Expression,
    dest_var: Option<&Variable>,
) -> Argument {
    let value = match expr {
        ast::Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            Argument::Integer(parsed)
        }
        ast::Expression::BinaryExpr(bin_expr) => {
            let lhs = compile_expression(bc, ast, &bin_expr.left, None);
            let rhs = compile_expression(bc, ast, &bin_expr.right, None);
            let dest_ref = maybe_add_temp_variable(bc, dest_var);
            bc.instructions
                .push(Instruction::Add(dest_ref.clone(), lhs, rhs));
            Argument::Variable(dest_ref)
        }
        ast::Expression::Identifier(ident) => {
            Argument::Variable(Variable(ident.name.value.clone()))
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
                call.name.value.clone(),
                args,
            ));
            Argument::Variable(dest_ref)
        }
        _ => panic!(),
    };
    return value;
}

fn compile_function(bc: &mut Bytecode, ast: &ast::Ast, fx: &ast::Function) {
    let temps_prev = bc.temporaries;
    bc.temporaries = 0;

    let arg_vars: Vec<Variable> = fx
        .arguments
        .iter()
        .map(|fx_arg_index| {
            let stmt = ast.get_statement(fx_arg_index);
            let arg = match stmt {
                ast::Statement::FunctionArgument(fx_arg) => fx_arg,
                _ => panic!(),
            };
            Variable(arg.name.value.clone())
        })
        .collect();

    bc.instructions
        .push(Instruction::Function(fx.name.value.clone(), arg_vars));

    compile_statement(bc, ast, &fx.body);

    // add an implicit return statement if the function doesn't have one.
    if !matches!(bc.instructions.last().unwrap(), Instruction::Return(_)) {
        bc.instructions
            .push(Instruction::Return(Argument::Integer(0)));
    }

    bc.temporaries = temps_prev;
}

fn compile_statement(bc: &mut Bytecode, ast: &ast::Ast, stmt_index: &ast::StatementIndex) {
    let stmt = ast.get_statement(stmt_index);

    match stmt {
        ast::Statement::Function(fx) => {
            compile_function(bc, ast, fx);
        }
        ast::Statement::Block(block) => {
            for stmt_index in &block.statements {
                compile_statement(bc, ast, stmt_index);
            }
        }
        ast::Statement::Variable(var) => {
            let var_ref = Variable(var.name.value.clone());
            let init_arg = compile_expression(bc, ast, &var.initializer, Some(&var_ref));

            // literals and identifier expressions don't emit any stack
            // variables, so we need an implicit copy here.
            if matches!(var.initializer, ast::Expression::Literal(_) | ast::Expression::Identifier(_)) {
                bc.instructions.push(Instruction::Copy(var_ref, init_arg));
            }
        }
        ast::Statement::Return(ret) => {
            let ret_reg = compile_expression(bc, ast, &ret.expr, None);
            bc.instructions.push(Instruction::Return(ret_reg));
        }
        _ => {}
    }
}

pub fn from_code(code: &str) -> Bytecode {
    let ast = ast::from_code(code).unwrap();
    let mut bc = Bytecode {
        instructions: Vec::new(),
        temporaries: 0,
    };

    compile_statement(&mut bc, &ast, &ast.body);

    return bc;
}

#[cfg(test)]
mod tests {
    use crate::bytecode::*;

    #[test]
    fn should_compile_assignment() {
        let code = r###"
            var x: int = 6;
        "###;

        let bc = from_code(code);
        let instructions = bc.instructions;

        assert_eq!(1, instructions.len());
        assert_eq!("  x = copy 6", format!("{}", instructions[0]));
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = from_code(code);
        println!("cowabunga!! {bc}");
        let instructions = bc.instructions;

        assert_eq!(2, instructions.len());
        assert_eq!("  x = copy 6", format!("{}", instructions[0]));
        assert_eq!("  y = copy x", format!("{}", instructions[1]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = from_code(code);
        let instructions = bc.instructions;

        assert_eq!(1, instructions.len());
        assert_eq!("  x = add 1, 2", format!("{}", instructions[0]));
    }

    #[test]
    fn should_compile_function() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
        "###;

        let bc = from_code(code);
        let instructions = bc.instructions;

        assert_eq!(3, instructions.len());
        assert_eq!("add(x, y):", format!("{}", instructions[0]));
        assert_eq!("  %0 = add x, y", format!("{}", instructions[1]));
        assert_eq!("  return %0", format!("{}", instructions[2]));
    }
}
