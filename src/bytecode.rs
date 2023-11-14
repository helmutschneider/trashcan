use crate::ast::{self, SymbolKind};

#[derive(Debug, Clone, Copy)]
struct Register(usize);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("r{}", self.0));
    }
}

#[derive(Debug, Clone)]
enum Value {
    Void,
    Integer(i64),
    Register(Register),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Integer(x) => f.write_str(&format!("{}", x)),
            Self::Register(reg) => reg.fmt(f),
            Self::Void => std::fmt::Result::Ok(()),
        };
    }
}

#[derive(Debug, Clone)]
enum Instruction {
    Function(ast::Symbol, Vec<ast::FunctionArgument>),
    Label(ast::Symbol),
    Alloc(Register, ast::Symbol),
    Store(Register, Value),
    Return(Value),
    Add(Register, Value, Value),
    FunctionCall(Register, ast::Symbol, Vec<Value>),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(sym, args) => {
                let args_s = (0..args.len())
                    .map(|arg_reg_num| format!("{}", Register(arg_reg_num)))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", sym.name, args_s)
            },
            Self::Label(sym) => {
                format!("{}:\n", sym.name)
            },
            Self::Alloc(reg, sym) => {
                format!("  {} = alloc {}", reg, sym.name)
            },
            Self::Store(reg, value) => {
                format!("  store {} {}", reg, value)
            },
            Self::Return(value) => {
                format!("  return {}", value)
            },
            Self::Add(reg, x, y) => {
                format!("  {} = add {} {}", reg, x, y)
            },
            Self::FunctionCall(reg, sym, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  {} = call {}({})", reg, sym.name, arg_s)
            },
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    instructions: Vec<Instruction>,
    registers: usize,
}

impl Bytecode {
    fn add_register(&mut self) -> Register {
        let reg = Register(self.registers);
        self.registers += 1;
        return reg;
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

fn compile_expression(bc: &mut Bytecode, ast: &ast::Ast, expr: &ast::Expression) -> Value {
    let value = match expr {
        ast::Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            Value::Integer(parsed)
        },
        ast::Expression::BinaryExpr(bin_expr) => {
            let lhs = compile_expression(bc, ast, &bin_expr.left);
            let rhs = compile_expression(bc, ast, &bin_expr.right);

            let res_reg = bc.add_register();
            bc.instructions.push(Instruction::Add(res_reg, lhs, rhs));
            Value::Register(res_reg)
        },
        ast::Expression::Identifier(ident) => {
            let ident_sym = ast
                .symbols
                .iter()
                .find(|s| {
                    s.name == ident.value
                        && (s.kind == SymbolKind::Local || s.kind == SymbolKind::FunctionArgument)
                })
                .unwrap();
            let alloc_reg: Register = {
                let mut found: Option<Register> = None;

                for k in &bc.instructions {
                    if let Instruction::Alloc(maybe_reg, maybe_sym) = k {
                        if maybe_sym == ident_sym {
                            found = Some(*maybe_reg);
                        }
                    }
                }

                found.expect(&format!(
                    "Could not find register for '{}'.",
                    ident_sym.name
                ))
            };
            Value::Register(alloc_reg)
        },
        ast::Expression::FunctionCall(call) => {
            let args: Vec<Value> = call.arguments.iter().map(|x| compile_expression(bc, ast, x)).collect();
            let result_reg = bc.add_register();
            let fn_sym = ast.get_symbol(&call.name.value, ast::SymbolKind::Function);
            bc.instructions.push(Instruction::FunctionCall(result_reg, fn_sym.clone(), args));
            Value::Register(result_reg)
        },
        _ => panic!(),
    };
    return value;
}

fn compile_statement(bc: &mut Bytecode, ast: &ast::Ast, stmt_index: &ast::StatementIndex) {
    let stmt = ast.get_statement(stmt_index);

    match stmt {
        ast::Statement::Function(fx) => {
            let regs_prev = bc.registers;
            bc.registers = fx.arguments.len();

            let fx_sym = ast.get_symbol(&fx.name.value, ast::SymbolKind::Function);
            let args: Vec<ast::FunctionArgument> = fx
                .arguments
                .iter()
                .map(|x| {
                    let arg_stmt = ast.get_statement(x);
                    return match arg_stmt {
                        ast::Statement::FunctionArgument(x) => x.clone(),
                        _ => panic!(),
                    };
                })
                .collect();
            bc.instructions
                .push(Instruction::Function(fx_sym.clone(), args));

            for arg_index in &fx.arguments {
                let arg_stmt = ast.get_statement(arg_index);
                let arg = if let ast::Statement::FunctionArgument(x) = arg_stmt {
                    x
                } else {
                    panic!()
                };
                let arg_sym = ast.get_symbol(&arg.name.value, ast::SymbolKind::FunctionArgument);
                let reg = bc.add_register();
                bc.instructions
                    .push(Instruction::Alloc(reg, arg_sym.clone()));
            }

            compile_statement(bc, ast, &fx.body);

            bc.registers = regs_prev;
        }
        ast::Statement::Block(block) => {
            for stmt_index in &block.statements {
                compile_statement(bc, ast, stmt_index);
            }
        }
        ast::Statement::Variable(var) => {
            let var_sym = ast.get_symbol(&var.name.value, ast::SymbolKind::Local);
            let alloc_reg = bc.add_register();
            bc.instructions
                .push(Instruction::Alloc(alloc_reg, var_sym.clone()));

            let store_val = compile_expression(bc, ast, &var.initializer);
            bc.instructions
                .push(Instruction::Store(alloc_reg, store_val));
        }
        ast::Statement::Return(expr) => {
            let ret_reg = compile_expression(bc, ast, expr);
            bc.instructions.push(Instruction::Return(ret_reg));
        }
        _ => {}
    }
}

fn from_code(code: &str) -> Bytecode {
    let ast = ast::from_code(code).unwrap();
    let mut bc = Bytecode {
        instructions: Vec::new(),
        registers: 0,
    };

    for stmt_index in &ast.body.statements {
        compile_statement(&mut bc, &ast, stmt_index);
    }

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

        assert_eq!(2, instructions.len());
        assert_eq!("  r0 = alloc x", format!("{}", instructions[0]));
        assert_eq!("  store r0 6", format!("{}", instructions[1]));
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = from_code(code);
        let instructions = bc.instructions;

        assert_eq!(4, instructions.len());
        assert_eq!("  r0 = alloc x", format!("{}", instructions[0]));
        assert_eq!("  store r0 6", format!("{}", instructions[1]));
        assert_eq!("  r1 = alloc y", format!("{}", instructions[2]));
        assert_eq!("  store r1 r0", format!("{}", instructions[3]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            1 + 2;
        "###;

        let bc = from_code(code);
        let instructions = bc.instructions;

        assert_eq!(1, instructions.len());
        assert_eq!("r0 = add 1 2", format!("{}", instructions[0]));
    }
    
    #[test]
    fn should_compile_yee() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }

            var x: int = add(5, 6);
        "###;

        let bc = from_code(code);

        println!("{}", bc);

        assert!(false);
    }
}
