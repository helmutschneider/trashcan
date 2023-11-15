use crate::ast::{self, StatementIndex, SymbolKind};

#[derive(Debug, Clone, Copy)]
pub struct Register(pub usize);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("%{}", self.0));
    }
}

#[derive(Debug, Clone)]
pub enum StoreArgument {
    Integer(i64),
    Register(Register),
}

impl std::fmt::Display for StoreArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Integer(x) => f.write_str(&format!("{}", x)),
            Self::Register(reg) => reg.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum LoadArgument {
    Integer(i64),
    Register(Register),
    Symbol(ast::Symbol),
}

impl std::fmt::Display for LoadArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Integer(x) => f.write_str(&format!("{}", x)),
            Self::Register(reg) => reg.fmt(f),
            Self::Symbol(sym) => f.write_str(&sym.name),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(ast::Symbol, Vec<Register>),
    Label(ast::Symbol),
    Load(Register, LoadArgument),
    Store(ast::Symbol, StoreArgument),
    Return(Register),
    Add(Register, Register, Register),
    FunctionCall(Register, ast::Symbol, Vec<Register>),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(sym, args) => {
                let args_s = (0..args.len())
                    .map(|i| format!("%{i}"))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", sym.name, args_s)
            }
            Self::Label(sym) => {
                format!("{}:\n", sym.name)
            }
            Self::Load(reg, arg) => {
                format!("  {:>2} = load {}", reg, arg)
            }
            Self::Store(sym, arg) => {
                format!("  {:>2} = store {}", sym.name, arg)
            }
            Self::Return(value) => {
                format!("  return {}", value)
            }
            Self::Add(reg, x, y) => {
                format!("  {} = add {}, {}", reg, x, y)
            }
            Self::FunctionCall(reg, sym, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  {} = call {}({})", reg, sym.name, arg_s)
            }
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub registers: usize,
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

fn find_parent_block(ast: &ast::Ast, stmt_index: StatementIndex) -> Option<StatementIndex> {
    let mut idx = stmt_index;

    loop {
        let stmt = ast.get_statement(&idx);

        if let ast::Statement::Block(_) = stmt {
            return Some(idx);
        }

        match stmt {
            ast::Statement::Expression(expr) => match expr {
                ast::Expression::BinaryExpr(bin_expr) => {
                    idx = bin_expr.parent_index;
                }
                ast::Expression::FunctionCall(call) => {
                    idx = call.parent_index;
                }
                ast::Expression::Identifier(ident) => {
                    idx = ident.parent_index;
                }
                _ => {
                    return None;
                }
            },
            ast::Statement::Return(ret) => {
                idx = ret.parent_index;
            }
            ast::Statement::Variable(var) => {
                idx = var.parent_index;
            }
            _ => {
                return None;
            }
        };
    }
}

fn find_symbols_in_scope(ast: &ast::Ast, at_stmt_index: StatementIndex) -> Vec<ast::Symbol> {
    let block_idx = find_parent_block(ast, at_stmt_index);
    let mut out: Vec<ast::Symbol> = Vec::new();

    if block_idx.is_none() {
        return out;
    }

    let block_idx = block_idx.unwrap();
    let block = ast.get_block(&block_idx);

    for inner_idx in &block.statements {
        if let ast::Statement::Variable(var) = ast.get_statement(inner_idx) {
            let var_sym = ast.get_symbol(&var.name.value, SymbolKind::Local);
            out.push(var_sym.clone());
        }
    }

    if let Some(block_parent_index) = &block.parent_index {
        if let ast::Statement::Function(fx) = ast.get_statement(block_parent_index) {
            for arg_idx in &fx.arguments {
                if let ast::Statement::FunctionArgument(fx_arg) = ast.get_statement(arg_idx) {
                    // this is bad... we're not using the function scope.
                    let arg_sym = ast.get_symbol(&fx_arg.name.value, SymbolKind::FunctionArgument);
                    out.push(arg_sym.clone());
                }
            }
        }
    }

    return out;
}

fn compile_expression(bc: &mut Bytecode, ast: &ast::Ast, expr: &ast::Expression) -> Register {
    let value = match expr {
        ast::Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            let load_reg = bc.add_register();
            bc.instructions
                .push(Instruction::Load(load_reg, LoadArgument::Integer(parsed)));
            load_reg
        }
        ast::Expression::BinaryExpr(bin_expr) => {
            let lhs = compile_expression(bc, ast, &bin_expr.left);
            let rhs = compile_expression(bc, ast, &bin_expr.right);

            let res_reg = bc.add_register();
            bc.instructions.push(Instruction::Add(res_reg, lhs, rhs));
            res_reg
        }
        ast::Expression::Identifier(ident) => {
            let symbols = find_symbols_in_scope(ast, ident.parent_index);
            let ident_sym = symbols.iter().find(|s| s.name == ident.name.value).unwrap();
            let load_reg = bc.add_register();
            let value = LoadArgument::Symbol(ident_sym.clone());
            bc.instructions.push(Instruction::Load(load_reg, value));

            load_reg
        }
        ast::Expression::FunctionCall(call) => {
            let args: Vec<Register> = call
                .arguments
                .iter()
                .map(|x| compile_expression(bc, ast, x))
                .collect();
            let result_reg = bc.add_register();
            let fn_sym = ast.get_symbol(&call.name.value, ast::SymbolKind::Function);
            bc.instructions
                .push(Instruction::FunctionCall(result_reg, fn_sym.clone(), args));
            result_reg
        }
        _ => panic!(),
    };
    return value;
}

fn compile_function(bc: &mut Bytecode, ast: &ast::Ast, fx: &ast::Function) {
    let regs_prev = bc.registers;
    bc.registers = fx.arguments.len();

    let fx_sym = ast.get_symbol(&fx.name.value, ast::SymbolKind::Function);
    let arg_regs: Vec<Register> = (0..fx.arguments.len()).map(|i| Register(i)).collect();

    bc.instructions
        .push(Instruction::Function(fx_sym.clone(), arg_regs));

    for i in 0..fx.arguments.len() {
        let arg_stmt_index = fx.arguments[i];
        let arg_stmt = match ast.get_statement(&arg_stmt_index) {
            ast::Statement::FunctionArgument(x) => x,
            _ => panic!(),
        };
        let arg_reg = Register(i);
        let arg_sym = ast.get_symbol(&arg_stmt.name.value, SymbolKind::FunctionArgument);

        bc.instructions.push(Instruction::Store(
            arg_sym.clone(),
            StoreArgument::Register(arg_reg),
        ));
    }

    compile_statement(bc, ast, &fx.body);

    // add an implicit return statement if the function doesn't have one.
    if !matches!(bc.instructions.last().unwrap(), Instruction::Return(_)) {
        let ret_reg = bc.add_register();
        bc.instructions
            .push(Instruction::Load(ret_reg, LoadArgument::Integer(0)));
        bc.instructions.push(Instruction::Return(ret_reg));
    }

    bc.registers = regs_prev;
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
            let var_sym = ast.get_symbol(&var.name.value, SymbolKind::Local);
            let store_reg = compile_expression(bc, ast, &var.initializer);
            let store_arg = StoreArgument::Register(store_reg);
            bc.instructions
                .push(Instruction::Store(var_sym.clone(), store_arg));
        }
        ast::Statement::Return(ret) => {
            let ret_reg = compile_expression(bc, ast, &ret.expr);
            bc.instructions.push(Instruction::Return(ret_reg));
        }
        _ => {}
    }
}

pub fn from_code(code: &str) -> Bytecode {
    let ast = ast::from_code(code).unwrap();
    let mut bc = Bytecode {
        instructions: Vec::new(),
        registers: 0,
    };
    let body = ast.get_block(&ast.body);

    for stmt_index in &body.statements {
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
        assert_eq!("  %0 = load 6", format!("{}", instructions[0]));
        assert_eq!("   x = store %0", format!("{}", instructions[1]));
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
        assert_eq!("  %0 = load 6", format!("{}", instructions[0]));
        assert_eq!("   x = store %0", format!("{}", instructions[1]));
        assert_eq!("  %1 = load x", format!("{}", instructions[2]));
        assert_eq!("   y = store %1", format!("{}", instructions[3]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = from_code(code);
        println!("{}", &bc);
        let instructions = bc.instructions;

        assert_eq!(4, instructions.len());
        assert_eq!("  %0 = load 1", format!("{}", instructions[0]));
        assert_eq!("  %1 = load 2", format!("{}", instructions[1]));
        assert_eq!("  %2 = add %0, %1", format!("{}", instructions[2]));
        assert_eq!("   x = store %2", format!("{}", instructions[3]));
    }
}
