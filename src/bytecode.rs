use crate::ast::{self, StatementIndex, SymbolKind};

#[derive(Debug, Clone, Copy)]
pub struct Temporary(pub usize);

impl std::fmt::Display for Temporary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("%{}", self.0));
    }
}

#[derive(Debug, Clone)]
pub enum Reference {
    Temporary(Temporary),
    Variable(ast::Symbol),
}

impl Reference {
    pub fn name(&self) -> String {
        return match self {
            Self::Temporary(t) => t.to_string(),
            Self::Variable(v) => v.name.clone(),
        };
    }
}

impl std::fmt::Display for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Temporary(t) => t.fmt(f),
            Self::Variable(v) => v.name.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Reference(Reference),
    Integer(i64),
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Reference(r) => r.fmt(f),
            Self::Integer(i) => i.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(ast::Symbol, Vec<ast::Symbol>),
    Label(ast::Symbol),
    Copy(Reference, Argument),
    Return(Argument),
    Add(Temporary, Argument, Argument),
    FunctionCall(Temporary, ast::Symbol, Vec<Argument>),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(sym, args) => {
                let args_s = args
                    .iter()
                    .map(|a| a.name.clone())
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", sym.name, args_s)
            }
            Self::Label(sym) => {
                format!("{}:\n", sym.name)
            }
            Self::Copy(dest, source) => {
                format!("  {} = {}", dest, source)
            },
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
    pub temporaries: usize,
}

impl Bytecode {
    fn add_temporary(&mut self) -> Temporary {
        let temp = Temporary(self.temporaries);
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

fn compile_expression(bc: &mut Bytecode, ast: &ast::Ast, expr: &ast::Expression) -> Argument {
    let value = match expr {
        ast::Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            Argument::Integer(parsed)
        }
        ast::Expression::BinaryExpr(bin_expr) => {
            let lhs = compile_expression(bc, ast, &bin_expr.left);
            let rhs = compile_expression(bc, ast, &bin_expr.right);

            let res_temp = bc.add_temporary();
            bc.instructions.push(Instruction::Add(res_temp, lhs, rhs));
            Argument::Reference(Reference::Temporary(res_temp))
        }
        ast::Expression::Identifier(ident) => {
            let symbols = find_symbols_in_scope(ast, ident.parent_index);
            let ident_sym = symbols.iter().find(|s| s.name == ident.name.value).unwrap();
            Argument::Reference(Reference::Variable(ident_sym.clone()))
        }
        ast::Expression::FunctionCall(call) => {
            let args: Vec<Argument> = call
                .arguments
                .iter()
                .map(|x| compile_expression(bc, ast, x))
                .collect();
            let result_temp = bc.add_temporary();
            let fn_sym = ast.get_symbol(&call.name.value, ast::SymbolKind::Function);
            bc.instructions
                .push(Instruction::FunctionCall(result_temp, fn_sym.clone(), args));
            Argument::Reference(Reference::Temporary(result_temp))
        }
        _ => panic!(),
    };
    return value;
}

fn compile_function(bc: &mut Bytecode, ast: &ast::Ast, fx: &ast::Function) {
    let temps_prev = bc.temporaries;
    bc.temporaries = 0;

    let fx_sym = ast.get_symbol(&fx.name.value, ast::SymbolKind::Function);
    let arg_syms: Vec<ast::Symbol> = fx.arguments.iter().map(|fx_arg_index| {
        let stmt = ast.get_statement(fx_arg_index);
        let arg = match stmt {
            ast::Statement::FunctionArgument(fx_arg) => fx_arg,
            _ => panic!(),
        };
        let arg_sym = ast.get_symbol(&arg.name.value, SymbolKind::FunctionArgument);
        return arg_sym.clone();
    }).collect();

    bc.instructions
        .push(Instruction::Function(fx_sym.clone(), arg_syms));

    compile_statement(bc, ast, &fx.body);

    // add an implicit return statement if the function doesn't have one.
    if !matches!(bc.instructions.last().unwrap(), Instruction::Return(_)) {
        let ret_temp = bc.add_temporary();
        let ret_ref = Reference::Temporary(ret_temp);
        bc.instructions
            .push(Instruction::Copy(ret_ref.clone(), Argument::Integer(0)));
        bc.instructions.push(Instruction::Return(Argument::Reference(ret_ref)));
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
            let var_sym = ast.get_symbol(&var.name.value, SymbolKind::Local);
            let store_ref = compile_expression(bc, ast, &var.initializer);
            let var_ref = Reference::Variable(var_sym.clone());
            bc.instructions
                .push(Instruction::Copy(var_ref, store_ref));
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
        temporaries: 0,
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

        println!("{bc}");

        let instructions = bc.instructions;

        assert_eq!(1, instructions.len());
        assert_eq!("  x = 6", format!("{}", instructions[0]));
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = from_code(code);
        println!("{bc}");

        let instructions = bc.instructions;

        assert_eq!(2, instructions.len());
        assert_eq!("  x = 6", format!("{}", instructions[0]));
        assert_eq!("  y = x", format!("{}", instructions[1]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = from_code(code);
        println!("{}", &bc);
        let instructions = bc.instructions;

        assert_eq!(2, instructions.len());
        assert_eq!("  %0 = add 1, 2", format!("{}", instructions[0]));
        assert_eq!("  x = %0", format!("{}", instructions[1]));
    }

    #[test]
    fn should_compile_function() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
        "###;

        let bc = from_code(code);
        println!("{}", &bc);
        let instructions = bc.instructions;

        assert_eq!(3, instructions.len());
        assert_eq!("add(x, y):", format!("{}", instructions[0]));
        assert_eq!("  %0 = add x, y", format!("{}", instructions[1]));
        assert_eq!("  return %0", format!("{}", instructions[2]));
    }

    #[test]
    fn should_do_thing() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y + 420 + 69;
            }
        "###;

        let bc = from_code(code);
        println!("{}", &bc);
        let instructions = bc.instructions;

        assert_eq!(3, instructions.len());
        assert_eq!("add(x, y):", format!("{}", instructions[0]));
        assert_eq!("  %0 = add x, y", format!("{}", instructions[1]));
        assert_eq!("  return %0", format!("{}", instructions[2]));
    }
}
