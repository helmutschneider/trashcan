use crate::ast::Expression;
use crate::ast::Statement;
use crate::ast::Error;
use crate::ast::Variable;
use crate::tokenizer::TokenKind;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Register(i64);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("r{}", self.0);
        f.write_str(&s);
        return std::fmt::Result::Ok(());
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Void,
    Integer(i64),
    Register(Register),
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Integer(x) => {
                let s = format!("{}", x);
                f.write_str(&s)
            }
            Self::Register(reg) => reg.fmt(f),
            Self::Void => std::fmt::Result::Ok(()),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolKind {
    Variable,
    FunctionArgument(Register),
    Function,
    Label,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Alloc(Register, Symbol),
    Load(Register, Argument),
    Store(Register, Argument),
    Add(Register, Argument, Argument),
    Return(Argument),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::Alloc(reg, sym) => {
                format!("{} = alloc {}", reg, sym.name)
            },
            Self::Load(reg, val) => {
                format!("{} = load {}", reg, val)
            },
            Self::Store(reg, value) => {
                format!("store {} {}", reg, value)
            },
            Self::Add(reg, a, b) => {
                format!("{} = add {} {}", reg, a, b)
            },
            Self::Return(a) => {
                format!("return {}", a)
            },
        };
        return f.write_str(&str);
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Symbol>,
    pub body: Block,
}

impl Function {
    fn new(name: &str) -> Self {
        return Self {
            name: name.to_string(),
            arguments: Vec::new(),
            body: Block::new(),
        };
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub instructions: Vec<Instruction>,
    pub registers: i64,
    pub symbols: Vec<Symbol>,
}

impl Block {
    fn new() -> Self {
        return Self {
            instructions: Vec::new(),
            registers: 0,
            symbols: Vec::new(),
        };
    }

    fn add_register(&mut self) -> Register {
        let num = self.registers;
        self.registers += 1;
        return Register(num);
    }

    fn add_symbol(&mut self, name: &str, kind: SymbolKind) -> Symbol {
        let sym = Symbol {
            name: name.to_string(),
            kind: kind,
        };
        self.symbols.push(sym.clone());
        return sym;
    }

    fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    fn find_symbol(&self, name: &str) -> &Symbol {
        return self.symbols.iter().find(|s| s.name == name).unwrap();
    }

    fn find_register(&self, sym: &Symbol) -> Register {
        return match sym.kind {
            SymbolKind::FunctionArgument(reg) => reg,
            SymbolKind::Variable => {
                for instr in &self.instructions {
                    if let Instruction::Alloc(reg, maybe_sym) = instr {
                        if maybe_sym == sym {
                            return *reg;
                        }
                    }
                }
                panic!("Could not find symbol '{}'.", sym.name);
            },
            _ => panic!("Could not find symbol '{}'.", sym.name),
        };
    }
}

fn compile_expression(block: &mut Block, expr: &Expression) -> Argument {
    return match expr {
        Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            Argument::Integer(parsed)
        },
        Expression::Identifier(x) => {
            let sym = block.find_symbol(&x.value);
            let ident_reg = block.find_register(sym);
            Argument::Register(ident_reg)
        },
        Expression::BinaryExpr(bin_expr) => {
            let left_reg = compile_expression(block, &bin_expr.left);
            let right_reg = compile_expression(block, &bin_expr.right);
            let result_reg = block.add_register();
            block.add_instruction(Instruction::Add(result_reg, left_reg, right_reg));

            Argument::Register(result_reg)
        },
        _ => panic!(),
    };
}

fn compile_variable(block: &mut Block, var: &Variable) {
    let alloc_sym = block.add_symbol(&var.name.value, SymbolKind::Variable);
    let alloc_reg = block.add_register();
    block.instructions.push(Instruction::Alloc(alloc_reg, alloc_sym.clone()));
    let init_value = compile_expression(block, &var.initializer);
    block.instructions.push(Instruction::Store(alloc_reg, init_value))
}

fn compile_function(ast_fx: crate::ast::Function) -> Function {
    let mut fx = Function::new(&ast_fx.name.value);

    for arg in ast_fx.arguments {
        let arg_reg = fx.body.add_register();
        let arg_sym = fx.body.add_symbol(&arg.name.value, SymbolKind::FunctionArgument(arg_reg));
       
        fx.arguments.push(arg_sym.clone());
    }

    for stmt in &ast_fx.body.statements {
        compile_statement(&mut fx.body, stmt);
    }

    return fx;
}

fn compile_statement(block: &mut Block, stmt: &Statement) {
    match stmt {
        Statement::Variable(var) => {
            compile_variable(block, &var);
        },
        Statement::Expression(expr) => {
            compile_expression(block, &expr);
        },
        Statement::Return(expr) => {
            let ret_arg = compile_expression(block, expr);
            block.add_instruction(Instruction::Return(ret_arg));
        },
        _ => panic!(),
    };
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub main: Function,
    pub fns: Vec<Function>,
}

impl Bytecode {
    fn from_str(code: &str) -> Self {
        let ast = crate::ast::from_code(code).unwrap();
        let mut bc = Bytecode {
            main: Function::new("main"),
            fns: Vec::new(),
        };
    
        for stmt in ast.statements {
            match stmt {
                Statement::Function(ast_fx) => {
                    if ast_fx.name.value == bc.main.name {
                        let fx = compile_function(ast_fx);
                        bc.main = fx;
                    } else {
                        let fx = compile_function(ast_fx);
                        bc.fns.push(fx);
                    }
                },
                _ => compile_statement(&mut bc.main.body, &stmt)
            };
        }

        let main_has_return = bc.main.body.instructions.iter().any(|x| matches!(x, Instruction::Return(_)));

        if !main_has_return {
            bc.main.body.add_instruction(Instruction::Return(Argument::Void));
        }

        return bc;
    }
}

impl std::fmt::Display for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fmt_fn = |fx: &Function| {
            let mut out_s = String::with_capacity(128);
            let arg_s = fx.arguments.iter().map(|s| s.name.to_string()).collect::<Vec<String>>().join(", ");
            out_s.push_str(&format!("{}({}):\n", fx.name, arg_s));
            for instr in &fx.body.instructions {
                let s = format!("  {}\n", instr);
                out_s.push_str(&s);
            }
            return out_s;
        };

        for fx in &self.fns {
            let s = fmt_fn(fx);
            f.write_str(&s)?;
        }

        let s = fmt_fn(&self.main);
        f.write_str(&s)?;

        return std::fmt::Result::Ok(());
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
        
        let bc = Bytecode::from_str(code);
        let instructions = bc.main.body.instructions;

        assert_eq!(3, instructions.len());
        assert_eq!("r0 = alloc x", format!("{}", instructions[0]));
        assert_eq!("store r0 6", format!("{}", instructions[1]));
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = Bytecode::from_str(code);
        let instructions = bc.main.body.instructions;

        assert_eq!(5, instructions.len());
        assert_eq!("r0 = alloc x", format!("{}", instructions[0]));
        assert_eq!("store r0 6", format!("{}", instructions[1]));
        assert_eq!("r1 = alloc y", format!("{}", instructions[2]));
        assert_eq!("store r1 r0", format!("{}", instructions[3]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            1 + 2;
        "###;

        let bc = Bytecode::from_str(code);
        let instructions = bc.main.body.instructions;

        assert_eq!(2, instructions.len());
        assert_eq!("r0 = add 1 2", format!("{}", instructions[0]));
    }

    #[test]
    fn should_compile_function() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
        "###;

        let bc = Bytecode::from_str(code);
        let add = &bc.fns[0].body;

        assert_eq!(1, bc.fns.len());
        assert_eq!(2, add.instructions.len());
        assert_eq!("r2 = add r0 r1", format!("{}", add.instructions[0]));
        assert_eq!("return r2", format!("{}", add.instructions[1]));
    }
}
