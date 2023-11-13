use crate::ast::Expression;
use crate::ast::Statement;
use crate::ast::Error;
use crate::ast::Variable;
use crate::tokenizer::TokenKind;

#[derive(Debug, Clone, Copy)]
struct Register(i64);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("r{}", self.0);
        f.write_str(&s);
        return std::fmt::Result::Ok(());
    }
}

#[derive(Debug, Clone)]
enum Argument {
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
            Self::Void => f.write_str("void"),
        };
    }
}

#[derive(Debug, Clone)]
struct Symbol {
    name: String,
    register: Register,
}

#[derive(Debug, Clone)]
enum Instruction {
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
struct Label {
    name: String,
    instructions: Vec<Instruction>,
    registers: i64,
    symbols: Vec<Symbol>,
}

impl Label {
    fn new(name: &str) -> Self {
        return Self {
            name: name.to_string(),
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

    fn add_symbol(&mut self, name: &str) -> Symbol {
        let reg = self.add_register();
        let sym = Symbol {
            name: name.to_string(),
            register: reg,
        };
        self.symbols.push(sym.clone());
        return sym;
    }

    fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }
}

fn compile_expression(label: &mut Label, expr: &Expression) -> Argument {
    return match expr {
        Expression::Literal(x) => {
            let parsed: i64 = x.value.parse().unwrap();
            Argument::Integer(parsed)
        }
        Expression::Identifier(x) => {
            let ident_reg = label.symbols.iter()
                .find(|s| s.name == x.value)
                .map(|s| s.register)
                .unwrap();
            Argument::Register(ident_reg)
        },
        Expression::BinaryExpr(bin_expr) => {
            let left_reg = compile_expression(label, &bin_expr.left);
            let right_reg = compile_expression(label, &bin_expr.right);
            let result_reg = label.add_register();
            label.add_instruction(Instruction::Add(result_reg, left_reg, right_reg));

            Argument::Register(result_reg)
        },
        _ => panic!(),
    };
}

fn compile_variable(label: &mut Label, var: &Variable) {
    let alloc_sym = label.add_symbol(&var.name.value);
    label.instructions.push(Instruction::Alloc(alloc_sym.register, alloc_sym.clone()));
    let init_value = compile_expression(label, &var.initializer);
    label.instructions.push(Instruction::Store(alloc_sym.register, init_value))
}

fn compile(label: &mut Label, code: &str) {
    let ast = crate::ast::from_code(code).unwrap();

    for stmt in ast.statements {
        match stmt {
            Statement::Variable(var) => {
                compile_variable(label, &var);
            },
            Statement::Expression(expr) => {
                compile_expression(label, &expr);
            },
            _ => panic!(),
        };
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

        let mut label = Label::new("<unnamed>");
        compile(&mut label, code);

        assert_eq!(2, label.instructions.len());
        assert_eq!("r0 = alloc x", format!("{}", label.instructions[0]));
        assert_eq!("store r0 6", format!("{}", label.instructions[1]));
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let mut label = Label::new("<unnamed>");
        compile(&mut label, code);

        assert_eq!(4, label.instructions.len());
        assert_eq!("r0 = alloc x", format!("{}", label.instructions[0]));
        assert_eq!("store r0 6", format!("{}", label.instructions[1]));
        assert_eq!("r1 = alloc y", format!("{}", label.instructions[2]));
        assert_eq!("store r1 r0", format!("{}", label.instructions[3]));
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            1 + 2;
        "###;

        let mut label = Label::new("<unnamed>");
        compile(&mut label, code);

        assert_eq!(1, label.instructions.len());
        assert_eq!("r0 = add 1 2", format!("{}", label.instructions[0]));
    }
}
