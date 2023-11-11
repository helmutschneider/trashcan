use crate::ast::Expression;
use crate::ast::Statement;
use crate::ast::Error;

#[derive(Debug, Clone)]
struct Symbol {
    id: i64,
    name: String,
}

#[derive(Debug)]
enum Value {
    Integer(i64),
    Identifier(Symbol),
}

#[derive(Debug)]
enum Instruction {
    Store(Symbol, Value)
}

#[derive(Debug)]
struct Block {
    name: String,
    instructions: Vec<Instruction>,
    symbols: Vec<Symbol>,
}

const UNNAMED_BLOCK: &'static str = "<unnamed>";

fn from_code(code: &str) -> Block {
    let ast = crate::ast::from_code(code).unwrap();
    let mut block = Block {
        name: UNNAMED_BLOCK.to_string(),
        instructions: Vec::new(),
        symbols: Vec::new(),
    };

    for stmt in ast.statements {
        if let Statement::Variable(var) = stmt {
            let sym = Symbol {
                id: block.symbols.len() as i64,
                name: var.name.value,
            };

            block.symbols.push(sym.clone());

            let instr = match var.initializer {
                Expression::Literal(lit) => {
                    let value: i64 = lit.value.parse().unwrap();
                    Instruction::Store(sym, Value::Integer(value))
                },
                Expression::Identifier(ident) => {
                    let ident_sym = block.symbols.iter().find(|x| x.name == ident.value).unwrap();
                    Instruction::Store(sym, Value::Identifier(ident_sym.clone()))
                },
                _ => panic!(),
            };

            block.instructions.push(instr);
        } else {
            panic!();
        }
    }

    return block;
}

#[cfg(test)]
mod tests {
    use crate::ast::{Ast};
    use crate::bytecode::*;

    #[test]
    fn should_compile_assignment() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;
        let block = from_code(code);

        println!("{block:?}");

        assert!(false);
    }
}
