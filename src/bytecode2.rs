use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::ast;
use crate::ast::AST;
use crate::ast::Statement;
use crate::ast::SymbolKind;
use crate::tokenizer::TokenKind;
use crate::typer::Type;
use crate::typer::Typer;
use crate::util::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Reg(i64);

const GPR0: Reg = Reg(0);
const GPR1: Reg = Reg(1);
const GPR2: Reg = Reg(2);
const GPR3: Reg = Reg(3);
const REGISTERS: [Reg; 4] = [
    GPR0,
    GPR1,
    GPR2,
    GPR3,
];

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = format!("GPR{}", self.0);
        return f.write_str(&name);
    }
}

#[derive(Debug, Clone)]
struct Mem {
    name: String,
    offset: i64,
    type_: Type,
}

impl std::fmt::Display for Mem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.name.fmt(f);
    }
}

#[derive(Debug, Clone)]
enum Instr {
    Alloc(Rc<Mem>),
    LoadMem(Reg, Rc<Mem>),
    LoadInt(Reg, i64),
    Store(Rc<Mem>, Reg),
    Add(Reg, Reg),
}

impl Instr {
    fn reg_reserved(&self) -> Option<Reg> {
        return match self {
            Self::LoadMem(r1, _) => Some(*r1),
            Self::LoadInt(r1, _) => Some(*r1),
            Self::Add(r1, _) => Some(*r1),
            _ => None,
        };
    }

    fn reg_released(&self) -> Option<Reg> {
        return match self {
            Self::Store(_, r1) => Some(*r1),

            // release the argument register but not the result.
            Self::Add(_, r2) => Some(*r2),
            _ => None,
        };
    }
}

impl std::fmt::Display for Instr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Alloc(x) => {
                format!("{:>8}  {}, {}", "alloc", x.name, x.type_)
            }
            Self::LoadMem(reg, mem) => {
                format!("{:>8}  {}, {}", "loadm", reg, mem)
            }
            Self::LoadInt(reg, v) => {
                format!("{:>8}  {}, {}", "loadi", reg, v)
            }
            Self::Store(mem, reg) => {
                format!("{:>8}  {}, {}", "store", mem, reg)
            }
            Self::Add(dest, arg) => {
                format!("{:>8}  {}, {}", "add", dest, arg)
            }
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
struct Stack {
    data: Vec<Rc<Mem>>,
    aliases: HashMap<String, Rc<Mem>>,
}

impl Stack {
    fn new() -> Self {
        return Self {
            data: Vec::new(),
            aliases: HashMap::new(),
        };
    }

    fn find(&self, name: &str) -> Rc<Mem> {
        if let Some(x) = self.aliases.get(name) {
            return Rc::clone(x);
        }
        return self.data.iter()
            .find(|x| x.name == name)
            .map(|x| Rc::clone(x))
            .expect(&format!("variable '{}' does not exist on the stack", name));
    }
    
    fn push(&mut self, type_: &Type) -> Rc<Mem> {
        let name = format!("%{}", self.data.len());
        let next_offset = self.data.last()
            .map(|x| {
                return x.offset + x.type_.size();
            })
            .unwrap_or(0);
        let var = Rc::new(Mem {
            name: name,
            type_: type_.clone(),
            offset: next_offset,
        });
        self.data.push(Rc::clone(&var));
        return var;
    }
    
    fn add_alias(&mut self, var: &Rc<Mem>, as_name: &str) {
        self.aliases.insert(as_name.to_string(), Rc::clone(var));
    }
}

struct BC {
    instructions: Vec<Instr>,
    live_regs: HashSet<Reg>,
    typer: Rc<Typer>,
}

impl BC {
    fn find_available_reg(&self) -> Reg {
        for reg in &REGISTERS {
            if !self.live_regs.contains(reg) {
                return *reg;
            }
        }
        panic!("no available registers!");
    }

    fn emit(&mut self, instr: Instr) {
        self.instructions.push(instr.clone());

        if let Some(x) = &instr.reg_reserved() {
            self.live_regs.insert(*x);
        }
        
        if let Some(x) = &instr.reg_released() {
            self.live_regs.remove(x);
        }
    }

    fn compile_stmt(&mut self, stmt: &ast::Statement, stack: &mut Stack) {
        match stmt {
            Statement::Variable(x) => {
                let sym = self.typer
                    .try_find_symbol(&x.name_token.value, SymbolKind::Local, x.parent)
                    .unwrap();
                let mem = stack.push(&sym.type_);
                stack.add_alias(&mem, &x.name_token.value);
                self.emit(Instr::Alloc(Rc::clone(&mem)));
                let reg = self.compile_expr(&x.initializer, stack);
                self.emit(Instr::Store(mem, reg));
            }
            _ => {
                todo!("{:?}", stmt);
            }
        }
    }

    fn compile_expr(&mut self, expr: &ast::Expression, stack: &mut Stack) -> Reg {
        return match expr {
            ast::Expression::Identifier(x) => {
                let mem = stack.find(&x.name);
                let reg = self.find_available_reg();
                self.emit(Instr::LoadMem(reg, mem));
                reg   
            }
            ast::Expression::IntegerLiteral(x) => {
                let reg = self.find_available_reg();
                self.emit(Instr::LoadInt(reg, x.value));
                reg
            }
            ast::Expression::BinaryExpr(x) => {
                match x.operator.kind {
                    TokenKind::Plus => {
                        let r1 = self.compile_expr(&x.left, stack);
                        let r2 = self.compile_expr(&x.right, stack);
                        self.emit(Instr::Add(r1, r2));
                        r1
                    }
                    _ => todo!("{}", x.operator.kind),
                }
            }
            _ => {
                todo!("{:?}", expr);
            }
        }
    }
}

impl std::fmt::Display for BC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::with_capacity(512);
        for instr in &self.instructions {
            s.push_str(&instr.to_string());
            s.push_str("\n");
        }
        return f.write_str(&s);
    }
}

fn from_code(code: &str) -> Result<BC, Error> {
    let typer = Typer::from_code(code)?;
    let ast = Rc::clone(&typer.ast);
    let mut bc = BC {
        instructions: Vec::new(),
        live_regs: HashSet::new(),
        typer: Rc::new(typer),
    };
    let root = ast.root.as_block();
    let mut stack = Stack::new();
    for stmt in &root.statements {
        bc.compile_stmt(stmt, &mut stack);
    }

    return Ok(bc);
}

#[cfg(test)]
mod tests {
    use crate::bytecode2::BC;
    use crate::bytecode2::from_code;

    #[test]
    fn thing() {
        let code = r###"
        var x = 1 + 1;
        var y = x;
        "###;
        let bc = do_test(code);
        println!("{}", bc);
        assert!(false);
    }

    fn do_test(code: &str) -> BC {
        let bc = from_code(code)
            .unwrap();

        return bc;
    }
}