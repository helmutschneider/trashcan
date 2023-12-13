use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write;

use crate::ast;
use crate::ast::ElementAccess;
use crate::ast::Expression;
use crate::ast::Identifier;
use crate::ast::Statement;
use crate::ast::StructLiteralMember;
use crate::ast::SymbolKind;
use crate::typer::Type;
use crate::typer::Typer;
use crate::util::Offset;
use crate::{tokenizer::TokenKind, typer, util::Error};
use std::rc::Rc;

pub const ENTRYPOINT_NAME: &'static str = "__trashcan__main";

#[derive(Debug, Clone)]
pub struct Memory {
    pub name: String,
    pub type_: Type,
    pub offset: Offset,
}

impl std::fmt::Display for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("{}", self.name));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Register(&'static str);

pub const REG_R0: Register = Register("R0");
pub const REG_R1: Register = Register("R1");
pub const REG_R2: Register = Register("R2");
pub const REG_R3: Register = Register("R3");
pub const REG_R4: Register = Register("R4");
pub const REG_R5: Register = Register("R5");
pub const REG_R6: Register = Register("R6");
pub const REG_RET: Register = Register("RET");

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(self.0);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Address(pub Register, pub Offset);

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("[{}{}]", self.0, self.1))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstId(i64);

impl std::fmt::Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!(".LC{}", self.0));
    }
}

#[derive(Debug, Clone)]
pub struct Const {
    pub id: ConstId,

    // TODO: at some point we want other values than strings in constants.
    pub value: String,
}

type CreatedByExpr = bool;

#[derive(Debug, Clone)]
pub enum ExprOutput {
    Reg(Register, Type),
    Mem(Rc<Memory>, CreatedByExpr),
}

impl ExprOutput {
    pub fn get_type(&self) -> Type {
        return match self {
            Self::Reg(_, t) => t.clone(),
            Self::Mem(var, _) => var.type_.clone(),
        };
    }
}

impl std::fmt::Display for ExprOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Reg(r1, _) => r1.fmt(f),
            Self::Mem(var, _) => var.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Rc<Memory>>),
    Local(Rc<Memory>),
    Label(String),

    /** indirect memory store of immediate value. \[r1\] <- value */
    StoreInt(Address, i64),

    /** indirect memory store of register value. \[r1\] <- r2 */
    StoreReg(Address, Register),

    /** stack load r1 <- variable */
    LoadMem(Register, Rc<Memory>),

    /** immediate load r1 <- value */
    LoadInt(Register, i64),

    /** indirect load r1 <- \[r2\], where r2 should contain an address */
    LoadAddr(Register, Address),

    /** plain copy r1 <- r2 */
    LoadReg(Register, Register),
    
    AddrOf(Register, Rc<Memory>),
    AddrOfConst(Register, ConstId),
    Return,
    Add(Register, Register),
    Sub(Register, Register),
    Mul(Register, Register),
    Div(Register, Register),
    Call(String, Vec<Rc<Memory>>),
    IsEqual(Register, Register),
    Jump(String),
    JumpZero(String, Register),
    Const(Const),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(name, args) => {
                let args_s = args
                    .iter()
                    .map(|v| format!("{}: {}", v.name, v.type_))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{:>12}  {}({})", "function", name, args_s)
            }
            Self::Local(var) => {
                format!("{:>12}  {}, {}", "local", var.name, var.type_)
            }
            Self::Label(name) => {
                format!("{:>12}  {}", "label", name)
            }
            Self::StoreInt(dest_var, source) => {
                format!("{:>12}  {}, {}", "store", dest_var, source)
            }
            Self::StoreReg(r1, r2) => {
                format!("{:>12}  {}, {}", "store", r1, r2)
            }
            Self::LoadMem(reg, mem) => {
                format!("{:>12}  {}, {}", "load", reg, mem)
            }
            Self::LoadInt(reg, x) => {
                format!("{:>12}  {}, {}", "load", reg, x)
            }
            Self::LoadAddr(r1, r2) => {
                format!("{:>12}  {}, {}", "load", r1, r2)
            }
            Self::LoadReg(r1, r2) => {
                format!("{:>12}  {}, {}", "load", r1, r2)
            }
            Self::AddrOf(dest_var, source) => {
                format!("{:>12}  {}, {}", "lea", dest_var, source)
            }
            Self::AddrOfConst(dest_var, source) => {
                format!("{:>12}  {}, {}", "lea", dest_var, source)
            }
            Self::Return => {
                format!("{:>12}", "ret")
            }
            Self::Add(dest_var, x) => {
                format!("{:>12}  {}, {}", "add", dest_var, x)
            }
            Self::Sub(dest_var, x) => {
                format!("{:>12}  {}, {}", "sub", dest_var, x)
            }
            Self::Mul(dest_var, x) => {
                format!("{:>12}  {}, {}", "mul", dest_var, x)
            }
            Self::Div(dest_var, x) => {
                format!("{:>12}  {}, {}", "div", dest_var, x)
            }
            Self::Call(name, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{:>12}  {}({})", "call", name, arg_s)
            }
            Self::IsEqual(r1, r2) => {
                format!("{:>12}  {}, {}", "eq", r1, r2)
            }
            Self::Jump(to_label) => {
                format!("{:>12}  {}", "jump", to_label)
            }
            Self::JumpZero(to_label, reg) => {
                format!("{:>12}  {}, {}", "jumpz", to_label, reg)
            }
            Self::Const(cons) => {
                let escaped = cons.value.replace("\n", "\\n");
                format!("{:>12}  {}, \"{}\"", "const", cons.id, escaped)
            }
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
struct Stack {
    data: Vec<Rc<Memory>>,
    aliases: HashMap<String, Rc<Memory>>,
}

impl Stack {
    fn new() -> Self {
        return Self {
            data: Vec::new(),
            aliases: HashMap::new(),
        };
    }

    fn find(&self, name: &str) -> Rc<Memory> {
        if let Some(x) = self.aliases.get(name) {
            return Rc::clone(x);
        }
        return self
            .data
            .iter()
            .find(|x| x.name == name)
            .map(|x| Rc::clone(x))
            .expect(&format!("variable '{}' does not exist on the stack", name));
    }

    fn push(&mut self, type_: &Type) -> Rc<Memory> {
        let name = format!("%{}", self.data.len());
        let next_offset = self
            .data
            .last()
            .map(|x| {
                return x.offset.add(x.type_.size());
            })
            .unwrap_or(Offset::ZERO);
        let mem = Rc::new(Memory {
            name: name,
            type_: type_.clone(),
            offset: next_offset,
        });
        self.data.push(Rc::clone(&mem));
        return mem;
    }

    fn add_alias(&mut self, var: &Rc<Memory>, as_name: &str) {
        self.aliases.insert(as_name.to_string(), Rc::clone(var));
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: i64,
    pub registers: Vec<Register>,
    pub typer: Rc<typer::Typer>,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = Rc::new(typer::Typer::from_code(code)?);
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
            registers: [REG_R0, REG_R1, REG_R2, REG_R3, REG_R4, REG_R5, REG_R6].to_vec(),
            typer: Rc::clone(&typer),
        };

        let mut main_statements: Vec<Rc<Statement>> = Vec::new();

        for stmt in &typer.ast.root.as_block().statements {
            let goes_in_main = match stmt.as_ref() {
                Statement::Function(_) => false,
                Statement::Variable(_) => true,
                Statement::Expression(_) => true,
                Statement::Return(_) => false,
                Statement::Block(_) => true,
                Statement::If(_) => true,
                Statement::While(_) => true,
                Statement::Type(_) => false,
            };

            if goes_in_main {
                main_statements.push(Rc::clone(stmt));
            } else {
                let mut stack = Stack::new();
                bc.compile_statement(stmt, &mut stack);
            }
        }

        let mut main_stack = Stack::new();
        bc.emit(Instruction::Function(
            ENTRYPOINT_NAME.to_string(),
            Vec::new(),
        ));
        for stmt in &main_statements {
            bc.compile_statement(stmt, &mut main_stack);
        }

        let main_has_return_statement = main_statements
            .last()
            .map(|x| matches!(x.as_ref(), Statement::Return(_)))
            .unwrap_or(false);

        if !main_has_return_statement {
            bc.emit(Instruction::LoadInt(REG_RET, 0));
            bc.emit(Instruction::Return);
        }

        return Ok(bc);
    }

    fn emit(&mut self, instr: Instruction) {
        self.instructions.push(instr.clone());
    }

    fn take_register(&mut self) -> Register {
        return self.registers.remove(0);
    }

    fn release_register(&mut self, reg: Register) {
        if reg == REG_RET {
            return;
        }
        if self.registers.contains(&reg) {
            panic!("register '{}' is not locked.", reg);
        }
        self.registers.push(reg);
        self.registers.sort();
    }

    /** TODO: review */
    fn load_expr_immediate(&mut self, expr: &ExprOutput) -> Register {
        let reg = match expr {
            ExprOutput::Reg(r1, t) => {
                if t.is_pointer() {
                    self.emit(Instruction::LoadAddr(*r1, Address(*r1, Offset::ZERO)));
                }
                *r1
            },
            ExprOutput::Mem(var, _) => {
                let r1 = self.take_register();
                self.emit(Instruction::LoadMem(r1, Rc::clone(var)));
                r1
            },
        };

        return reg;
    }

    fn release_expr(&mut self, expr: &ExprOutput) {
        match expr {
            ExprOutput::Reg(r1, _) => {
                self.release_register(*r1);
            }
            _ => {}
        };
    }

    fn add_label(&mut self) -> String {
        let label = format!(".LB{}", self.labels);
        self.labels += 1;
        return label;
    }

    fn emit_constant(&mut self, value: &str) -> ConstId {
        let num_constants = self
            .instructions
            .iter()
            .filter(|x| matches!(x, Instruction::Const(_)))
            .count();
        let id = ConstId(num_constants as i64);
        let cons = Const {
            id: id,
            value: value.to_string(),
        };

        let to_insert = Instruction::Const(cons);

        // find the start of the function or the most recently allocated variable.
        for k in 0..self.instructions.len() {
            let index = self.instructions.len() - k - 1;
            let instr = &self.instructions[index];
            let do_insert = match instr {
                Instruction::Local(_) => true,
                Instruction::Function(_, _) => true,
                Instruction::Const(_) => true,
                _ => false,
            };

            if do_insert {
                self.instructions.insert(index + 1, to_insert);
                return id;
            }
        }

        panic!("could not find start of function");
    }

    fn compile_expression(&mut self, expr: &ast::Expression, stack: &mut Stack) -> ExprOutput {
        return match expr {
            ast::Expression::Void => {
                let r1 = self.take_register();
                self.emit(Instruction::LoadInt(r1, 0xDEAD_BEEF));
                ExprOutput::Reg(r1, Type::Void)
            }
            ast::Expression::IntegerLiteral(x) => {
                let r1 = self.take_register();
                self.emit(Instruction::LoadInt(r1, x.value));
                ExprOutput::Reg(r1, Type::Int)
            }
            ast::Expression::StringLiteral(s) => {
                let cons = self.emit_constant(&s.value);
                let var_ref = self.emit_variable(&Type::String, stack);
                let r1 = self.take_register();
                let r2 = self.take_register();

                self.emit(Instruction::AddrOf(r1, Rc::clone(&var_ref)));
                self.emit(Instruction::StoreInt(Address(r1, Offset::ZERO), s.value.len() as i64));
                
                self.emit(Instruction::AddrOfConst(r2, cons));
                self.emit(Instruction::StoreReg(Address(r1, Offset(8)), r2));

                self.release_register(r2);
                self.release_register(r1);
            
                ExprOutput::Mem(var_ref, true)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                if bin_expr.operator.kind == TokenKind::Equals {
                    let lhs = match bin_expr.left.as_ref() {
                        // the star '*' operator dereferences an expression into its underlying
                        // value. that's not what we want here: instead, we want the star operator
                        // to store the right hand side to the memory location that the left
                        // hand side is pointing at.
                        //   -johan, 2023-12-12
                        Expression::UnaryPrefix(x) if x.operator.kind == TokenKind::Star => {
                            self.compile_expression(&x.expr, stack)
                        }
                        _ => self.compile_expression(&bin_expr.left, stack),
                    };
                    let reg = match &lhs {
                        ExprOutput::Reg(r1, t) => {
                            assert!(t.is_pointer());
                            *r1
                        }

                        // TODO: review this. i feel like we shouldn't have to massage
                        //   the memory this much...
                        ExprOutput::Mem(mem, _) => {
                            let r1 = self.take_register();
                            if mem.type_.is_pointer() {
                                self.emit(Instruction::LoadMem(r1, Rc::clone(mem)));
                            } else {
                                self.emit(Instruction::AddrOf(r1, Rc::clone(mem)));
                            }                            
                            r1
                        },
                    };

                    let rhs = self.compile_expression(&bin_expr.right, stack);
                    let type_ = rhs.get_type();

                    self.emit_copy(Address(reg, Offset::ZERO), &type_, &rhs);
                    self.release_expr(&rhs);

                    return ExprOutput::Reg(reg, type_)
                } else {    
                    let lhs = self.compile_expression(&bin_expr.left, stack);
                    let rhs = self.compile_expression(&bin_expr.right, stack);
                    let r1 = self.load_expr_immediate(&lhs);
                    let r2 = self.load_expr_immediate(&rhs);
    
                    let res = match bin_expr.operator.kind {
                        TokenKind::Plus => {
                            self.emit(Instruction::Add(r1, r2));
                            ExprOutput::Reg(r1, Type::Int)
                        }
                        TokenKind::Minus => {
                            self.emit(Instruction::Sub(r1, r2));
                            ExprOutput::Reg(r1, Type::Int)
                        }
                        TokenKind::Star => {
                            self.emit(Instruction::Mul(r1, r2));
                            ExprOutput::Reg(r1, Type::Int)
                        }
                        TokenKind::Slash => {
                            self.emit(Instruction::Div(r1, r2));
                            ExprOutput::Reg(r1, Type::Int)
                        }
                        TokenKind::DoubleEquals => {
                            self.emit(Instruction::IsEqual(r1, r2));
                            ExprOutput::Reg(r1, Type::Int)
                        }
                        TokenKind::NotEquals => {
                            self.emit(Instruction::IsEqual(r1, r2));
                            self.emit(Instruction::LoadInt(r2, 0));
                            self.emit(Instruction::IsEqual(r1, r2));
                            ExprOutput::Reg(r1, Type::Bool)
                        }
                        _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
                    };
    
                    self.release_register(r2);
    
                    return res;
                }
            }
            ast::Expression::Identifier(ident) => {
                let var = stack.find(&ident.name);
                ExprOutput::Mem(var, false)
            }
            ast::Expression::FunctionCall(call) => {
                let fx_sym = self
                    .typer
                    .try_find_symbol(&call.name_token.value, SymbolKind::Function, call.parent)
                    .unwrap();

                let (arg_types, ret_type) = match fx_sym.type_ {
                    Type::Function(x, y) => (x, y),
                    _ => panic!(),
                };

                let mut args: Vec<Rc<Memory>> = Vec::new();

                for k in 0..arg_types.len() {
                    let given_arg = self.compile_expression(&call.arguments[k], stack);
                    let var = match &given_arg {
                        ExprOutput::Mem(x, _) => Rc::clone(&x),
                        _ => {
                            let var = self.emit_variable(&arg_types[k], stack);
                            self.emit_copy_to_variable(&var, &given_arg);
                            var
                        }
                    };
                    args.push(var);
                    self.release_expr(&given_arg);
                }

                self.emit(Instruction::Call(call.name_token.value.clone(), args));
                ExprOutput::Reg(REG_RET, *ret_type)
            }
            ast::Expression::StructLiteral(s) => {
                let type_ = self
                    .typer
                    .try_find_symbol(&s.name_token.value, SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();

                // make sure to initialize the struct as declared by the type
                // and not in the order of initialized arguments.
                let members_by_name: HashMap<String, StructLiteralMember> = s
                    .members
                    .iter()
                    .map(|m| (m.field_name_token.value.clone(), m.clone()))
                    .collect();

                let dest_var = self.emit_variable(&type_, stack);

                let r1 = self.take_register();
                self.emit(Instruction::AddrOf(r1, Rc::clone(&dest_var)));

                let members = match &type_ {
                    Type::Struct(_, members) => members,
                    _ => panic!("type '{}' is not a struct", type_),
                };

                for m in members {
                    let value = &members_by_name.get(&m.name).unwrap().value;
                    let arg = self.compile_expression(&value, stack);
                    self.emit_copy(Address(r1, m.offset), &m.type_, &arg);
                    self.release_expr(&arg);
                }

                self.release_register(r1);

                ExprOutput::Mem(dest_var, true)
            }
            ast::Expression::MemberAccess(access) => {
                let left = self.compile_expression(&access.left, stack);
                let left_type = left.get_type();
                let member = left_type
                    .find_member(&access.right.name)
                    .unwrap();

                let reg = match left {
                    ExprOutput::Reg(x, t) => {
                        assert!(t.is_pointer());
                        x
                    },
                    ExprOutput::Mem(x, _) => {
                        let r1 = self.take_register();
                        match x.type_ {
                            Type::Pointer(_) => {
                                self.emit(Instruction::LoadMem(r1, Rc::clone(&x)));
                            }
                            _ => {
                                self.emit(Instruction::AddrOf(r1, Rc::clone(&x)));
                            }
                        };
                        r1
                    }
                };

                // let's signal to the bytecode that all member accesses result in
                // pointers. this is useful because it allows the receiver to
                // choose whether if they want to copy the value out of the pointer
                // or use it as-is.
                let type_ = Type::Pointer(Box::new(member.type_));

                if member.offset != Offset::ZERO {
                    let r2 = self.take_register();
                    self.emit(Instruction::LoadInt(r2, member.offset.0));
                    self.emit(Instruction::Add(reg, r2));
                    self.release_register(r2);
                }

                ExprOutput::Reg(reg, type_)
            }
            ast::Expression::BooleanLiteral(b) => {
                let r1 = self.take_register();
                self.emit(Instruction::LoadInt(r1, b.value.into()));
                ExprOutput::Reg(r1, Type::Bool)
            }
            ast::Expression::UnaryPrefix(unary_expr) => {
                let type_ = self
                    .typer
                    .try_infer_expression_type(&unary_expr.expr)
                    .unwrap();
                let arg = self.compile_expression(&unary_expr.expr, stack);
                let res = match unary_expr.operator.kind {
                    TokenKind::Ampersand => {
                        let ptr_type = Type::Pointer(Box::new(type_));
                        let r1 = match &arg {
                            ExprOutput::Reg(x, t) => {
                                assert!(t.is_pointer());
                                *x
                            }
                            ExprOutput::Mem(x, _) => {
                                let r1 = self.take_register();
                                self.emit(Instruction::AddrOf(r1, Rc::clone(&x)));
                                r1
                            }
                        };
                        
                        ExprOutput::Reg(r1, ptr_type)
                    }
                    TokenKind::Star => {
                        let type_ = match type_ {
                            Type::Pointer(x) => *x,
                            _ => panic!("not a pointer bro")
                        };
                        let r1 = match &arg {
                            ExprOutput::Reg(x, _) => *x,
                            ExprOutput::Mem(x, _) => {
                                let r1 = self.take_register();
                                self.emit(Instruction::LoadMem(r1, Rc::clone(&x)));
                                r1
                            }
                        };
                        self.emit(Instruction::LoadAddr(r1, Address(r1, Offset::ZERO)));
                        ExprOutput::Reg(r1, type_)
                    }
                    TokenKind::Minus => {
                        let r1 = self.take_register();
                        self.emit(Instruction::LoadInt(r1, 0));
                        let r2 = self.load_expr_immediate(&arg);
                        self.emit(Instruction::Sub(r1, r2));
                        self.release_register(r2);

                        ExprOutput::Reg(r1, Type::Int)
                    }
                    _ => panic!(),
                };
                res
            }
            ast::Expression::ArrayLiteral(array_lit) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let elem_type = match &type_ {
                    Type::Array(x, _) => x,
                    _ => panic!("not an array bro")
                };

                let dest_var = self.emit_variable(&type_, stack);
                let r1 = self.take_register();
                self.emit(Instruction::AddrOf(r1, Rc::clone(&dest_var)));

                let length = array_lit.elements.len() as i64;
                self.emit(Instruction::StoreInt(Address(r1, Offset::ZERO), length));
                let mut offset = Offset(Type::Int.size());

                for k in 0..array_lit.elements.len() {
                    let elem_expr = &array_lit.elements[k];
                    let elem_arg = self.compile_expression(elem_expr, stack);

                    let dest = Address(r1, offset);
                    self.emit_copy(dest, elem_type, &elem_arg);
                    self.release_expr(&elem_arg);

                    offset = offset.add(elem_type.size());
                }

                self.release_register(r1);

                ExprOutput::Mem(dest_var, true)
            }
            ast::Expression::ElementAccess(elem_access) => {
                let left = self.compile_expression(&elem_access.left, stack);
                let left_type = left.get_type();
                let elem_type = match &left_type {
                    Type::Array(x, _) => x,
                    Type::Pointer(inner) => {
                        if let Type::Array(x, _) = inner.as_ref() {
                            x
                        } else {
                            panic!("not an array bro")
                        }
                    }
                    _ => panic!("not an array bro")
                };

                let r1 = match left {
                    ExprOutput::Reg(x, t) => {
                        assert!(t.is_pointer());
                        x
                    },
                    ExprOutput::Mem(x, _) => {
                        let r1 = self.take_register();
                        match x.type_ {
                            Type::Pointer(_) => {
                                self.emit(Instruction::LoadMem(r1, Rc::clone(&x)));
                            }
                            _ => {
                                self.emit(Instruction::AddrOf(r1, Rc::clone(&x)));
                            }
                        };
                        r1
                    }
                };

                let right = self.compile_expression(&elem_access.right, stack);
                let r2 = match right {
                    ExprOutput::Reg(x, _) => x,
                    ExprOutput::Mem(mem, _) => {
                        let r1 = self.take_register();
                        self.emit(Instruction::LoadMem(r1, Rc::clone(&mem)));
                        r1
                    }
                };
                let r3 = self.take_register();

                // calculate and add the element offset.
                self.emit(Instruction::LoadInt(r3, elem_type.size()));
                self.emit(Instruction::Mul(r2, r3));
                self.release_register(r3);

                self.emit(Instruction::Add(r1, r2));

                // the first 8 bytes of an array is its length.
                self.emit(Instruction::LoadInt(r2, Type::Int.size()));
                self.emit(Instruction::Add(r1, r2));
                
                self.release_register(r2);

                // signal to the bytecode that all element accesses result
                // in a pointer. the caller can copy the value if they want.
                let type_ = Type::Pointer(Box::new(*elem_type.clone()));

                ExprOutput::Reg(r1, type_)
            }
        };
    }

    fn emit_variable(&mut self, type_: &Type, stack: &mut Stack) -> Rc<Memory> {
        let var = stack.push(type_);
        let to_insert = Instruction::Local(Rc::clone(&var));

        // find the start of the function or the most recently allocated variable.
        for k in 0..self.instructions.len() {
            let index = self.instructions.len() - k - 1;
            let instr = &self.instructions[index];
            let do_insert = match instr {
                Instruction::Local(_) => true,
                Instruction::Function(_, _) => true,
                _ => false,
            };

            if do_insert {
                self.instructions.insert(index + 1, to_insert);
                return var;
            }
        }

        panic!("could not find start of function");
    }

    fn compile_function(&mut self, fx: &ast::Function) {
        let mut stack = Stack::new();
        let arg_vars: Vec<Rc<Memory>> = fx
            .arguments
            .iter()
            .map(|fx_arg| {
                let fx_arg_sym = self
                    .typer
                    .try_find_symbol(&fx_arg.name_token.value, SymbolKind::Local, fx.body.id())
                    .unwrap();

                let fx_arg_type = fx_arg_sym.type_;
                let fx_arg_var = stack.push(&fx_arg_type);
                stack.add_alias(&fx_arg_var, &fx_arg_sym.name);
                fx_arg_var
            })
            .collect();

        self.instructions
            .push(Instruction::Function(fx.name_token.value.clone(), arg_vars));

        self.compile_block(fx.body.as_block(), &mut stack);

        // add an implicit return statement if the function doesn't have one.
        if !matches!(self.instructions.last(), Some(Instruction::Return)) {
            self.emit(Instruction::LoadInt(REG_RET, 0));
            self.emit(Instruction::Return);
        }
    }

    fn compile_block(&mut self, block: &ast::Block, stack: &mut Stack) {
        for stmt in &block.statements {
            self.compile_statement(stmt, stack);
        }
    }

    fn compile_statement(&mut self, stmt: &ast::Statement, stack: &mut Stack) {
        match stmt {
            ast::Statement::Function(fx) => {
                self.compile_function(fx);
            }
            ast::Statement::Block(block) => {
                self.compile_block(block, stack);
            }
            ast::Statement::Variable(var) => {
                let var_sym = self
                    .typer
                    .try_find_symbol(&var.name_token.value, SymbolKind::Local, var.parent)
                    .unwrap();

                let init_arg = self.compile_expression(&var.initializer, stack);
                let var = match &init_arg {
                    ExprOutput::Mem(x, created_by_expr) => {
                        let var = if *created_by_expr { Rc::clone(x) } else {
                            let x = self.emit_variable(&var_sym.type_, stack);
                            self.emit_copy_to_variable(&x, &init_arg);
                            x
                        };
                        var
                    },
                    _ => {
                        let var = self.emit_variable(&var_sym.type_, stack);
                        self.emit_copy_to_variable(&var, &init_arg);
                        var
                    }
                };
                self.release_expr(&init_arg);

                stack.add_alias(&var, &var_sym.name);
            }
            ast::Statement::Return(ret) => {
                let ret_arg = self.compile_expression(&ret.expr, stack);
                let r1 = self.load_expr_immediate(&ret_arg);
                self.emit(Instruction::LoadReg(REG_RET, r1));
                self.release_register(r1);
                self.emit(Instruction::Return);
            }
            ast::Statement::If(if_stmt) => {
                let label_after_last_block = self.add_label();

                self.compile_if(&if_stmt, &label_after_last_block, stack);
            }
            ast::Statement::While(while_) => {
                let label_before_condition = self.add_label();
                let label_after_block = self.add_label();

                self.emit(Instruction::Label(label_before_condition.clone()));
                let cmp_arg = self.compile_expression(&while_.condition, stack);
                let r1 = self.load_expr_immediate(&cmp_arg);
                self.emit(Instruction::JumpZero(label_after_block.clone(), r1));
                self.release_register(r1);

                self.compile_block(while_.block.as_block(), stack);
                self.emit(Instruction::Jump(label_before_condition));
                self.emit(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                let expr = self.compile_expression(&expr.expr, stack);
                self.release_expr(&expr);
            }
            ast::Statement::Type(_) => {
                // do nothing. type declarations aren't represented by specific
                // instructions in the bytecode.
            }
        }
    }

    fn compile_if(&mut self, if_stmt: &ast::If, label_after_last_block: &str, stack: &mut Stack) {
        let condition = self.compile_expression(&if_stmt.condition, stack);
        let label_after_block = self.add_label();

        let r1 = self.load_expr_immediate(&condition);
        self.emit(Instruction::JumpZero(label_after_block.clone(), r1));
        self.release_register(r1);

        self.compile_block(if_stmt.block.as_block(), stack);
        self.emit(Instruction::Jump(label_after_last_block.to_string()));
        self.emit(Instruction::Label(label_after_block));

        let else_ = if_stmt.else_.as_ref();
        let mut is_last_block = false;

        if let Some(next) = else_ {
            if let Statement::If(next_if) = next.as_ref() {
                self.compile_if(next_if, label_after_last_block, stack);
            }

            if let Statement::Block(else_) = next.as_ref() {
                self.compile_block(else_, stack);
                is_last_block = true;
            }
        } else {
            is_last_block = true;
        }

        if is_last_block {
            self.emit(Instruction::Label(label_after_last_block.to_string()));
        }
    }

    fn emit_copy(&mut self, dest: Address, dest_type: &Type, source: &ExprOutput) {
        let (source_reg, must_release) = match source {
            ExprOutput::Reg(reg, type_) => {
                let needs_deref_and_copy = match type_ {
                    Type::Pointer(inner) => inner.as_ref() == dest_type,
                    _ => false,
                };

                if needs_deref_and_copy {
                    (*reg, false)
                } else {
                    self.emit(Instruction::StoreReg(dest, *reg));
                    return;
                }
            }
            ExprOutput::Mem(var, _) => {
                let r1 = self.take_register();
                self.emit(Instruction::AddrOf(r1, Rc::clone(var)));
                (r1, true)
            },
        };

        let size = dest_type.size();
        let size_t = Type::Pointer(Box::new(Type::Void)).size();
        let r2 = self.take_register();
        let mut offset = Offset::ZERO;

        while offset.0 < size {
            let source = Address(source_reg, offset);
            let dest = Address(dest.0, dest.1.add(offset));
            self.emit(Instruction::LoadAddr(r2, source));
            self.emit(Instruction::StoreReg(dest, r2));
            offset = offset.add(size_t);
        }

        self.release_register(r2);

        if must_release {
            self.release_register(source_reg);
        }
    }

    fn emit_copy_to_variable(&mut self, dest: &Rc<Memory>, source: &ExprOutput) {
        let r1 = self.take_register();
        self.emit(Instruction::AddrOf(r1, Rc::clone(dest)));
        self.emit_copy(Address(r1, Offset::ZERO), &dest.type_, source);
        self.release_register(r1);
    }
}

impl std::fmt::Display for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for k in 0..self.instructions.len() {
            let instr = &self.instructions[k];

            if matches!(instr, Instruction::Function(_, _)) && k != 0 {
                f.write_str("\n")?;
            }

            instr.fmt(f)?;
            f.write_str("\n")?;
        }

        for instr in &self.instructions {}
        return std::fmt::Result::Ok(());
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode::*;

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function  __trashcan__main()
             local %0, int
              load  R0, 1
              load  R1, 2
               add  R0, R1
               lea  R1, %0
             store  [R1+0], R0
              load RET, 0
               ret
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_if_with_jump() {
        let code = r###"
        var x: bool = 1 == 2;
        if x {
            var y: int = 42;
        } else {
            var z: int = 3;
        }
    "###;
        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function __trashcan__main()
            local %0, bool
            local %1, int
            local %2, int
             load R0, 1
             load R1, 2
               eq R0, R1            
              lea R1, %0
            store [R1+0], R0
             load R0, %0
            jumpz .LB1, R0
             load R0, 42
              lea R1, %1
            store [R1+0], R0
             jump .LB0
            label .LB1
             load R0, 3
              lea R1, %2
            store [R1+0], R0
            label .LB0
             load RET, 0
            ret
        "###;

        println!("{bc}");

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_layout_simple_struct() {
        let code = r###"
        type person = {
            id: int,
            age: int,
        };
        var x = person {
            id: 6,
            age: 5,
        };
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function  __trashcan__main()
        local  %0, person
          lea  R0, %0
         load  R1, 6
        store  [R0+0], R1
         load  R1, 5
        store  [R0+8], R1
         load  RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_layout_struct_in_correct_order_regardless_of_the_initialzer() {
        let code = r###"
        type person = {
            age: int,
            name: string,
        };
        var x = person {
            name: "helmut",
            age: 5,
        };
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function  __trashcan__main()
        local  %0, person
        local  %1, string
        const  .LC0, "helmut"
          lea  R0, %0
         load  R1, 5
        store  [R0+0], R1
          lea  R1, %1
        store  [R1+0], 6
          lea  R2, .LC0
        store  [R1+8], R2
          lea  R1, %1
         load  R2, [R1+0]
        store  [R0+8], R2
         load  R2, [R1+8]
        store  [R0+16], R2
         load  RET, 0
          ret
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_emit_copy_instructions_for_member_access_into_local() {
        let code = r###"
        type person = {
            age: int,
            name: string,
        };
        var x = person {
            name: "helmut",
            age: 5,
        };
        var y = x.name;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
    function  __trashcan__main()
       local  %0, person
       local  %1, string
       local  %2, string
       const  .LC0, "helmut"
         lea  R0, %0
        load  R1, 5
       store  [R0+0], R1
         lea  R1, %1
       store  [R1+0], 6
         lea  R2, .LC0
       store  [R1+8], R2
         lea  R1, %1
        load  R2, [R1+0]
       store  [R0+8], R2
        load  R2, [R1+8]
       store  [R0+16], R2
         lea  R0, %0
        load  R1, 8
         add  R0, R1
         lea  R1, %2
        load  R2, [R0+0]
       store  [R1+0], R2
        load  R2, [R0+8]
       store  [R1+8], R2
        load  RET, 0
         ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_else_if() {
        let code = r###"
        if 1 == 1 {

        } else if 2 == 2 {
            var x = 5;
        }
        var z = 5;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
      function __trashcan__main()
        local %0, int
        local %1, int
         load R0, 1
         load R1, 1
           eq R0, R1
        jumpz .LB1, R0
         jump .LB0
        label .LB1
         load R0, 2
         load R1, 2
           eq R0, R1
        jumpz .LB2, R0
         load R0, 5
          lea R1, %0
        store [R1+0], R0
         jump .LB0
        label .LB2
        label .LB0        
         load R0, 5
          lea R1, %1
        store [R1+0], R0
         load RET, 0
        ret
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_while() {
        let code = r###"
        while 1 == 2 {
          var x = 5;
        }
        "###;
        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function __trashcan__main()
          local %0, int
          label .LB0
           load R0, 1
           load R1, 2
             eq R0, R1
          jumpz .LB1, R0
           load R0, 5
            lea R1, %0
          store [R1+0], R0
           jump .LB0
          label .LB1
           load RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    fn assert_bytecode_matches(expected: &str, bc: &crate::bytecode::Bytecode) {
        let expected = expected.replace("  ", " ");
        let expected_lines: Vec<&str> = expected.trim().lines().map(|l| l.trim()).collect();
        let bc_s = bc.to_string().replace("  ", " ");
        let bc_lines: Vec<&str> = bc_s.trim().lines().map(|l| l.trim()).collect();

        println!("{}", bc);

        for i in 0..expected_lines.len() {
            let expected_line = &expected_lines[i];
            let actual_line = bc_lines.get(i).unwrap_or(&"<unknown line>");

            assert_eq!(expected_line, actual_line);
        }
    }
}
