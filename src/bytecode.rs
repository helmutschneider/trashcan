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
pub enum VariableOffset {
    Stack(Offset),
    Parent(Rc<Variable>, Offset),

    // offset dynamically from some parent variable with an offset
    // stored in the 2nd argument. used for array access. the 3rd
    // argument is for static offsets of from the dynamically
    // resolved value.
    Dynamic(Rc<Variable>, Rc<Variable>, Offset),
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: Type,
    pub offset: VariableOffset,
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("{}", self.name));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    RET,
}
const GENERAL_PURPOSE_REGISTERS: [Reg; 7] = [
    Reg::R0,
    Reg::R1,
    Reg::R2,
    Reg::R3,
    Reg::R4,
    Reg::R5,
    Reg::R6,
];

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Reg::*;
        let name = match self {
            R0 => stringify!(R0),
            R1 => stringify!(R1),
            R2 => stringify!(R2),
            R3 => stringify!(R3),
            R4 => stringify!(R4),
            R5 => stringify!(R5),
            R6 => stringify!(R6),
            RET => stringify!(RET),
        };
        return f.write_str(&name);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Indirect(pub Reg, pub i64);

impl std::fmt::Display for Indirect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = if self.1 < 0 { "-" } else { "+" };
        if self.1 != 0 {
            f.write_str(&format!("[{} {} {}]", self.0, op, self.1.abs()))
        } else {
            f.write_str(&format!("[{}]", self.0))
        }
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

#[derive(Debug, Clone)]
pub enum Argument {
    Void,
    Bool(bool),
    Int(i64),
    Variable(Rc<Variable>),
}

impl Argument {
    pub fn get_type(&self) -> Type {
        return match self {
            Self::Void => Type::Void,
            Self::Bool(_) => Type::Bool,
            Self::Int(_) => Type::Int,
            Self::Variable(v) => v.type_.clone(),
        };
    }

    pub fn load_into(&self, reg: Reg, bc: &mut Bytecode) {
        match self {
            Argument::Int(x) => {
                bc.emit(Instruction::LoadImm(reg, *x));
            }
            Argument::Variable(x) => {
                bc.emit(Instruction::LoadMem(reg, Rc::clone(x)));
            }
            _ => panic!("bad argument {}", self),
        }
    }
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Void => f.write_str("void"),
            Self::Bool(x) => x.fmt(f),
            Self::Int(x) => x.fmt(f),
            Self::Variable(v) => v.name.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Rc<Variable>>),
    Alloc(Rc<Variable>),
    Label(String),

    /** indirect memory store of register value. \[sp + offset] <- r2 */
    StoreMem(Rc<Variable>, Reg),

    /** indirect memory store of immediate value. \[r1\] <- value */
    StoreImm(Indirect, i64),

    /** indirect memory store of register value. \[r1\] <- r2 */
    StoreReg(Indirect, Reg),

    /** stack load r1 <- variable */
    LoadMem(Reg, Rc<Variable>),

    /** immediate load r1 <- value */
    LoadImm(Reg, i64),

    /** indirect load r1 <- \[r2\], where r2 should contain an address */
    LoadInd(Reg, Indirect),

    /** plain copy r1 <- r2 */
    LoadReg(Reg, Reg),
    
    AddressOf(Reg, Rc<Variable>),
    AddressOfConst(Reg, ConstId),
    Return,
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    Call(String, Vec<Argument>),
    IsEqual(Reg, Reg),
    Jump(String),
    JumpZero(String, Reg),
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
            Self::Alloc(var) => {
                format!("{:>12}  {}, {}", "alloc", var.name, var.type_)
            }
            Self::Label(name) => {
                format!("{:>12}  {}", "label", name)
            }
            Self::StoreMem(dest_var, source) => {
                format!("{:>12}  {}, {}", "store", dest_var, source)
            }
            Self::StoreImm(dest_var, source) => {
                format!("{:>12}  {}, {}", "store", dest_var, source)
            }
            Self::StoreReg(r1, r2) => {
                format!("{:>12}  {}, {}", "store", r1, r2)
            }
            Self::LoadMem(reg, mem) => {
                format!("{:>12}  {}, {}", "load", reg, mem)
            }
            Self::LoadImm(reg, x) => {
                format!("{:>12}  {}, {}", "load", reg, x)
            }
            Self::LoadInd(r1, r2) => {
                format!("{:>12}  {}, {}", "load", r1, r2)
            }
            Self::LoadReg(r1, r2) => {
                format!("{:>12}  {}, {}", "load", r1, r2)
            }
            Self::AddressOf(dest_var, source) => {
                format!("{:>12}  {}, {}", "lea", dest_var, source)
            }
            Self::AddressOfConst(dest_var, source) => {
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

fn expression_needs_explicit_copy(expr: &ast::Expression) -> bool {
    return match expr {
        ast::Expression::Void => true,
        ast::Expression::BooleanLiteral(_) => true,
        ast::Expression::IntegerLiteral(_) => true,
        ast::Expression::Identifier(_) => true,
        ast::Expression::MemberAccess(_) => true,
        ast::Expression::ElementAccess(_) => true,
        _ => false,
    };
}

#[derive(Debug, Clone)]
struct Stack {
    data: Vec<Rc<Variable>>,
    aliases: HashMap<String, Rc<Variable>>,
}

impl Stack {
    fn new() -> Self {
        return Self {
            data: Vec::new(),
            aliases: HashMap::new(),
        };
    }

    fn find(&self, name: &str) -> Rc<Variable> {
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

    fn push(&mut self, type_: &Type) -> Rc<Variable> {
        let name = format!("%{}", self.data.len());
        let next_offset = self
            .data
            .last()
            .map(|x| {
                let offset = match x.offset {
                    VariableOffset::Stack(x) => x,
                    _ => panic!("member variables should not be visible on the stack."),
                };
                return offset.add(x.type_.size());
            })
            .unwrap_or(Offset::ZERO);
        let var = Rc::new(Variable {
            name: name,
            type_: type_.clone(),
            offset: VariableOffset::Stack(next_offset),
        });
        self.data.push(Rc::clone(&var));
        return var;
    }

    fn add_alias(&mut self, var: &Rc<Variable>, as_name: &str) {
        self.aliases.insert(as_name.to_string(), Rc::clone(var));
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: i64,
    pub registers: Vec<Reg>,
    pub typer: Rc<typer::Typer>,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = Rc::new(typer::Typer::from_code(code)?);
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
            registers: GENERAL_PURPOSE_REGISTERS.to_vec(),
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
            bc.emit(Instruction::LoadImm(Reg::RET, 0));
            bc.emit(Instruction::Return);
        }

        return Ok(bc);
    }

    fn emit(&mut self, instr: Instruction) {
        self.instructions.push(instr.clone());
    }

    fn lock_registers<const N: usize, F: FnOnce(&mut Bytecode, [Reg; N]) -> ()>(&mut self, fx: F) {
        let mut regs = [Reg::R0; N];
        for k in 0..N {
            regs[k] = self.registers.remove(0);
        }
        fx(self, regs);

        for reg in regs {
            self.registers.push(reg);
        }
        self.registers.sort();
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
        self.emit(Instruction::Const(cons));
        return id;
    }

    fn compile_expression(&mut self, expr: &ast::Expression, stack: &mut Stack) -> Argument {
        return match expr {
            ast::Expression::Void => Argument::Void,
            ast::Expression::IntegerLiteral(x) => Argument::Int(x.value),
            ast::Expression::StringLiteral(s) => {
                let cons = self.emit_constant(&s.value);
                let var_ref = self.emit_variable(&Type::String, stack);

                self.lock_registers(|bc, [r1, r2]| {
                    bc.emit(Instruction::AddressOf(r1, Rc::clone(&var_ref)));
                    bc.emit(Instruction::StoreImm(Indirect(r1, 0), s.value.len() as i64));
                    
                    bc.emit(Instruction::AddressOfConst(r2, cons));
                    bc.emit(Instruction::StoreReg(Indirect(r1, 8), r2));
                });

                Argument::Variable(Rc::clone(&var_ref))
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let dest_ref = self.emit_variable(&type_, stack);

                match bin_expr.operator.kind {
                    TokenKind::Plus => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::Add(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::Minus => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::Sub(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::Star => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::Mul(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::Slash => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::Div(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::DoubleEquals => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::IsEqual(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::NotEquals => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            lhs.load_into(r1, bc);
                            rhs.load_into(r2, bc);
    
                            bc.emit(Instruction::IsEqual(r1, r2));
                            bc.emit(Instruction::LoadImm(r2, 0));
                            bc.emit(Instruction::IsEqual(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&dest_ref), r1));
                        });
                    }
                    TokenKind::Equals => {
                        // indirect means that we're storing something at the address
                        // of the left hand side. the left hand side will look like
                        // a pointer deref.
                        let (left_expr, is_indirect) = match bin_expr.left.as_ref() {
                            Expression::UnaryPrefix(expr) => {
                                if expr.operator.kind == TokenKind::Star {
                                    (&expr.expr, true)
                                } else {
                                    (&bin_expr.left, false)
                                }
                            }
                            _ => (&bin_expr.left, false),
                        };
                        let lhs = self.compile_expression(&left_expr, stack);
                        let lhs_var = match lhs {
                            Argument::Variable(x) => x,
                            _ => panic!("left hand side is not a variable."),
                        };
                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        self.emit_copy_v2(&lhs_var, &rhs, stack);
                    }
                    _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
                };

                Argument::Variable(dest_ref)
            }
            ast::Expression::Identifier(ident) => {
                let var = stack.find(&ident.name);
                Argument::Variable(var)
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

                let mut args: Vec<Argument> = Vec::new();

                for k in 0..arg_types.len() {
                    let given_arg = self.compile_expression(&call.arguments[k], stack);
                    args.push(given_arg);
                }

                let dest_ref = self.emit_variable(&ret_type, stack);
                self.emit(Instruction::Call(call.name_token.value.clone(), args));
                self.emit(Instruction::StoreMem(Rc::clone(&dest_ref), Reg::RET));
                Argument::Variable(dest_ref)
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

                self.lock_registers(|bc, [r1, r2]| {
                    bc.emit(Instruction::AddressOf(r1, Rc::clone(&dest_var)));

                    let members = match &type_ {
                        Type::Struct(_, members) => members,
                        _ => panic!("type '{}' is not a struct", type_),
                    };
    
                    for m in members {
                        bc.emit_member_address(r2, r1, &dest_var.type_, &m.name);
                        let value = &members_by_name.get(&m.name).unwrap().value;
                        let arg = bc.compile_expression(&value, stack);
                        // self.emit_copy(&member_dest, arg);
                    }
                });

                Argument::Variable(dest_var)
            }
            ast::Expression::MemberAccess(access) => {
                todo!();
            }
            ast::Expression::BooleanLiteral(b) => Argument::Bool(b.value),
            ast::Expression::UnaryPrefix(unary_expr) => {
                let type_ = self
                    .typer
                    .try_infer_expression_type(&unary_expr.expr)
                    .unwrap();
                let arg = self.compile_expression(&unary_expr.expr, stack);
                let dest_ref: Rc<Variable> = match unary_expr.operator.kind {
                    TokenKind::Ampersand => {
                        let ptr_type = Type::Pointer(Box::new(type_));
                        let ptr_var = self.emit_variable(&ptr_type, stack);
                        let arg_var = match arg {
                            Argument::Variable(x) => x,
                            _ => panic!("not a variable bro"),
                        };

                        self.lock_registers(|bc, [r1]| {
                            bc.emit(Instruction::AddressOf(r1, Rc::clone(&arg_var)));
                            bc.emit(Instruction::StoreMem(Rc::clone(&ptr_var), r1));
                        });

                        ptr_var
                    }
                    TokenKind::Star => {
                        let inner_type: Type = match type_ {
                            Type::Pointer(x) => *x,
                            _ => type_,
                        };
                        let deref_var = self.emit_variable(&inner_type, stack);
                        self.emit_copy_v2(&deref_var, &arg, stack);
                        deref_var
                    }
                    TokenKind::Minus => {
                        let neged_var = self.emit_variable(&Type::Int, stack);

                        self.lock_registers(|bc, [r1, r2]| {
                            bc.emit(Instruction::LoadImm(r1, 0));
                            arg.load_into(r2, bc);
                            bc.emit(Instruction::Sub(r1, r2));
                            bc.emit(Instruction::StoreMem(Rc::clone(&neged_var), r1));
                        });

                        neged_var
                    }
                    _ => panic!(),
                };
                Argument::Variable(dest_ref)
            }
            ast::Expression::ArrayLiteral(array_lit) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let dest_var = self.emit_variable(&type_, stack);

                self.lock_registers(|bc, [r1]| {
                    bc.emit(Instruction::AddressOf(r1, Rc::clone(&dest_var)));

                    let length = array_lit.elements.len() as i64;
                    bc.emit(Instruction::StoreImm(Indirect(r1, 0), length));
    
                    for k in 0..array_lit.elements.len() {
                        let elem_expr = &array_lit.elements[k];
    
                        // self.emit_member_address(r1, dest_addr, &dest_var.type_, &k.to_string());
                        // let elem_arg = self.compile_expression(elem_expr, stack);
    
                        // self.emit_copy_v2(&elem_segment, elem_arg);
                    }
                });



                Argument::Variable(dest_var)
            }
            ast::Expression::ElementAccess(elem_access) => {
                todo!();
            }
        };
    }

    fn emit_variable(&mut self, type_: &Type, stack: &mut Stack) -> Rc<Variable> {
        let var = stack.push(type_);
        self.emit(Instruction::Alloc(Rc::clone(&var)));
        return var;
    }

    fn compile_function(&mut self, fx: &ast::Function) {
        let mut stack = Stack::new();
        let arg_vars: Vec<Rc<Variable>> = fx
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
            self.emit(Instruction::LoadImm(Reg::RET, 0));
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
                let init_var = match &init_arg {
                    Argument::Variable(x) => {
                        if expression_needs_explicit_copy(&var.initializer) {
                            let var = self.emit_variable(&var_sym.type_, stack);
                            self.emit_copy_v2(&var, &init_arg, stack);
                            var
                        } else {
                            Rc::clone(x)
                        }
                    }
                    _ => {
                        let var = self.emit_variable(&var_sym.type_, stack);
                        self.emit_copy_v2(&var, &init_arg, stack);
                        var
                    }
                };
                stack.add_alias(&init_var, &var_sym.name);
            }
            ast::Statement::Return(ret) => {
                let ret_arg = self.compile_expression(&ret.expr, stack);
                ret_arg.load_into(Reg::RET, self);
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

                self.lock_registers(|bc, [r1]| {
                    match cmp_arg {
                        Argument::Bool(x) => {
                            bc.emit(Instruction::LoadImm(r1, x.into()));
                        }
                        Argument::Variable(x) => {
                            bc.emit(Instruction::LoadMem(r1, Rc::clone(&x)));
                        }
                        _ => panic!(),
                    };
    
                    bc.emit(Instruction::JumpZero(label_after_block.clone(), r1));
                });

                self.compile_block(while_.block.as_block(), stack);
                self.emit(Instruction::Jump(label_before_condition));
                self.emit(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                self.compile_expression(&expr.expr, stack);
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

        self.lock_registers(|bc, [r1]| {
            match condition {
                Argument::Bool(x) => {
                    bc.emit(Instruction::LoadImm(r1, x.into()));
                }
                Argument::Variable(x) => {
                    bc.emit(Instruction::LoadMem(r1, Rc::clone(&x)));
                }
                _ => panic!(),
            };
    
            bc.emit(Instruction::JumpZero(label_after_block.clone(), r1));
        });

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

    fn emit_copy_v2(&mut self, dest: &Rc<Variable>, source: &Argument, stack: &mut Stack) {
        self.lock_registers(|bc, [r1, r2, r3]| {
            let size: i64;

            if let Type::Pointer(inner) = &dest.type_ {
                bc.emit(Instruction::LoadMem(r1, Rc::clone(dest)));
                size = inner.size();
            } else {
                bc.emit(Instruction::AddressOf(r1, Rc::clone(dest)));
                size = dest.type_.size();
            }
    
            if let Argument::Bool(x) = source {
                bc.emit(Instruction::StoreImm(Indirect(r1, 0), (*x).into()));
                return;
            }
            if let Argument::Int(x) = source {
                bc.emit(Instruction::StoreImm(Indirect(r1, 0), *x));
                return;
            }
    
            if let Argument::Variable(x) = source {
                if x.type_.is_pointer() {
                    bc.emit(Instruction::LoadMem(r2, Rc::clone(&x)));
                } else {
                    bc.emit(Instruction::AddressOf(r2, Rc::clone(&x)));
                }
            } else {
                panic!();
            }
    
            let size_t = Type::Pointer(Box::new(Type::Void)).size();
    
            let mut offset: i64 = 0;
            while offset < size {
                bc.emit(Instruction::LoadInd(r3, Indirect(r2, offset)));
                bc.emit(Instruction::StoreReg(Indirect(r1, offset), r3));
                offset += size_t;
            }
        });
    }

    fn emit_member_address(&mut self, dest: Reg, source_addr: Reg, source_type: &Type, member: &str) {
        self.emit(Instruction::LoadReg(dest, source_addr));
        let member = source_type.find_member(member)
            .unwrap();

        if member.offset != Offset::ZERO {
            self.lock_registers(|bc, [r1]| {
                bc.emit(Instruction::LoadImm(r1, member.offset.0));
                bc.emit(Instruction::Add(dest, r1));
            });
        }
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
             alloc  %0, int
              load  R0, 1
              load  R1, 2
               add  R0, R1
             store  %0, R0
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
            alloc %0, bool
             load R0, 1
             load R1, 2
               eq R0, R1
            store %0, R0
             load R0, %0
            jumpz .LB1, R0
            alloc %1, int
              lea R0, %1
            store [R0], 42
             jump .LB0
            label .LB1
            alloc %2, int
              lea R0, %2
            store [R0], 3
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
        function __trashcan__main()
          alloc %0, person
          store %0.id, 6
          store %0.age, 5
          loadimm RET, 0
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
        function __trashcan__main()
          alloc %0, person
          store %0.age, 5
          const .LC0, "helmut"
          alloc %1, string
          store %1.length, 6
           leac R0, .LC0
         storer %1.data, R0
          store %0.name.length, %1.length
          store %0.name.data, %1.data
          loadimm RET, 0
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
        alloc  %0, person
          lea  R0, %0
        store  [R0], 5
        const  .LC0, "helmut"
        alloc  %1, string
        store  %1.length, 6
         leac  R0, .LC0
        store  %1.data, R0
        store  %0.name.length, %1.length
        store  %0.name.data, %1.data
        alloc  %2, string
          lea  R0, %2
          lea  R1, %0.name
         load  R2, [R1]
        store  [R0], R2
         load  R2, 8
          add  R0, R2
          add  R1, R2
         load  R2, [R1]
        store  [R0], R2
         load  R2, 8
          add  R0, R2
          add  R1, R2
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
        alloc %0, bool
         load R0, 1
         load R1, 1
           eq R0, R1
        store %0, R0
         load R0, %0
        jumpz .LB1, R0
         jump .LB0
        label .LB1
        alloc %1, bool
         load R0, 2
         load R1, 2
           eq R0, R1
        store %1, R0
         load R0, %1
        jumpz .LB2, R0
        alloc %2, int
          lea R0, %2
        store [R0], 5
         jump .LB0
        label .LB2
        label .LB0
        alloc %3, int
          lea R0, %3
        store [R0], 5
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
          label .LB0
          alloc %0, bool
           load R0, 1
           load R1, 2
             eq R0, R1
          store %0, R0
           load R0, %0
          jumpz .LB1, R0
          alloc %1, int
            lea R0, %1
          store [R0], 5
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
