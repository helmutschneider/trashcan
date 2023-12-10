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

pub trait VariableLike {
    fn subsegment_for_member(&self, name: &str) -> Rc<Variable>;
    fn find_parent_segment_and_member_offset(&self) -> (Rc<Variable>, Offset);
}

impl VariableLike for Rc<Variable> {
    fn subsegment_for_member(&self, name: &str) -> Rc<Variable> {
        let member = self.type_.find_member(name)
            .expect(&format!("struct member '{}' does not exist", name));

        let offset = match &self.offset {
            VariableOffset::Stack(_) => VariableOffset::Parent(Rc::clone(self), member.offset),
            VariableOffset::Parent(p, o) => VariableOffset::Parent(Rc::clone(p), o.add(member.offset)),
            VariableOffset::Dynamic(p, dyn_, o) => VariableOffset::Dynamic(
                Rc::clone(p), Rc::clone(dyn_), o.add(member.offset)
            ),
        };

        let type_ = if self.type_.is_pointer() {
            Type::Pointer(Box::new(member.type_))
        } else {
            member.type_
        };

        let next = Variable {
            name: format!("{}.{}", self.name, name),
            type_: type_,
            offset: offset,
        };

        return Rc::new(next);
    }

    // returns the segment that owns this subsegment (if there is one),
    // and the offset from that segment to this subsegment.
    fn find_parent_segment_and_member_offset(&self) -> (Rc<Variable>, Offset) {
        let mut iter = self;
        let mut offset = Offset::ZERO;
        while let VariableOffset::Parent(p, o) = &iter.offset {
            iter = p;
            offset = offset.add(*o);
        }
        return (Rc::clone(iter), offset);
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("{}", self.name));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
                bc.emit(Instruction::LoadInt(reg, *x));
            }
            Argument::Variable(x) => {
                bc.emit(Instruction::Load(reg, Rc::clone(x)));
            }
            _ => panic!("bad argument {}", self)
        }
    }
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Void => f.write_str("void"),
            Self::Bool(x) => x.fmt(f),
            Self::Int(x) => x.fmt(f),
            Self::Variable(v) => {
                v.name.fmt(f)
            }
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Rc<Variable>>),
    Alloc(Rc<Variable>),
    Label(String),
    Store(Rc<Variable>, Argument),
    StoreReg(Rc<Variable>, Reg),
    Load(Reg, Rc<Variable>),
    LoadInt(Reg, i64),
    AddressOf(Reg, Rc<Variable>),
    AddressOfConst(Reg, ConstId),
    Return,
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    Call(String, Vec<Argument>),
    IsEqual(Reg, Reg, Reg),
    Jump(String),
    JumpZero(String, Reg),
    Deref(Rc<Variable>, Rc<Variable>),
    StoreIndirect(Rc<Variable>, Argument),
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
            Self::Store(dest_var, source) => {
                format!("{:>12}  {}, {}", "store", dest_var, source)
            }
            Self::StoreReg(dest_var, reg) => {
                format!("{:>12}  {}, {}", "storer", dest_var, reg)
            }
            Self::Load(reg, mem) => {
                format!("{:>12}  {}, {}", "load", reg, mem)
            }
            Self::LoadInt(reg, x) => {
                format!("{:>12}  {}, {}", "loadi", reg, x)
            }
            Self::AddressOf(dest_var, source) => {
                format!("{:>12}  {}, {}", "lea", dest_var, source)
            }
            Self::AddressOfConst(dest_var, source) => {
                format!("{:>12}  {}, {}", "leac", dest_var, source)
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
            Self::IsEqual(dest_var, a, b) => {
                format!("{:>12}  {}, {}, {}", "eq", dest_var, a, b)
            }
            Self::Jump(to_label) => {
                format!("{:>12}  {}", "jump", to_label)
            }
            Self::JumpZero(to_label, reg) => {
                format!("{:>12}  {}, {}", "jumpz", to_label, reg)
            }
            Self::Deref(dest_var, source_var) => {
                format!("{:>12}  {}, {}", "deref", dest_var, source_var)
            }
            Self::StoreIndirect(dest_var, source) => {
                format!("{:>12}  {}, {}", "storeind", dest_var, source)
            }
            Self::Const(cons) => {
                let escaped = cons.value
                    .replace("\n", "\\n");
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
        return self.data.iter()
            .find(|x| x.name == name)
            .map(|x| Rc::clone(x))
            .expect(&format!("variable '{}' does not exist on the stack", name));
    }
    
    fn push(&mut self, type_: &Type) -> Rc<Variable> {
        let name = format!("%{}", self.data.len());
        let next_offset = self.data.last()
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
    pub live_regs: HashSet<Reg>,
    pub typer: Rc<typer::Typer>,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = Rc::new(typer::Typer::from_code(code)?);
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
            live_regs: HashSet::new(),
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
        bc.emit(Instruction::Function(ENTRYPOINT_NAME.to_string(), Vec::new()));
        for stmt in &main_statements {
            bc.compile_statement(stmt, &mut main_stack);
        }

        let main_has_return_statement = main_statements.last()
            .map(|x| matches!(x.as_ref(), Statement::Return(_)))
            .unwrap_or(false);

        if !main_has_return_statement {
            bc.emit(Instruction::LoadInt(Reg::RET, 0));
            bc.emit(Instruction::Return);
        }

        return Ok(bc);
    }

    fn emit(&mut self, instr: Instruction) {
        self.instructions.push(instr.clone());

        let reserve: Option<Reg> = match instr {
            Instruction::Load(r, _) => Some(r),
            Instruction::LoadInt(r, _) => Some(r),
            Instruction::AddressOf(r, _) => Some(r),
            Instruction::AddressOfConst(r, _) => Some(r),
            Instruction::Add(r1, _) => Some(r1),
            Instruction::Sub(r1, _) => Some(r1),
            Instruction::Mul(r1, _) => Some(r1),
            Instruction::Div(r1, _) => Some(r1),
            Instruction::IsEqual(r1, _, _) => Some(r1),
            _ => None,
        };

        if let Some(x) = reserve {
            self.live_regs.insert(x);
        }

        let release: Vec<Reg> = match instr {
            Instruction::StoreReg(_, r) => vec![r],
            Instruction::Add(_, r2) => vec![r2],
            Instruction::Sub(_, r2) => vec![r2],
            Instruction::Mul(_, r2) => vec![r2],
            Instruction::Div(_, r2) => vec![r2],
            Instruction::IsEqual(_, r2, r3) => vec![r2, r3],
            Instruction::JumpZero(_, r1) => vec![r1],
            _ => Vec::new(),
        };

        for x in &release {
            self.live_regs.remove(x);
        }
    }

    fn find_available_reg(&self) -> Reg {
        for reg in &GENERAL_PURPOSE_REGISTERS {
            if !self.live_regs.contains(reg) {
                return *reg;
            }
        }
        panic!("no available registers!");
    }

    fn add_label(&mut self) -> String {
        let label = format!(".LB{}", self.labels);
        self.labels += 1;
        return label;
    }

    fn emit_constant(&mut self, value: &str) -> ConstId {
        let num_constants = self.instructions.iter()
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
                let len_seg = var_ref.subsegment_for_member("length");
                self.emit(Instruction::Store(len_seg, Argument::Int(s.value.len() as i64)));

                let data_seg = var_ref.subsegment_for_member("data");
                let r1 = self.find_available_reg();
                self.emit(Instruction::AddressOfConst(r1, cons));
                self.emit(Instruction::StoreReg(Rc::clone(&data_seg), r1));

                Argument::Variable(Rc::clone(&var_ref))
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let dest_ref = self.emit_variable(&type_, stack);
                
                match bin_expr.operator.kind {
                    TokenKind::Plus => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);

                        self.emit(Instruction::Add(r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r1));
                    }
                    TokenKind::Minus => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let rhs = self.compile_expression(&bin_expr.right, stack);

                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);  
                        self.emit(Instruction::Sub(r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r1));
                    }
                    TokenKind::Star => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);
                    
                        self.emit(Instruction::Mul(r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r1));
                    }
                    TokenKind::Slash => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);
                    
                        self.emit(Instruction::Div(r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r1));
                    }
                    TokenKind::DoubleEquals => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);
                        
                        let r3 = self.find_available_reg();
                        self.emit(Instruction::IsEqual(r3, r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r3));
                    }
                    TokenKind::NotEquals => {
                        let lhs = self.compile_expression(&bin_expr.left, stack);
                        let r1 = self.find_available_reg();
                        lhs.load_into(r1, self);

                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        let r2 = self.find_available_reg();
                        rhs.load_into(r2, self);

                        let r3 = self.find_available_reg();
                        self.emit(Instruction::IsEqual(r3, r1, r2));
                        self.emit(Instruction::LoadInt(r1, 0));
                        self.emit(Instruction::IsEqual(r2, r3, r1));
                        self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), r2));
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
                            _ => (&bin_expr.left, false)
                        };
                        let lhs = self.compile_expression(&left_expr, stack);
                        let lhs_var = match lhs {
                            Argument::Variable(x) => x,
                            _ => panic!("left hand side is not a variable.")
                        };
                        let rhs = self.compile_expression(&bin_expr.right, stack);
                        if is_indirect {
                            self.emit_copy_indirect(&lhs_var, rhs);
                        } else {
                            self.emit_copy(&lhs_var, rhs);
                        }
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
                let fx_sym = self.typer
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
                self.emit(Instruction::Call(
                    call.name_token.value.clone(),
                    args,
                ));
                self.live_regs.insert(Reg::RET);
                self.emit(Instruction::StoreReg(Rc::clone(&dest_ref), Reg::RET));
                Argument::Variable(dest_ref)
            }
            ast::Expression::StructLiteral(s) => {
                let type_ = self.typer.try_find_symbol(&s.name_token.value, SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();

                // make sure to initialize the struct as declared by the type
                // and not in the order of initialized arguments.
                let members_by_name: HashMap<String, StructLiteralMember> = s.members.iter()
                    .map(|m| (m.field_name_token.value.clone(), m.clone()))
                    .collect();

                let dest_var = self.emit_variable(&type_, stack);
                let members = match &type_ {
                    Type::Struct(_, members) => members,
                    _ => panic!("type '{}' is not a struct", type_)
                };

                for m in members {
                    let member_dest = dest_var.subsegment_for_member(&m.name);
                    let value = &members_by_name.get(&m.name).unwrap().value;
                    let arg = self.compile_expression(&value, stack);
                    self.emit_copy(&member_dest, arg);
                }

                Argument::Variable(dest_var)
            }
            ast::Expression::MemberAccess(access) => {
                let left_arg = self.compile_expression(&access.left, stack);
                let left_var = match &left_arg {
                    Argument::Variable(x) => x,
                    _ => panic!("bad. wanted variable"),
                };
                let segment = left_var.subsegment_for_member(&access.right.name);
                let left_type = &left_var.type_;

                // if we're accessing a member of a pointer we should
                // emit a pointer add.
                if let Type::Pointer(_) = left_type {
                    let offset = match segment.offset {
                        VariableOffset::Parent(_, o) => o,
                        _ => panic!()
                    };
                    let ptr_temp = self.emit_variable(&segment.type_, stack);

                    let r1 = self.find_available_reg();
                    left_arg.load_into(r1, self);
                    let r2 = self.find_available_reg();
                    self.emit(Instruction::LoadInt(r2, offset.0));
                    self.emit(Instruction::Add(r1, r2));
                    self.emit(Instruction::StoreReg(Rc::clone(&ptr_temp), r1));

                    return Argument::Variable(ptr_temp);
                }

                return Argument::Variable(segment);
            }
            ast::Expression::BooleanLiteral(b) => Argument::Bool(b.value),
            ast::Expression::UnaryPrefix(unary_expr) => {
                let type_ = self.typer.try_infer_expression_type(&unary_expr.expr)
                    .unwrap();
                let arg = self.compile_expression(&unary_expr.expr, stack);
                let dest_ref: Rc<Variable> = match unary_expr.operator.kind {
                    TokenKind::Ampersand => {
                        let ptr_type = Type::Pointer(Box::new(type_));
                        let ptr_var = self.emit_variable(&ptr_type, stack);
                        let arg_var = match arg {
                            Argument::Variable(x) => x,
                            _ => panic!("not a variable bro")
                        };
                        let r1 = self.find_available_reg();
                        self.emit(Instruction::AddressOf(r1, Rc::clone(&arg_var)));
                        self.emit(Instruction::StoreReg(Rc::clone(&ptr_var), r1));
                        ptr_var
                    }
                    TokenKind::Star => {
                        let inner_type: Type = match type_ {
                            Type::Pointer(x) => *x,
                            _ => type_,
                        };
                        let arg_var = match arg {
                            Argument::Variable(x) => x,
                            _ => panic!("cannot dereference a constant"),
                        };
                        let deref_var = self.emit_variable(&inner_type, stack);
                        self.emit(Instruction::Deref(Rc::clone(&deref_var), arg_var));
                        deref_var
                    }
                    TokenKind::Minus => {
                        let neged_var = self.emit_variable(&Type::Int, stack);

                        let r1 = self.find_available_reg();
                        self.emit(Instruction::LoadInt(r1, 0));

                        let r2 = self.find_available_reg();
                        arg.load_into(r2, self);

                        self.emit(Instruction::Sub(r1, r2));
                        self.emit(Instruction::StoreReg(Rc::clone(&neged_var), r1));

                        neged_var
                    }
                    _ => panic!()
                };
                Argument::Variable(dest_ref)
            }
            ast::Expression::ArrayLiteral(array_lit) => {
                let type_ = self.typer.try_infer_expression_type(expr)
                    .unwrap();
                let dest_var = self.emit_variable(&type_, stack);
                let length_segment = dest_var.subsegment_for_member("length");
                let length = array_lit.elements.len() as i64;

                self.emit(Instruction::Store(length_segment, Argument::Int(length)));

                for k in 0..array_lit.elements.len() {
                    let elem_expr = &array_lit.elements[k];
                    let elem_segment = dest_var.subsegment_for_member(&k.to_string());
                    let elem_arg = self.compile_expression(elem_expr, stack);

                    self.emit_copy(&elem_segment, elem_arg);
                }

                Argument::Variable(dest_var)
            }
            ast::Expression::ElementAccess(elem_access) => {
                let mut iter = Some(elem_access);
                let mut root_left: Option<&Expression> = None;
                let mut path_to_value: Vec<&Expression> = Vec::new();

                while let Some(x) = iter {
                    path_to_value.insert(0, &x.right);
                    
                    match x.left.as_ref() {
                        Expression::ElementAccess(next) => {
                            iter = Some(next);
                        }
                        _ => {
                            root_left = Some(&x.left);
                            iter = None;
                        }
                    }
                }

                let ident = match root_left.unwrap() {
                    Expression::Identifier(x) => x,
                    _ => panic!("bad lvalue for element access bro."),
                };

                let root_var = stack.find(&ident.name);
                let total_offset = self.emit_variable(&Type::Int, stack);

                // zero-initialize the index variable.
                let r1 = self.find_available_reg();
                self.emit(Instruction::LoadInt(r1, 0));

                let mut iter_element_type = &root_var.type_;

                for x in path_to_value {
                    let length_member =  iter_element_type.find_member("length")
                        .unwrap();
                    iter_element_type = match iter_element_type {
                        Type::Array(x, _) => x,
                        _=> panic!(),
                    };
                    let iter_arg = self.compile_expression(x, stack);                    
                    
                    let r2 = self.find_available_reg();
                    iter_arg.load_into(r2, self);

                    let r3 = self.find_available_reg();
                    self.emit(Instruction::LoadInt(r3, iter_element_type.size()));
                    self.emit(Instruction::Mul(r2, r3));

                    // the first 8 bytes of an array is the length.
                    self.emit(Instruction::LoadInt(r3, length_member.type_.size()));
                    self.emit(Instruction::Add(r2, r3));

                    // move the calculated offset of this iteration into the total offset
                    // from the root array.
                    self.emit(Instruction::Add(r1, r2));
                }

                self.emit(Instruction::StoreReg(Rc::clone(&total_offset), r1));
                
                let element_var = Rc::new(Variable {
                    name: format!("{}[{}]", root_var, total_offset),
                    type_: self.typer.try_infer_expression_type(expr).unwrap(),
                    offset: VariableOffset::Dynamic(Rc::clone(&root_var), Rc::clone(&total_offset), Offset::ZERO)
                });

                Argument::Variable(element_var)
            }
        };
    }

    fn emit_variable(&mut self, type_: &Type, stack: &mut Stack) -> Rc<Variable> {
        let var = stack.push(type_);
        self.emit(Instruction::Alloc(Rc::clone(&var)));
        return var
    }

    fn compile_function(&mut self, fx: &ast::Function) {
        let mut stack = Stack::new();
        let arg_vars: Vec<Rc<Variable>> = fx
            .arguments
            .iter()
            .map(|fx_arg| {
                let fx_arg_sym = self.typer
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
            self.emit(Instruction::LoadInt(Reg::RET, 0));
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
                let var_sym = self.typer
                    .try_find_symbol(&var.name_token.value, SymbolKind::Local, var.parent)
                    .unwrap();

                let init_arg = self.compile_expression(&var.initializer, stack);
                let init_var = match &init_arg {
                    Argument::Variable(x) => {
                        if expression_needs_explicit_copy(&var.initializer) {
                            let var = self.emit_variable(&var_sym.type_, stack);
                            self.emit_copy(&var, init_arg);
                            var
                        } else {
                            Rc::clone(x)
                        }
                    }
                    _ => {
                        let var = self.emit_variable(&var_sym.type_, stack);
                        self.emit_copy(&var, init_arg);
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
                let r1 = self.find_available_reg();

                match cmp_arg {
                    Argument::Bool(x) => {
                        self.emit(Instruction::LoadInt(r1, x.into()));
                    }
                    Argument::Variable(x) => {
                        self.emit(Instruction::Load(r1, Rc::clone(&x)));
                    }
                    _ => panic!(),
                };

                self.emit(Instruction::JumpZero(label_after_block.clone(), r1));
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
        let r1 = self.find_available_reg();

        match condition {
            Argument::Bool(x) => {
                self.emit(Instruction::LoadInt(r1, x.into()));
            }
            Argument::Variable(x) => {
                self.emit(Instruction::Load(r1, Rc::clone(&x)));
            }
            _ => panic!()
        };

        self.emit(Instruction::JumpZero(label_after_block.clone(), r1));
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

    fn emit_copy(&mut self, dest_var: &Rc<Variable>, source: Argument) {
        assert_eq!(dest_var.type_, source.get_type());

        let type_ = source.get_type();
        let type_members = type_.members();

        if type_members.len() != 0 {
            for m in type_members {
                let member_seg = dest_var.subsegment_for_member(&m.name);
                let source_seg = match &source {
                    Argument::Variable(v) => Argument::Variable(v.subsegment_for_member(&m.name)),
                    _ => panic!("bad!")
                };
                self.emit_copy(&member_seg, source_seg);
            }
        } else {
            self.emit(Instruction::Store(Rc::clone(dest_var), source));
        }
    }

    fn emit_copy_indirect(&mut self, dest_var: &Rc<Variable>, source: Argument) {
        if let Type::Pointer(inner) = &dest_var.type_ {
            assert_eq!(inner.as_ref(), &source.get_type());
        } else {
            panic!("cannot store indirectly to a non-pointer.");
        }

        let type_ = source.get_type();
        let type_members = type_.members();

        if type_members.len() != 0 {
            for m in type_members {
                let member_seg = dest_var.subsegment_for_member(&m.name);
                let source_seg = match &source {
                    Argument::Variable(v) => v.subsegment_for_member(&m.name),
                    _ => panic!("bad!")
                };
                self.emit_copy_indirect(&member_seg, Argument::Variable(source_seg));
            }
        } else {
            self.emit(Instruction::StoreIndirect(Rc::clone(dest_var), source));
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

        for instr in &self.instructions {

        }
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

        let bc = Bytecode::from_code(code).unwrap();

        let expected = r###"
        function __trashcan__main()
          alloc %0, int
          store %0, 6
          loadi RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_assignment_with_reference() {
        let code = r###"
            var x: int = 6;
            var y: int = x;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function __trashcan__main()
          alloc %0, int
          store %0, 6
          alloc %1, int
          store %1, %0
          loadi RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_constant_add() {
        let code = r###"
            var x: int = 1 + 2;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function  __trashcan__main()
           alloc  %0, int
           loadi  R0, 1
           loadi  R1, 2
             add  R0, R1
          storer  %0, R0
           loadi RET, 0
             ret
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_function() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            function add(%0: int, %1: int)
                alloc %2, int
                load R0, %0
                load R1, %1
                add R0, R1
                storer %2, R0
                load RET, %2
                ret

            function __trashcan__main()
            loadi RET, 0
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
            loadi R0, 1
            loadi R1, 2
               eq R2, R0, R1
            storer %0, R2
            load R0, %0
            jumpz .LB1, R0
            alloc %1, int
            store %1, 42
             jump .LB0
            label .LB1
            alloc %2, int
            store %2, 3
            label .LB0
            loadi RET, 0
            ret
        "###;

        println!("{bc}");

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_struct() {
        let code = r###"
            var x: string = "hello";
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        println!("{bc}");

        let expected = r###"
        function __trashcan__main()
          const .LC0, "hello"
          alloc %0, string
          store %0.length, 5
           leac R0, .LC0
         storer %0.data, R0
          loadi RET, 0
          ret
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_pass_struct_argument_by_reference() {
        let code = r###"
            fun takes_str(x: &string): void {}
            takes_str(&"yee!");
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function takes_str(%0: &string)
          loadi RET, 0
          ret

        function __trashcan__main()
          const .LC0, "yee!"
          alloc %0, string
          store %0.length, 4
           leac R0, .LC0
         storer %0.data, R0
          alloc %1, &string
          lea R0, %0
         storer %1, R0
          alloc %2, void
          call takes_str(%1)
          storer %2, RET
          loadi RET, 0
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
          loadi RET, 0
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
          loadi RET, 0
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
          alloc %2, string
          store %2.length, %0.name.length
          store %2.data, %0.name.data
          loadi RET, 0
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
        loadi R0, 1
        loadi R1, 1
           eq R2, R0, R1
       storer %0, R2
         load R0, %0
        jumpz .LB1, R0
         jump .LB0
        label .LB1
        alloc %1, bool
        loadi R0, 2
        loadi R1, 2
           eq R2, R0, R1
       storer %1, R2
         load R0, %1
        jumpz .LB2, R0
        alloc %2, int
        store %2, 5
         jump .LB0
        label .LB2
        label .LB0
        alloc %3, int
        store %3, 5
        loadi RET, 0
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
          loadi R0, 1
          loadi R1, 2
             eq R2, R0, R1
          storer %0, R2
           load R0, %0
          jumpz .LB1, R0
          alloc %1, int
          store %1, 5
          jump .LB0
          label .LB1
          loadi RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_array() {
        let code = r###"
        var x = [420, 69];
        "###;
        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        function __trashcan__main()
          alloc %0, [int; 2]
          store %0.length, 2
          store %0.0, 420
          store %0.1, 69
          loadi RET, 0
          ret
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    fn assert_bytecode_matches(expected: &str, bc: &crate::bytecode::Bytecode) {
        let expected = expected
            .replace("  ", " ");
        let expected_lines: Vec<&str> = expected
            .trim()
            .lines().map(|l| l.trim()).collect();
        let bc_s = bc.to_string()
            .replace("  ", " ");
        let bc_lines: Vec<&str> = bc_s.trim().lines().map(|l| l.trim()).collect();

        println!("{}", bc);

        for i in 0..expected_lines.len() {
            let expected_line = &expected_lines[i];
            let actual_line = bc_lines.get(i).unwrap_or(&"<unknown line>");

            assert_eq!(expected_line, actual_line);
        }
    }
}
