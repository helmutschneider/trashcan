use std::collections::HashMap;
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
    Parent(Box<Variable>, Offset),

    // offset dynamically from some parent variable with an offset
    // stored in the 2nd argument. used for array access. the 3rd
    // argument is for static offsets of from the dynamically
    // resolved value.
    Dynamic(Box<Variable>, Box<Variable>, Offset),
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: Type,
    pub offset: VariableOffset,
}

impl Variable {
    pub fn subsegment_for_member(&self, name: &str) -> Variable {
        let member = self.type_.find_member(name)
            .expect(&format!("struct member '{}' does not exist", name));

        let offset = match &self.offset {
            VariableOffset::Stack(_) => VariableOffset::Parent(Box::new(self.clone()), member.offset),
            VariableOffset::Parent(p, o) => VariableOffset::Parent(p.clone(), o.add(member.offset)),
            VariableOffset::Dynamic(p, dyn_, o) => VariableOffset::Dynamic(p.clone(), dyn_.clone(), o.add(member.offset)),
        };

        let type_ = if self.type_.is_pointer() {
            Type::Pointer(Box::new(member.type_))
        } else {
            member.type_
        };

        return Self {
            name: format!("{}.{}", self.name, name),
            type_: type_,
            offset: offset,
        };
    }

    // returns the segment that owns this subsegment (if there is one),
    // and the offset from that segment to this subsegment.
    pub fn find_parent_segment_and_member_offset(&self) -> (Variable, Offset) {
        let mut iter = self;
        let mut offset = Offset::ZERO;
        while let VariableOffset::Parent(p, o) = &iter.offset {
            iter = p.as_ref();
            offset = offset.add(*o);
        }
        return (iter.clone(), offset);
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("{}", self.name));
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Void,
    Bool(bool),
    Int(i64),
    String(String),
    Variable(Variable),
}

impl Argument {
    pub fn get_type(&self) -> Type {
        return match self {
            Self::Void => Type::Void,
            Self::Bool(_) => Type::Bool,
            Self::Int(_) => Type::Int,
            Self::String(_) => panic!("bad! got string argument."),
            Self::Variable(v) => v.type_.clone(),
        };
    }
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Void => f.write_str("void"),
            Self::Bool(x) => x.fmt(f),
            Self::Int(x) => x.fmt(f),
            Self::String(s) => {
                f.write_str(&format!("\"{}\"", escape(s)))
            }
            Self::Variable(v) => {
                v.name.fmt(f)
            }
        };
    }
}

fn escape(value: &str) -> String {
    return value
        .replace("\n", "\\n")
        .replace("\t", "\\t");
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Local(Variable),
    Label(String),
    Store(Variable, Argument),
    AddressOf(Variable, Argument),
    Return(Argument),
    Add(Variable, Argument, Argument),
    Sub(Variable, Argument, Argument),
    Mul(Variable, Argument, Argument),
    Div(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
    IsEqual(Variable, Argument, Argument),
    JumpNotEqual(String, Argument, Argument),
    Deref(Variable, Variable),
    StoreIndirect(Variable, Argument),
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
                format!("{}({}):", name, args_s)
            }
            Self::Local(var) => {
                format!("  local {}, {}", var.name, var.type_)
            }
            Self::Label(name) => {
                format!("{}:", name)
            }
            Self::Store(dest_var, source) => {
                format!("  store {}, {}", dest_var, source)
            }
            Self::AddressOf(dest_var, source) => {
                format!("  lea {}, {}", dest_var, source)
            }
            Self::Return(value) => {
                format!("  ret {}", value)
            }
            Self::Add(dest_var, x, y) => {
                format!("  add {}, {}, {}", dest_var, x, y)
            }
            Self::Sub(dest_var, x, y) => {
                format!("  sub {}, {}, {}", dest_var, x, y)
            }
            Self::Mul(dest_var, x, y) => {
                format!("  mul {}, {}, {}", dest_var, x, y)
            }
            Self::Div(dest_var, x, y) => {
                format!("  div {}, {}, {}", dest_var, x, y)
            }
            Self::Call(dest_var, name, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  call {}, {}({})", dest_var, name, arg_s)
            }
            Self::IsEqual(dest_var, a, b) => {
                format!("  eq {}, {}, {}", dest_var, a, b)
            }
            Self::JumpNotEqual(to_label, a, b) => {
                format!("  jne {}, {}, {}", to_label, a, b)
            }
            Self::Deref(dest_var, source_var) => {
                format!("  deref {}, {}", dest_var, source_var)
            }
            Self::StoreIndirect(dest_var, source) => {
                format!("  storeind {}, {}", dest_var, source)
            }
        };
        return f.write_str(&s);
    }
}

fn expression_needs_explicit_copy(expr: &ast::Expression) -> bool {
    return match expr {
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
    data: Vec<Variable>,
    temporaries: i64,
}

impl Stack {
    fn new() -> Self {
        return Self {
            data: Vec::new(),
            temporaries: 0,
        };
    }

    fn find(&self, name: &str) -> Variable {
        return self.data.iter()
            .find(|x| x.name == name)
            .cloned()
            .expect(&format!("variable '{}' does not exist on the stack", name));
    }
    
    fn push(&mut self, name: &str, type_: &Type) -> Variable {
        let next_offset = self.data.last()
            .map(|x| {
                let offset = match x.offset {
                    VariableOffset::Stack(x) => x,
                    _ => panic!("member variables should not be visible on the stack."),
                };
                return offset.add(x.type_.size());
            })
            .unwrap_or(Offset::ZERO);
        let var = Variable {
            name: name.to_string(),
            type_: type_.clone(),
            offset: VariableOffset::Stack(next_offset),
        };
        self.data.push(var.clone());
        return var;
    }

    fn push_temporary(&mut self, type_: &Type) -> Variable {
        let name = format!("%{}", self.temporaries);
        self.temporaries += 1;
        return self.push(&name, type_);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: i64,
    pub typer: Rc<typer::Typer>,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = Rc::new(typer::Typer::from_code(code)?);
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
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
        bc.emit(Instruction::Return(Argument::Void));

        return Ok(bc);
    }

    fn emit(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    fn add_label(&mut self) -> String {
        let label = format!(".LB{}", self.labels);
        self.labels += 1;
        return label;
    }

    fn compile_expression(
        &mut self,
        expr: &ast::Expression,
        maybe_dest_var: Option<&Variable>,
        stack: &mut Stack
    ) -> Argument {
        let value = match expr {
            ast::Expression::Void => Argument::Void,
            ast::Expression::IntegerLiteral(x) => Argument::Int(x.value),
            ast::Expression::StringLiteral(s) => {
                let var_ref = self.maybe_add_temp_variable(maybe_dest_var, &Type::String, stack);

                let arg_len = Argument::Int(s.value.len() as i64);
                let arg_len_seg = var_ref.subsegment_for_member("length");
                let arg_data = Argument::String(escape(&s.value));
                let arg_data_seg = var_ref.subsegment_for_member("data");

                self.emit(Instruction::Store(arg_len_seg, arg_len));
                self.emit(Instruction::AddressOf(arg_data_seg, arg_data));

                Argument::Variable(var_ref)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let dest_ref: Variable;
                
                match bin_expr.operator.kind {
                    TokenKind::Plus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        let instr = Instruction::Add(dest_ref.clone(), lhs, rhs);
                        self.emit(instr);
                    }
                    TokenKind::Minus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        let instr = Instruction::Sub(dest_ref.clone(), lhs, rhs);
                        self.emit(instr);
                    }
                    TokenKind::Star => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        let instr = Instruction::Mul(dest_ref.clone(), lhs, rhs);
                        self.emit(instr);
                    }
                    TokenKind::Slash => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        let instr = Instruction::Div(dest_ref.clone(), lhs, rhs);
                        self.emit(instr);
                    }
                    TokenKind::DoubleEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        let instr = Instruction::IsEqual(dest_ref.clone(), lhs, rhs);
                        self.emit(instr);
                    }
                    TokenKind::NotEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let temp = stack.push_temporary(&Type::Bool);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        self.emit(Instruction::IsEqual(temp.clone(), lhs, rhs));
                        let instr = Instruction::IsEqual(dest_ref.clone(), Argument::Variable(temp), Argument::Bool(false));
                        self.emit(instr);
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
                        let lhs = self.compile_expression(&left_expr, None, stack);
                        let lhs_var = match lhs {
                            Argument::Variable(x) => x,
                            _ => panic!("left hand side is not a variable.")
                        };
                        dest_ref = lhs_var.clone();
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        if is_indirect {
                            self.compile_stack_copy_indirect(&dest_ref, &rhs, stack);
                        } else {
                            self.compile_stack_copy(&dest_ref, &rhs, stack);
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
                    let given_arg = self.compile_expression(&call.arguments[k], None, stack);
                    args.push(given_arg);
                }

                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &ret_type, stack);
                self.emit(Instruction::Call(
                    dest_ref.clone(),
                    call.name_token.value.clone(),
                    args,
                ));
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

                let dest_var = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                let members = match &type_ {
                    Type::Struct(_, members) => members,
                    _ => panic!("type '{}' is not a struct", type_)
                };

                for m in members {
                    let member_dest = dest_var.subsegment_for_member(&m.name);
                    let value = &members_by_name.get(&m.name).unwrap().value;
                    let arg = self.compile_expression(&value, Some(&member_dest), stack);

                    if expression_needs_explicit_copy(value) {
                        self.compile_stack_copy(&member_dest, &arg, stack);
                    }
                }

                Argument::Variable(dest_var)
            }
            ast::Expression::MemberAccess(access) => {
                let left_arg = self.compile_expression(&access.left, None, stack);
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
                    let ptr_temp = self.maybe_add_temp_variable(None, &segment.type_, stack);
                    self.emit(Instruction::Store(ptr_temp.clone(), left_arg));
                    self.emit(Instruction::Add(ptr_temp.clone(), Argument::Variable(ptr_temp.clone()), Argument::Int(offset.0)));
                    return Argument::Variable(ptr_temp);
                }

                return Argument::Variable(segment);
            }
            ast::Expression::BooleanLiteral(b) => Argument::Bool(b.value),
            ast::Expression::UnaryPrefix(unary_expr) => {
                let type_ = self.typer.try_infer_expression_type(&unary_expr.expr)
                    .unwrap();
                let arg = self.compile_expression(&unary_expr.expr, None, stack);
                let dest_ref: Variable;

                match unary_expr.operator.kind {
                    TokenKind::Ampersand => {
                        let ptr_type = Type::Pointer(Box::new(type_));
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &ptr_type, stack);
                        self.emit(Instruction::AddressOf(dest_ref.clone(), arg))
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
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &inner_type, stack);
                        self.emit(Instruction::Deref(dest_ref.clone(), arg_var))
                    }
                    TokenKind::Minus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        self.emit(Instruction::Sub(dest_ref.clone(), Argument::Int(0), arg))
                    }
                    _ => panic!()
                }

                Argument::Variable(dest_ref)
            }
            ast::Expression::ArrayLiteral(array_lit) => {
                let type_ = self.typer.try_infer_expression_type(expr)
                    .unwrap();
                let dest_var = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                let length_segment = dest_var.subsegment_for_member("length");
                let length = array_lit.elements.len() as i64;

                self.emit(Instruction::Store(length_segment, Argument::Int(length)));

                for k in 0..array_lit.elements.len() {
                    let elem_expr = &array_lit.elements[k];
                    let elem_segment = dest_var.subsegment_for_member(&k.to_string());
                    let elem_arg = self.compile_expression(elem_expr, Some(&elem_segment), stack);

                    if expression_needs_explicit_copy(elem_expr) {
                        self.emit(Instruction::Store(elem_segment, elem_arg));
                    }
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
                let total_offset = self.maybe_add_temp_variable(None, &Type::Int, stack);

                // zero-initialize the index variable.
                self.emit(Instruction::Store(total_offset.clone(), Argument::Int(0)));

                let iter_offset = self.maybe_add_temp_variable(None, &Type::Int, stack);
                let mut iter_element_type = &root_var.type_;

                for x in path_to_value {
                    let length_member =  iter_element_type.find_member("length")
                        .unwrap();
                    iter_element_type = match iter_element_type {
                        Type::Array(x, _) => x,
                        _=> panic!(),
                    };
                    let arg = self.compile_expression(x, Some(&iter_offset.clone()), stack);

                    self.emit(Instruction::Mul(iter_offset.clone(), arg, Argument::Int(iter_element_type.size())));

                    // the first 8 bytes of an array is the length.
                    self.emit(Instruction::Add(iter_offset.clone(), Argument::Variable(iter_offset.clone()), Argument::Int(length_member.type_.size())));

                    // move the calculated offset of this iteration into the total offset
                    // from the root array.
                    self.emit(Instruction::Add(total_offset.clone(), Argument::Variable(total_offset.clone()), Argument::Variable(iter_offset.clone())));
                }
                
                let element_var = Variable {
                    name: format!("{}[{}]", root_var, total_offset),
                    type_: self.typer.try_infer_expression_type(expr).unwrap(),
                    offset: VariableOffset::Dynamic(Box::new(root_var), Box::new(total_offset), Offset::ZERO)
                };

                Argument::Variable(element_var)
            }
        };
        return value;
    }

    fn maybe_add_temp_variable(&mut self, dest_var: Option<&Variable>, type_: &Type, stack: &mut Stack) -> Variable {
        return match dest_var {
            Some(v) => {
                assert_eq!(&v.type_, type_);
                v.clone()
            }
            None => {
                let var = stack.push_temporary(type_);
                self.emit(Instruction::Local(var.clone()));
                var
            }
        };
    }

    fn compile_function(&mut self, fx: &ast::Function) {
        let mut stack = Stack::new();
        let arg_vars: Vec<Variable> = fx
            .arguments
            .iter()
            .map(|fx_arg| {
                let fx_arg_sym = self.typer
                    .try_find_symbol(&fx_arg.name_token.value, SymbolKind::Local, fx.body.id())
                    .unwrap();

                let fx_arg_type = fx_arg_sym.type_;

                stack.push(&fx_arg.name_token.value, &fx_arg_type)
            })
            .collect();

        self.instructions
            .push(Instruction::Function(fx.name_token.value.clone(), arg_vars));

        self.compile_block(fx.body.as_block(), &mut stack);

        // add an implicit return statement if the function doesn't have one.
        if !matches!(self.instructions.last(), Some(Instruction::Return(_))) {
            self.emit(Instruction::Return(Argument::Void));
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

                let var_ref = stack.push(&var_sym.name, &var_sym.type_);
                self.emit(Instruction::Local(var_ref.clone()));

                let init_arg = self.compile_expression(&var.initializer, Some(&var_ref), stack);

                // literals and identifier expressions don't emit any stack
                // variables, so we need an implicit copy here.
                if expression_needs_explicit_copy(&var.initializer) {
                    self.compile_stack_copy(&var_ref, &init_arg, stack);
                }
            }
            ast::Statement::Return(ret) => {
                let ret_arg = self.compile_expression(&ret.expr, None, stack);
                self.emit(Instruction::Return(ret_arg));
            }
            ast::Statement::If(if_stmt) => {
                let label_after_last_block = self.add_label();

                self.compile_if(&if_stmt, &label_after_last_block, stack);
            }
            ast::Statement::While(while_) => {
                let label_before_condition = self.add_label();
                let label_after_block = self.add_label();

                self.emit(Instruction::Label(label_before_condition.clone()));
                let cmp_arg = self.compile_expression(&while_.condition, None, stack);
                self.emit(Instruction::JumpNotEqual(label_after_block.clone(), cmp_arg, Argument::Bool(true)));
                self.compile_block(while_.block.as_block(), stack);
                self.emit(Instruction::JumpNotEqual(label_before_condition, Argument::Bool(true), Argument::Bool(false)));
                self.emit(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                self.compile_expression(&expr.expr, None, stack);
            }
            ast::Statement::Type(_) => {
                // do nothing. type declarations aren't represented by specific
                // instructions in the bytecode.
            }
        }
    }

    fn compile_if(&mut self, if_stmt: &ast::If, label_after_last_block: &str, stack: &mut Stack) {
        let condition = self.compile_expression(&if_stmt.condition, None, stack);
        let label_after_block = self.add_label();
        self.emit(Instruction::JumpNotEqual(
            label_after_block.clone(),
            condition,
            Argument::Bool(true),
        ));
        self.compile_block(if_stmt.block.as_block(), stack);
        self.emit(Instruction::JumpNotEqual(label_after_last_block.to_string(), Argument::Bool(true), Argument::Bool(false)));
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

    fn compile_stack_copy(&mut self, dest_var: &Variable, source: &Argument, stack: &mut Stack) {
        assert_eq!(dest_var.type_, source.get_type());

        let type_ = source.get_type();
        let type_members = type_.members();

        if type_members.len() != 0 {
            for m in type_members {
                let member_seg = dest_var.subsegment_for_member(&m.name);
                let source_seg = match source {
                    Argument::Variable(v) => v.subsegment_for_member(&m.name),
                    _ => panic!("bad!")
                };
                self.compile_stack_copy(&member_seg, &Argument::Variable(source_seg), stack);
            }
        } else {
            self.emit(Instruction::Store(dest_var.clone(), source.clone()));
        }
    }

    fn compile_stack_copy_indirect(&mut self, dest_var: &Variable, source: &Argument, stack: &mut Stack) {
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
                let source_seg = match source {
                    Argument::Variable(v) => v.subsegment_for_member(&m.name),
                    _ => panic!("bad!")
                };
                self.compile_stack_copy_indirect(&member_seg, &Argument::Variable(source_seg), stack);
            }
        } else {
            self.emit(Instruction::StoreIndirect(dest_var.clone(), source.clone()));
        }
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
        __trashcan__main():
          local x, int
          store x, 6
          ret void
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
        __trashcan__main():
          local x, int
          store x, 6
          local y, int
          store y, x
          ret void
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
        __trashcan__main():
          local x, int
          add x, 1, 2
          ret void
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
            add(x: int, y: int):
                local %0, int
                add %0, x, y
                ret %0
            __trashcan__main():
              ret void
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_double_equals() {
        let code = r###"
        var x: int = 2;
        var y: bool = (x + 1) == 3;
    "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        __trashcan__main():
          local x, int
          store x, 2
          local y, bool
          local %0, int
          add %0, x, 1
          eq y, %0, 3
          ret void
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
        __trashcan__main():
            local x, bool
            eq x, 1, 2
            jne .LB1, x, true
            local y, int
            store y, 42
            jne .LB0, true, false
        .LB1:
            local z, int
            store z, 3
        .LB0:
            ret void
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
        __trashcan__main():
          local x, string
          store x.length, 5
          lea x.data, "hello"
          ret void
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
        takes_str(x: &string):
          ret void
        __trashcan__main():
          local %0, string
          store %0.length, 4
          lea %0.data, "yee!"
          local %1, &string
          lea %1, %0
          local %2, void
          call %2, takes_str(%1)
          ret void
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
        __trashcan__main():
          local x, person
          store x.id, 6
          store x.age, 5
          ret void
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
        __trashcan__main():
          local x, person
          store x.age, 5
          store x.name.length, 6
          lea x.name.data, "helmut"
          ret void
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
        __trashcan__main():
          local x, person
          store x.age, 5
          store x.name.length, 6
          lea x.name.data, "helmut"
          local y, string
          store y.length, x.name.length
          store y.data, x.name.data
          ret void
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
      __trashcan__main():
        local %0, bool
        eq %0, 1, 1
        jne .LB1, %0, true
        jne .LB0, true, false
      .LB1:
        local %1, bool
        eq %1, 2, 2
        jne .LB2, %1, true
        local x, int
        store x, 5
        jne .LB0, true, false
      .LB2:
      .LB0:
        local z, int
        store z, 5
        ret void
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
        __trashcan__main():
        .LB0:
          local %0, bool
          eq %0, 1, 2
          jne .LB1, %0, true
          local x, int
          store x, 5
          jne .LB0, true, false
        .LB1:
          ret void
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
        __trashcan__main():
          local x, [int; 2]
          store x.length, 2
          store x.0, 420
          store x.1, 69
          ret void
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    fn assert_bytecode_matches(expected: &str, bc: &crate::bytecode::Bytecode) {
        let expected_lines: Vec<&str> = expected.trim().lines().map(|l| l.trim()).collect();
        let bc_s = bc.to_string();
        let bc_lines: Vec<&str> = bc_s.trim().lines().map(|l| l.trim()).collect();

        println!("{}", bc);

        assert_eq!(expected_lines.len(), bc_lines.len());

        for i in 0..expected_lines.len() {
            assert_eq!(expected_lines[i], bc_lines[i]);
        }
    }
}
