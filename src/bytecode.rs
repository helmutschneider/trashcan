use std::collections::HashMap;
use std::fmt::Write;

use crate::ast;
use crate::ast::Expression;
use crate::ast::Identifier;
use crate::ast::Statement;
use crate::ast::StructMemberInitializer;
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
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: Type,
    pub offset: VariableOffset,
}

impl Variable {
    fn subsegment_for_member(&self, name: &str) -> Variable {
        let member = self.type_.find_struct_member(name)
            .expect(&format!("struct member '{}' does not exist", name));

        let offset = match &self.offset {
            VariableOffset::Stack(s) => VariableOffset::Parent(Box::new(self.clone()), member.offset),
            VariableOffset::Parent(p, o) => VariableOffset::Parent(p.clone(), o.add(member.offset)),
        };

        return Self {
            name: format!("{}.{}", self.name, name),
            type_: member.type_,
            offset: offset,
        };
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
    Int(i64),
    String(String),
    Variable(Variable),
}

impl Argument {
    pub fn get_type(&self) -> Type {
        return match self {
            Self::Void => Type::Void,
            Self::Int(_) => Type::Int,
            Self::String(_) => panic!("bad! got string argument."),
            Self::Variable(v) => v.type_.clone(),
        };
    }

    pub fn is_pointer(&self) -> bool {
        return self.get_type().is_pointer();
    }

    pub fn is_struct(&self) -> bool {
        return self.get_type().is_struct();
    }
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Void => f.write_str("void"),
            Self::Int(x) => x.fmt(f),
            Self::String(s) => f.write_str(&format!("\"{}\"", s)),
            Self::Variable(v) => {
                v.name.fmt(f)
            }
        };
    }
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
        let last = self.data.last();
        let next_offset = self.data.last()
            .map(|x| {
                let offset = match x.offset {
                    VariableOffset::Stack(x) => x,
                    _ => panic!("members variables should not be visible on the stack."),
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
        bc.instructions.push(Instruction::Function(ENTRYPOINT_NAME.to_string(), Vec::new()));
        for stmt in &main_statements {
            bc.compile_statement(stmt, &mut main_stack);
        }
        bc.instructions.push(Instruction::Return(Argument::Void));

        return Ok(bc);
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
            ast::Expression::IntegerLiteral(x) => Argument::Int(x.value),
            ast::Expression::StringLiteral(s) => {
                let type_str = self.typer.try_find_symbol("string", SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();
                let var_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_str, stack);

                let arg_len = Argument::Int(s.value.len() as i64);
                let arg_len_seg = var_ref.subsegment_for_member("length");
                let arg_data = Argument::String(s.value.clone());
                let arg_data_seg = var_ref.subsegment_for_member("data");

                self.instructions.push(Instruction::Store(arg_len_seg, arg_len));
                self.instructions.push(Instruction::AddressOf(arg_data_seg, arg_data));

                Argument::Variable(var_ref)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = self.typer.try_infer_expression_type(expr).unwrap();
                let dest_ref: Variable;
                
                let instr = match bin_expr.operator.kind {
                    TokenKind::Plus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        Instruction::Add(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Minus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        Instruction::Sub(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Star => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        Instruction::Mul(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Slash => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        Instruction::Div(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::DoubleEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        Instruction::IsEqual(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::NotEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);
                        let temp = stack.push_temporary(&Type::Bool);
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);
                        self.instructions.push(Instruction::IsEqual(temp.clone(), lhs, rhs));
                        Instruction::IsEqual(dest_ref.clone(), Argument::Variable(temp), Argument::Int(0))
                    }
                    TokenKind::Equals => {
                        let lhs = self.compile_expression(&bin_expr.left, None, stack);
                        let rhs = self.compile_expression(&bin_expr.right, None, stack);

                        let lhs_var = match lhs {
                            Argument::Variable(x) => x,
                            _ => panic!("left hand side is not a variable.")
                        };
                        dest_ref = lhs_var.clone();

                        Instruction::Store(lhs_var, rhs)
                    }
                    _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
                };
                self.instructions.push(instr);

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
                self.instructions.push(Instruction::Call(
                    dest_ref.clone(),
                    call.name_token.value.clone(),
                    args,
                ));
                Argument::Variable(dest_ref)
            }
            ast::Expression::Void => Argument::Void,
            ast::Expression::StructInitializer(s) => {
                let type_ = self.typer.try_find_symbol(&s.name_token.value, SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();

                // make sure to initialize the struct as declared by the type
                // and not in the order of initialized arguments.
                let members_by_name: HashMap<String, StructMemberInitializer> = s.members.iter()
                    .map(|m| (m.field_name_token.value.clone(), m.clone()))
                    .collect();

                let dest_var = self.maybe_add_temp_variable(maybe_dest_var, &type_, stack);

                if let Type::Struct(_, members) = &type_ {
                    for m in members {
                        let member_dest = dest_var.subsegment_for_member(&m.name);
                        let value = &members_by_name.get(&m.name).unwrap().value;
                        let arg = self.compile_expression(&value, Some(&member_dest), stack);

                        if expression_needs_explicit_copy(value) {
                            self.compile_stack_copy(&member_dest, &arg, stack);
                        }
                    }
                }

                Argument::Variable(dest_var)
            }
            ast::Expression::MemberAccess(prop_access) => {
                let mut iter = Some(prop_access);
                let mut root_left: Option<&Identifier> = None;
                let mut path_to_prop: Vec<&str> = Vec::new();

                while let Some(x) = iter {
                    path_to_prop.insert(0, &x.right.name);

                    match x.left.as_ref() {
                        Expression::MemberAccess(next) => {
                            iter = Some(next);
                        }
                        Expression::Identifier(ident) => {
                            root_left = Some(ident);
                            iter = None;
                        }
                        _ => {
                            panic!("expected and identifier or member access.");
                        }
                    }
                }

                let mut dest_seg = stack.find(&root_left.unwrap().name);

                for x in path_to_prop {
                    dest_seg = dest_seg.subsegment_for_member(x);
                }

                Argument::Variable(dest_seg)
            }
            ast::Expression::Pointer(ptr) => {
                let inner_type = self.typer.try_infer_expression_type(&ptr.expr)
                    .unwrap();
                let ptr_type = Type::Pointer(Box::new(inner_type));
                let source = self.compile_expression(&ptr.expr, None, stack);
                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &ptr_type, stack);
                self.instructions.push(Instruction::AddressOf(dest_ref.clone(), source));

                Argument::Variable(dest_ref)
            }
            ast::Expression::BooleanLiteral(b) => {
                Argument::Int(b.value.into())
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
                self.instructions.push(Instruction::Local(var.clone()));
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
            self.instructions.push(Instruction::Return(Argument::Void));
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
                self.instructions.push(Instruction::Local(var_ref.clone()));

                let init_arg = self.compile_expression(&var.initializer, Some(&var_ref), stack);

                // literals and identifier expressions don't emit any stack
                // variables, so we need an implicit copy here.
                if expression_needs_explicit_copy(&var.initializer) {
                    self.compile_stack_copy(&var_ref, &init_arg, stack);
                }
            }
            ast::Statement::Return(ret) => {
                let ret_reg = self.compile_expression(&ret.expr, None, stack);
                self.instructions.push(Instruction::Return(ret_reg));
            }
            ast::Statement::If(if_stmt) => {
                let label_after_last_block = self.add_label();

                self.compile_if(&if_stmt, &label_after_last_block, stack);
            }
            ast::Statement::While(while_) => {
                let label_before_condition = self.add_label();
                let label_after_block = self.add_label();

                self.instructions.push(Instruction::Label(label_before_condition.clone()));
                let cmp_arg = self.compile_expression(&while_.condition, None, stack);
                self.instructions.push(Instruction::JumpNotEqual(label_after_block.clone(), cmp_arg, Argument::Int(1)));
                self.compile_block(while_.block.as_block(), stack);
                self.instructions.push(Instruction::JumpNotEqual(label_before_condition, Argument::Int(1), Argument::Int(0)));
                self.instructions.push(Instruction::Label(label_after_block));
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
        self.instructions.push(Instruction::JumpNotEqual(
            label_after_block.clone(),
            condition,
            Argument::Int(1),
        ));
        self.compile_block(if_stmt.block.as_block(), stack);
        self.instructions.push(Instruction::JumpNotEqual(label_after_last_block.to_string(), Argument::Int(1), Argument::Int(0)));
        self.instructions.push(Instruction::Label(label_after_block));

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
            self.instructions.push(Instruction::Label(label_after_last_block.to_string()));
        }
    }

    fn compile_stack_copy(&mut self, dest_var: &Variable, source: &Argument, stack: &mut Stack) {
        assert_eq!(dest_var.type_, source.get_type());

        let type_ = source.get_type();

        if let Type::Struct(_, members) = type_ {
            for m in members {
                let member_seg = dest_var.subsegment_for_member(&m.name);
                let source_seg = match source {
                    Argument::Variable(v) => v.subsegment_for_member(&m.name),
                    _ => panic!("bad!")
                };
                self.compile_stack_copy(&member_seg, &Argument::Variable(source_seg), stack);
            }
        } else {
            self.instructions.push(Instruction::Store(dest_var.clone(), source.clone()));
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
            jne .LB1, x, 1
            local y, int
            store y, 42
            jne .LB0, 1, 0
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
        jne .LB1, %0, 1
        jne .LB0, 1, 0
      .LB1:
        local %1, bool
        eq %1, 2, 2
        jne .LB2, %1, 1
        local x, int
        store x, 5
        jne .LB0, 1, 0
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
          jne .LB1, %0, 1
          local x, int
          store x, 5
          jne .LB0, 1, 0
        .LB1:
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
