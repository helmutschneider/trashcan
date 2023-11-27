use std::collections::HashMap;
use std::fmt::Write;

use crate::ast;
use crate::ast::Expression;
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
pub struct Variable {
    pub name: String,
    pub type_: Type,
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.name.fmt(f);
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Void,
    Int(i64),
    String(String),
    Variable(Variable, Offset),
}

impl Argument {
    pub fn get_type(&self) -> Type {
        return match self {
            Self::Variable(v, _) => v.type_.clone(),
            _ => Type::Void,
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
            Self::Variable(v, offset) => {
                if offset.to_i64() != 0 {
                    f.write_str(&format!("[{}{}]", v, offset))
                } else {
                    v.fmt(f)
                }
            }
        };
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Local(Variable),
    Label(String),
    Store(Variable, Offset, Argument),
    AddressOf(Variable, Offset, Argument),
    Return(Argument),
    Add(Variable, Argument, Argument),
    Sub(Variable, Argument, Argument),
    Mul(Variable, Argument, Argument),
    Div(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
    IsEqual(Variable, Argument, Argument),
    JumpNotEqual(String, Argument, Argument),
}

// pointers are invisible to the end-user and we don't
// show the '&' character while formatting types. in
// the bytecode it can be useful for debugging, though.
//   -johan, 2023-11-24
fn type_to_string(type_: &Type) -> String {
    if let Type::Pointer(inner) = type_ {
        return format!("&{}", type_to_string(inner));
    }
    return type_.to_string();
}

fn variable_with_offset_to_string(var: &Variable, offset: Offset) -> String {
    if let Offset::None = offset {
        return var.to_string();
    }
    return format!("[{}{}]", var, offset);
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(name, args) => {
                let args_s = args
                    .iter()
                    .map(|v| format!("{}: {}", v.name, type_to_string(&v.type_)))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", name, args_s)
            }
            Self::Local(var) => {
                format!("  local {}, {}", var.name, type_to_string(&var.type_))
            }
            Self::Label(name) => {
                format!("{}:", name)
            }
            Self::Store(dest_var, dest_offset, source) => {
                let dest = variable_with_offset_to_string(dest_var, *dest_offset);
                let source = match source {
                    Argument::Variable(x, source_offset) => variable_with_offset_to_string(x, *source_offset),
                    _ => source.to_string(),
                };
                format!("  store {}, {}", dest, source)
            }
            Self::AddressOf(dest_var, dest_offset, source) => {
                let dest = variable_with_offset_to_string(dest_var, *dest_offset);
                let source = match source {
                    Argument::Variable(x, source_offset) => variable_with_offset_to_string(x, *source_offset),
                    _ => source.to_string(),
                };
                format!("  lea {}, {}", dest, source)
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

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: i64,
    pub temporaries: i64,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = typer::Typer::from_code(code)?;
        typer.check()?;

        let mut bc = Self {
            instructions: Vec::new(),
            labels: 0,
            temporaries: 0,
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
                bc.compile_statement(&typer, stmt);
            }
        }

        let prev_temp = bc.temporaries;

        bc.temporaries = 0;
        bc.instructions.push(Instruction::Function(ENTRYPOINT_NAME.to_string(), Vec::new()));
        for stmt in &main_statements {
            bc.compile_statement(&typer, stmt);
        }
        bc.instructions.push(Instruction::Return(Argument::Void));
        bc.temporaries = prev_temp;

        return Ok(bc);
    }

    fn add_temporary(&mut self, type_: &Type) -> Variable {
        let temp = Variable {
            name: format!("%{}", self.temporaries),
            type_: type_.clone(),
        };
        self.instructions.push(Instruction::Local(temp.clone()));
        self.temporaries += 1;
        return temp;
    }

    fn add_label(&mut self) -> String {
        let label = format!(".LB{}", self.labels);
        self.labels += 1;
        return label;
    }

    fn compile_expression(
        &mut self,
        typer: &typer::Typer,
        expr: &ast::Expression,
        maybe_dest_var: Option<&Variable>,
    ) -> Argument {
        let value = match expr {
            ast::Expression::IntegerLiteral(x) => Argument::Int(x.value),
            ast::Expression::StringLiteral(s) => {
                let type_str = typer.try_find_symbol("string", SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();
                let var_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_str);

                let arg_len = Argument::Int(s.value.len() as i64);
                let arg_data = Argument::String(s.value.clone());

                self.compile_struct_initializer(&var_ref.clone(), type_str, &[arg_len, arg_data]);

                Argument::Variable(var_ref, Offset::None)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = typer.try_infer_expression_type(expr).unwrap();
                let dest_ref: Variable;
                
                let instr = match bin_expr.operator.kind {
                    TokenKind::Plus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        Instruction::Add(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Minus => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        Instruction::Sub(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Star => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        Instruction::Mul(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::Slash => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        Instruction::Div(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::DoubleEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        Instruction::IsEqual(dest_ref.clone(), lhs, rhs)
                    }
                    TokenKind::NotEquals => {
                        dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                        let temp = self.add_temporary(&Type::Bool);
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);
                        self.instructions.push(Instruction::IsEqual(temp.clone(), lhs, rhs));
                        Instruction::IsEqual(dest_ref.clone(), Argument::Variable(temp, Offset::None), Argument::Int(0))
                    }
                    TokenKind::Equals => {
                        let lhs = self.compile_expression(typer, &bin_expr.left, None);
                        let rhs = self.compile_expression(typer, &bin_expr.right, None);

                        let lhs_var = match lhs {
                            Argument::Variable(x, _) => x,
                            _ => panic!("left hand side is not a variable.")
                        };
                        dest_ref = lhs_var.clone();

                        Instruction::Store(lhs_var, Offset::None, rhs)
                    }
                    _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
                };
                self.instructions.push(instr);

                Argument::Variable(dest_ref, Offset::None)
            }
            ast::Expression::Identifier(ident) => {
                let var_sym = typer
                    .try_find_symbol(&ident.name, SymbolKind::Local, ident.parent)
                    .unwrap();
                let var_type = var_sym.type_;
                let var = Variable {
                    name: ident.name.clone(),
                    type_: var_type,
                };
                Argument::Variable(var, Offset::None)
            }
            ast::Expression::FunctionCall(call) => {
                let fx_sym = typer
                    .try_find_symbol(&call.name_token.value, SymbolKind::Function, call.parent)
                    .unwrap();

                let (arg_types, ret_type) = match fx_sym.type_ {
                    Type::Function(x, y) => (x, y),
                    _ => panic!(),
                };

                let mut args: Vec<Argument> = Vec::new();

                for k in 0..arg_types.len() {
                    let given_arg = self.compile_expression(typer, &call.arguments[k], None);
                    args.push(given_arg);
                }

                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &ret_type);
                self.instructions.push(Instruction::Call(
                    dest_ref.clone(),
                    call.name_token.value.clone(),
                    args,
                ));
                Argument::Variable(dest_ref, Offset::None)
            }
            ast::Expression::Void => Argument::Void,
            ast::Expression::StructInitializer(s) => {
                let type_ = typer.try_find_symbol(&s.name_token.value, SymbolKind::Type, s.parent)
                    .map(|s| s.type_)
                    .unwrap();
                let mut struct_args: Vec<Argument> = Vec::new();

                // make sure to initialize the struct as declared by the type
                // and not in the order of initialized arguments.
                let members_by_name: HashMap<String, StructMemberInitializer> = s.members.iter()
                    .map(|m| (m.field_name_token.value.clone(), m.clone()))
                    .collect();

                if let Type::Struct(_, members) = &type_ {
                    for m in members {
                        let value = &members_by_name.get(&m.name).unwrap().value;
                        let arg = self.compile_expression(typer, &value, None);
                        struct_args.push(arg);
                    }
                }

                let dest_var = self.maybe_add_temp_variable(maybe_dest_var, &type_);
                self.compile_struct_initializer(&dest_var, type_, &struct_args);

                Argument::Variable(dest_var, Offset::None)
            }
            ast::Expression::MemberAccess(prop_access) => {
                let source = self.compile_expression(typer, &prop_access.left, None);
                let source_var = match &source {
                    Argument::Variable(x, _) => x,
                    _ => panic!("expected a variable.")
                };
                let source_type = &source_var.type_;
                let member_name = &prop_access.right.name;
                let member = source_type.find_struct_member(member_name)
                    .unwrap();
                let member_offset = source_type.find_struct_member_offset(member_name)
                    .unwrap();
                let member_type = if source_type.is_pointer() {
                    Type::Pointer(Box::new(member.type_))
                } else {
                    member.type_
                };

                // TODO: maybe add a deref instruction here, if the right hand side is a pointer.

                let dest_var = self.maybe_add_temp_variable(maybe_dest_var, &member_type);
                let mut offset = Offset::Positive(0);

                for k in member_type.memory_layout() {
                    let source_with_offset = match &source {
                        Argument::Variable(x, o) => Argument::Variable(x.clone(), member_offset.add(offset)),
                        _ => source.clone(),
                    };
                    self.instructions.push(Instruction::Store(
                        dest_var.clone(),
                        offset,
                        source_with_offset
                    ));
                    offset = offset.add(k);
                }

                Argument::Variable(dest_var, Offset::None)
            }
            ast::Expression::Pointer(ptr) => {
                let inner_type = typer.try_infer_expression_type(&ptr.expr)
                    .unwrap();
                let ptr_type = Type::Pointer(Box::new(inner_type));
                let source = self.compile_expression(typer, &ptr.expr, None);
                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, &ptr_type);
                self.instructions.push(Instruction::AddressOf(dest_ref.clone(), Offset::None, source));

                Argument::Variable(dest_ref, Offset::None)
            }
            ast::Expression::BooleanLiteral(b) => {
                Argument::Int(b.value.into())
            }
        };
        return value;
    }

    fn maybe_add_temp_variable(&mut self, dest_var: Option<&Variable>, type_: &Type) -> Variable {
        return match dest_var {
            Some(v) => {
                assert_eq!(&v.type_, type_);
                v.clone()
            }
            None => self.add_temporary(type_),
        };
    }

    fn compile_function(&mut self, typer: &typer::Typer, fx: &ast::Function) {
        let temps_prev = self.temporaries;
        self.temporaries = 0;

        let arg_vars: Vec<Variable> = fx
            .arguments
            .iter()
            .map(|fx_arg| {
                let fx_arg_sym = typer
                    .try_find_symbol(&fx_arg.name_token.value, SymbolKind::Local, fx.body.id())
                    .unwrap();

                let fx_arg_type = fx_arg_sym.type_;

                Variable {
                    name: fx_arg.name_token.value.clone(),
                    type_: fx_arg_type,
                }
            })
            .collect();

        self.instructions
            .push(Instruction::Function(fx.name_token.value.clone(), arg_vars));

        self.compile_block(typer, fx.body.as_block());

        // add an implicit return statement if the function doesn't have one.
        if !matches!(self.instructions.last(), Some(Instruction::Return(_))) {
            self.instructions.push(Instruction::Return(Argument::Void));
        }

        self.temporaries = temps_prev;
    }

    fn compile_block(&mut self, typer: &typer::Typer, block: &ast::Block) {
        for stmt in &block.statements {
            self.compile_statement(typer, stmt);
        }
    }

    fn compile_statement(&mut self, typer: &typer::Typer, stmt: &ast::Statement) {
        match stmt {
            ast::Statement::Function(fx) => {
                self.compile_function(typer, fx);
            }
            ast::Statement::Block(block) => {
                self.compile_block(typer, block);
            }
            ast::Statement::Variable(var) => {
                let var_sym = typer
                    .try_find_symbol(&var.name_token.value, SymbolKind::Local, var.parent)
                    .unwrap();
                let var_type = var_sym.type_;
                let var_ref = Variable {
                    name: var_sym.name,
                    type_: var_type,
                };

                self.instructions.push(Instruction::Local(var_ref.clone()));

                let init_arg = self.compile_expression(typer, &var.initializer, Some(&var_ref));

                // literals and identifier expressions don't emit any stack
                // variables, so we need an implicit copy here.
                if matches!(
                    var.initializer,
                    ast::Expression::IntegerLiteral(_) | ast::Expression::Identifier(_)
                ) {
                    self.instructions.push(Instruction::Store(
                        var_ref,
                        Offset::None,
                        init_arg,
                    ));
                }
            }
            ast::Statement::Return(ret) => {
                let ret_reg = self.compile_expression(typer, &ret.expr, None);
                self.instructions.push(Instruction::Return(ret_reg));
            }
            ast::Statement::If(if_stmt) => {
                let label_after_last_block = self.add_label();

                self.compile_if(typer, &if_stmt, &label_after_last_block);
            }
            ast::Statement::While(while_) => {
                let label_before_condition = self.add_label();
                let label_after_block = self.add_label();

                self.instructions.push(Instruction::Label(label_before_condition.clone()));
                let cmp_arg = self.compile_expression(typer, &while_.condition, None);
                self.instructions.push(Instruction::JumpNotEqual(label_after_block.clone(), cmp_arg, Argument::Int(1)));
                self.compile_block(typer, while_.block.as_block());
                self.instructions.push(Instruction::JumpNotEqual(label_before_condition, Argument::Int(1), Argument::Int(0)));
                self.instructions.push(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                self.compile_expression(typer, &expr.expr, None);
            }
            ast::Statement::Type(_) => {
                // do nothing. type declarations aren't represented by specific
                // instructions in the bytecode.
            }
        }
    }

    fn compile_struct_initializer(
        &mut self,
        dest_var: &Variable,
        type_: Type,
        arguments: &[Argument],
    ) {
        let members = match type_ {
            Type::Struct(_, x) => x,
            _ => panic!("type '{}' is not a struct", type_),
        };

        assert_eq!(members.len(), arguments.len());

        let mut offset = Offset::Positive(0);

        for k in 0..members.len() {
            let member = &members[k];
            let arg = &arguments[k];
            let member_type = &member.type_;

            if member_type.is_pointer() && !arg.is_pointer() {
                // this will probably only happen for constant strings, for now.
                // you can't create pointers manually (yet) but structs passed
                // to functions are implicitly pointers.
                //   -johan, 2023-11-25
                self.instructions.push(Instruction::AddressOf(
                    dest_var.clone(),
                    offset,
                    arg.clone()
                ));
            } else {
                let memory_layout = match arg {
                    Argument::Variable(x, _) => x.type_.memory_layout(),
                    _ => vec![8],
                };
                let mut current_offset = Offset::Positive(0);

                for k in memory_layout {
                    let arg_with_offset = match &arg {
                        Argument::Variable(x, o) => Argument::Variable(x.clone(), o.add(current_offset)),
                        _ => arg.clone(),
                    };
                    let inner_dest_offset = offset.add(current_offset);
                    self.instructions.push(Instruction::Store(
                        dest_var.clone(),
                        inner_dest_offset,
                        arg_with_offset
                    ));
                    current_offset = current_offset.add(k);
                }
            }

            offset = offset.add(member_type.size());
        }
    }

    fn compile_if(&mut self, typer: &Typer, if_stmt: &ast::If, label_after_last_block: &str) {
        let condition = self.compile_expression(typer, &if_stmt.condition, None);
        let label_after_block = self.add_label();
        self.instructions.push(Instruction::JumpNotEqual(
            label_after_block.clone(),
            condition,
            Argument::Int(1),
        ));
        self.compile_block(typer, if_stmt.block.as_block());
        self.instructions.push(Instruction::JumpNotEqual(label_after_last_block.to_string(), Argument::Int(1), Argument::Int(0)));
        self.instructions.push(Instruction::Label(label_after_block));

        let else_ = if_stmt.else_.as_ref();
        let mut is_last_block = false;

        if let Some(next) = else_ {
            if let Statement::If(next_if) = next.as_ref() {
                self.compile_if(typer, next_if, label_after_last_block);
            }

            if let Statement::Block(else_) = next.as_ref() {
                self.compile_block(typer, else_);
                is_last_block = true;
            }
        } else {
            is_last_block = true;
        }

        if is_last_block {
            self.instructions.push(Instruction::Label(label_after_last_block.to_string()));
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
          store [x + 0], 5
          lea [x + 8], "hello"
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
          store [%0 + 0], 4
          lea [%0 + 8], "yee!"
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
    fn should_pass_literal_struct_argument_by_reference() {
        let code = r###"
            fun takes_str(x: &string): void {}
            var x: string = "yee!";
            takes_str(&x);
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        takes_str(x: &string):
          ret void
        __trashcan__main():
          local x, string
          store [x + 0], 4
          lea [x + 8], "yee!"
          local %0, &string
          lea %0, x
          local %1, void
          call %1, takes_str(%0)
          ret void
        "###;

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
          store [x + 0], 6
          store [x + 8], 5
          ret void
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_layout_struct_with_string() {
        let code = r###"
        type person = {
            name: string,
            age: int,
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
          local %0, string
          store [%0 + 0], 6
          lea [%0 + 8], "helmut"
          store [x + 0], [%0 + 0]
          store [x + 8], [%0 + 8]
          store [x + 16], 5
          ret void
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_pass_struct_by_reference() {
        let code = r###"
        type person = {
            name: string,
            age: int,
        };
        fun thing(x: &person): void {}
        var x = person {
            name: "helmut",
            age: 5,
        };
        thing(&x);
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        thing(x: &person):
          ret void
        __trashcan__main():
          local x, person
          local %0, string
          store [%0 + 0], 6
          lea [%0 + 8], "helmut"
          store [x + 0], [%0 + 0]
          store [x + 8], [%0 + 8]
          store [x + 16], 5
          local %1, &person
          lea %1, x
          local %2, void
          call %2, thing(%1)
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
          local %0, string
          store [%0 + 0], 6
          lea [%0 + 8], "helmut"
          store [x + 0], 5
          store [x + 8], [%0 + 0]
          store [x + 16], [%0 + 8]
          ret void
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_emit_copy_instructions_for_member_access() {
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
          local %0, string
          store [%0 + 0], 6
          lea [%0 + 8], "helmut"
          store [x + 0], 5
          store [x + 8], [%0 + 0]
          store [x + 16], [%0 + 8]
          local y, string
          store [y + 0], [x + 8]
          store [y + 8], [x + 16]
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
