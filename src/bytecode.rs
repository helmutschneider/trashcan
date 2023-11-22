use crate::ast;
use crate::ast::ASTLike;
use crate::ast::Expression;
use crate::typer::SymbolKind;
use crate::typer::Type;
use crate::{tokenizer::TokenKind, typer, util::Error};

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstId(pub i64);

impl std::fmt::Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!(".LC{}", self.0));
    }
}

#[derive(Debug, Clone)]
pub struct Constant {
    pub id: ConstId,
    pub value: ConstantValue,
}

#[derive(Debug, Clone)]
pub enum ConstantValue {
    Integer(i64),
    String(String),
}

impl std::fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            ConstantValue::Integer(x) => x.fmt(f),
            ConstantValue::String(s) => f.write_str(&format!("\"{}\"", s)),
        };
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    Void,
    Integer(i64),
    Constant(ConstId),
    Variable(Variable),
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Void => f.write_str("void"),
            Self::Integer(x) => x.fmt(f),
            Self::Constant(c) => c.fmt(f),
            Self::Variable(r) => r.fmt(f),
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PositiveOffset(pub i64);

#[derive(Debug, Clone, Copy)]
pub struct NegativeOffset(pub i64);

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Local(Variable),
    Constant(Constant),
    Label(String),
    Store(Variable, PositiveOffset, Argument),
    AddressOf(Variable, PositiveOffset, Argument),
    Return(Argument),
    Add(Variable, Argument, Argument),
    Sub(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
    IsEqual(Variable, Argument, Argument),
    JumpNotEqual(String, Argument, Argument),
    Noop,
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function(name, args) => {
                let args_s = args
                    .iter()
                    .map(|v| v.name.clone())
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({}):", name, args_s)
            }
            Self::Local(var) => {
                format!("  local {}, {}", var.name, var.type_)
            }
            Self::Constant(c) => {
                format!("  const {}, {}", c.id, c.value)
            }
            Self::Label(name) => {
                format!("{}:", name)
            }
            Self::Store(dest_var, offset, arg) => {
                if offset.0 != 0 {
                    let op = if offset.0 > 0 { "+" } else { "-" };
                    format!("  store [{}{}{}], {}", dest_var, op, offset.0, arg)
                } else {
                    format!("  store {}, {}", dest_var, arg)
                }
            }
            Self::AddressOf(dest_var, offset, arg) => {
                if offset.0 != 0 {
                    let op = if offset.0 > 0 { "+" } else { "-" };
                    format!("  lea [{}{}{}], {}", dest_var, op, offset.0, arg)
                } else {
                    format!("  lea {}, {}", dest_var, arg)
                }
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
            Self::Noop => format!("  noop"),
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub constants: i64,
    pub instructions: Vec<Instruction>,
    pub labels: i64,
    pub temporaries: i64,
}

impl Bytecode {
    pub fn from_code(code: &str) -> Result<Self, Error> {
        let typer = typer::Typer::from_code(code)?;
        typer.check()?;

        let mut bc = Self {
            constants: 0,
            instructions: Vec::new(),
            labels: 0,
            temporaries: 0,
        };

        bc.compile_block(&typer, typer.ast.body());

        return Ok(bc);
    }

    fn add_temporary(&mut self, type_: Type) -> Variable {
        let temp = Variable {
            name: format!("%{}", self.temporaries),
            type_: type_,
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

    fn add_constant(&mut self, value: ConstantValue) -> ConstId {
        let id = ConstId(self.constants);
        self.constants += 1;
        let c = Constant {
            id: id,
            value: value,
        };
        self.instructions.push(Instruction::Constant(c));
        return id;
    }

    fn compile_expression(
        &mut self,
        typer: &typer::Typer,
        expr: &ast::Expression,
        maybe_dest_var: Option<&Variable>,
    ) -> Argument {
        let types = &typer.types;
        let value = match expr {
            ast::Expression::IntegerLiteral(x) => {
                Argument::Integer(x.value)
            }
            ast::Expression::StringLiteral(s) => {
                // TODO: this code belongs in some generic struct-layout method.
                let type_str = types.get_type_by_name("string").unwrap();
                let var_ref = self.maybe_add_temp_variable(maybe_dest_var, type_str);

                let const_len = Argument::Constant(
                    self.add_constant(ConstantValue::Integer(s.value.len() as i64)),
                );
                let const_data =
                    Argument::Constant(self.add_constant(ConstantValue::String(s.value.clone())));
                self.instructions
                    .push(Instruction::Store(var_ref.clone(), PositiveOffset(0), const_len));
                self.instructions.push(Instruction::AddressOf(
                    var_ref.clone(),
                    PositiveOffset(8),
                    const_data,
                ));

                Argument::Variable(var_ref)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let type_ = typer.try_infer_expression_type(expr).unwrap();
                let lhs = self.compile_expression(typer, &bin_expr.left, None);
                let rhs = self.compile_expression(typer, &bin_expr.right, None);
                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, type_);
                let instr = match bin_expr.operator.kind {
                    TokenKind::Plus => Instruction::Add(dest_ref.clone(), lhs, rhs),
                    TokenKind::Minus => Instruction::Sub(dest_ref.clone(), lhs, rhs),
                    TokenKind::DoubleEquals => Instruction::IsEqual(dest_ref.clone(), lhs, rhs),
                    _ => panic!("Unknown operator: {:?}", bin_expr.operator.kind),
                };
                self.instructions.push(instr);

                Argument::Variable(dest_ref)
            }
            ast::Expression::Identifier(ident) => {
                let var_sym = typer
                    .try_find_symbol(&ident.name, SymbolKind::Local, ident.parent)
                    .unwrap();
                let var_type = typer.try_resolve_symbol_type(&var_sym).unwrap();
                let var = Variable {
                    name: ident.name.clone(),
                    type_: var_type,
                };
                Argument::Variable(var)
            }
            ast::Expression::FunctionCall(call) => {
                let fx_sym = typer
                    .try_find_symbol(&call.name_token.value, SymbolKind::Function, call.parent)
                    .unwrap();

                let (arg_types, ret_type) = match fx_sym.type_.unwrap() {
                    Type::Function(x, y) => (x, y),
                    _ => panic!(),
                };

                let mut args: Vec<Argument> = Vec::new();

                for k in 0..arg_types.len() {
                    let arg_type = &arg_types[k];
                    let given_arg = self.compile_expression(typer, &call.arguments[k], None);
                    let given_arg_maybe_by_reference: Argument;

                    if arg_type.is_struct() {
                        let temp = self.add_temporary(Type::Pointer(Box::new(arg_type.clone())));
                        self.instructions.push(Instruction::AddressOf(temp.clone(), PositiveOffset(0), given_arg));
                        given_arg_maybe_by_reference = Argument::Variable(temp);
                    } else {
                        given_arg_maybe_by_reference = given_arg;
                    }

                    args.push(given_arg_maybe_by_reference);
                }

                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, *ret_type);
                self.instructions.push(Instruction::Call(
                    dest_ref.clone(),
                    call.name_token.value.clone(),
                    args,
                ));
                Argument::Variable(dest_ref)
            }
            ast::Expression::Void => Argument::Void,
        };
        return value;
    }

    fn maybe_add_temp_variable(
        &mut self,
        dest_var: Option<&Variable>,
        type_: Type,
    ) -> Variable {
        return match dest_var {
            Some(v) => {
                assert_eq!(v.type_, type_);
                v.clone()
            },
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
                    .try_find_symbol(&fx_arg.name_token.value, SymbolKind::Local, fx.body)
                    .unwrap();
                let fx_arg_type = fx_arg_sym.type_.unwrap();

                Variable {
                    name: fx_arg.name_token.value.clone(),
                    type_: fx_arg_type,
                }
            })
            .collect();

        self.instructions
            .push(Instruction::Function(fx.name_token.value.clone(), arg_vars));

        let fx_body = typer.ast.get_block(fx.body);

        self.compile_block(typer, fx_body);

        // add an implicit return statement if the function doesn't have one.
        if !matches!(self.instructions.last(), Some(Instruction::Return(_))) {
            self.instructions
                .push(Instruction::Return(Argument::Void));
        }

        self.temporaries = temps_prev;
    }

    fn compile_block(&mut self, typer: &typer::Typer, block: &ast::Block) {
        for stmt in &block.statements {
            let stmt = typer.ast.get_statement(*stmt);
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
                let var_type = typer.try_resolve_symbol_type(&var_sym);
                let var_ref = Variable {
                    name: var_sym.name,
                    type_: var_type.unwrap(),
                };

                self.instructions.push(Instruction::Local(var_ref.clone()));

                let init_arg = self.compile_expression(typer, &var.initializer, Some(&var_ref));

                // literals and identifier expressions don't emit any stack
                // variables, so we need an implicit copy here.
                if matches!(
                    var.initializer,
                    ast::Expression::IntegerLiteral(_) | ast::Expression::Identifier(_)
                ) {
                    self.instructions.push(Instruction::Store(var_ref, PositiveOffset(0), init_arg));
                }
            }
            ast::Statement::Return(ret) => {
                let ret_reg = self.compile_expression(typer, &ret.expr, None);
                self.instructions.push(Instruction::Return(ret_reg));
            }
            ast::Statement::If(if_stmt) => {
                let label_after_block = self.add_label();
                let condition = self.compile_expression(typer, &if_stmt.condition, None);
                self.instructions.push(Instruction::JumpNotEqual(
                    label_after_block.clone(),
                    condition,
                    Argument::Integer(1),
                ));
                let if_block = typer.ast.get_block(if_stmt.block);
                self.compile_block(typer, if_block);
                self.instructions
                    .push(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                self.compile_expression(typer, expr, None);
            }
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
        local x, int
        store x, 6
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
        local x, int
        store x, 6
        local y, int
        store y, x
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
            local x, int
            add x, 1, 2
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
            add(x, y):
                local %0, int
                add %0, x, y
                ret %0
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
            local x, int
            store x, 2
            local y, bool
            local %0, int
            add %0, x, 1
            eq y, %0, 3
        "###;
        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_if_with_jump() {
        let code = r###"
        var x: bool = 1 == 2;
        if x {
            var y: int = 42;
        }
        var z: int = 3;
    "###;
        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
            local x, bool
            eq x, 1, 2
            jne .LB0, x, 1
            local y, int
            store y, 42
        .LB0:
            local z, int
            store z, 3
        "###;

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
        local x, string
        const .LC0, 5
        const .LC1, "hello"
        store x, .LC0
        lea [x+8], .LC1
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_pass_struct_argument_by_reference() {
        let code = r###"
            fun takes_str(x: string): void {}
            fun main(): void {
                takes_str("yee!");
            }
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        takes_str(x):
          ret void
        main():
          local %0, string
          const .LC0, 4
          const .LC1, "yee!"
          store %0, .LC0
          lea [%0+8], .LC1
          local %1, &string
          lea %1, %0
          local %2, void
          call %2, takes_str(%1)
          ret void
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_pass_literal_struct_argument_by_reference() {
        let code = r###"
            fun takes_str(x: string): void {}
            fun main(): void {
                var x: string = "yee!";
                takes_str(x);
            }
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        takes_str(x):
          ret void
        main():
          local x, string
          const .LC0, 4
          const .LC1, "yee!"
          store x, .LC0
          lea [x+8], .LC1
          local %0, &string
          lea %0, x
          local %1, void
          call %1, takes_str(%0)
          ret void
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    fn assert_bytecode_matches(expected: &str, bc: &crate::bytecode::Bytecode) {
        let expected_lines: Vec<&str> = expected.trim().lines().map(|l| l.trim()).collect();
        let bc_s = bc.to_string();
        let bc_lines: Vec<&str> = bc_s.trim().lines().map(|l| l.trim()).collect();

        assert_eq!(expected_lines.len(), bc_lines.len());

        for i in 0..expected_lines.len() {
            assert_eq!(expected_lines[i], bc_lines[i]);
        }
    }
}
