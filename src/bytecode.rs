use crate::{
    tokenizer::TokenKind, util::Error, typer,
};
use crate::ast;
use crate::ast::ASTLike;
use crate::typer::Type;
use crate::typer::TypeDefinition;
use crate::typer::Typer;
use crate::typer::SymbolKind;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub type_: Rc<Type>,
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.name.fmt(f);
    }
}

#[derive(Debug, Clone)]
pub enum Argument {
    IntegerLiteral(i64),
    StringLiteral(String),
    Variable(Variable),
}

impl std::fmt::Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::IntegerLiteral(c) => c.fmt(f),
            Self::StringLiteral(c) => f.write_str(&format!("\"{}\"", c)),
            Self::Variable(r) => r.fmt(f),
        };
    }
}

#[derive(Debug, Clone)]
pub struct Offset(pub i64);

#[derive(Debug, Clone)]
pub enum Instruction {
    Function(String, Vec<Variable>),
    Local(Variable),
    Label(String),
    Copy(Variable, Argument),
    Store(Variable, Offset, Argument),
    Return(Argument),
    Add(Variable, Argument, Argument),
    Sub(Variable, Argument, Argument),
    Call(Variable, String, Vec<Argument>),
    IsEqual(Variable, Argument, Argument),
    JumpNotEqual(String, Argument, Argument),
    Pointer(Variable, Argument),
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
            Self::Label(name) => {
                format!("{}:", name)
            }
            Self::Copy(dest_var, source) => {
                format!("  {} = copy {}", dest_var, source)
            }
            Self::Store(dest_var, offset, arg) => {
                format!("  store [{}+{}], {}", dest_var, offset.0, arg,)
            }
            Self::Return(value) => {
                format!("  ret {}", value)
            }
            Self::Add(dest_var, x, y) => {
                format!("  {} = add {}, {}", dest_var, x, y)
            }
            Self::Sub(dest_var, x, y) => {
                format!("  {} = sub {}, {}", dest_var, x, y)
            }
            Self::Call(dest_var, name, args) => {
                let arg_s = args
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("  {} = call {}({})", dest_var, name, arg_s)
            }
            Self::IsEqual(dest_var, a, b) => {
                format!("  {} = eq {}, {}", dest_var, a, b)
            }
            Self::JumpNotEqual(to_label, a, b) => {
                format!("  jne {}, {}, {}", to_label, a, b)
            }
            Self::Pointer(dest_var, a) => {
                format!("  {} = pointer {}", dest_var, a)
            }
            Self::Local(var) => {
                format!("  local {}, {}", var.name, var.type_)
            }
            Self::Noop => format!("  noop"),
        };
        return f.write_str(&s);
    }
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub labels: usize,
    pub temporaries: usize,
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
    
        bc.compile_block(&typer, typer.ast.body());
    
        return Ok(bc);
    }

    fn add_temporary(&mut self, type_: Rc<Type>) -> Variable {
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

    fn compile_expression(
        &mut self,
        typer: &typer::Typer,
        expr: &ast::Expression,
        maybe_dest_var: Option<&Variable>
    ) -> Argument {
        let value = match expr {
            ast::Expression::IntegerLiteral(x) => {
                Argument::IntegerLiteral(x.value)
            }
            ast::Expression::StringLiteral(s) => {
                // TODO: this code belongs in some generic struct-layout method.
                let types = &typer.types;
                let var_ref = self.maybe_add_temp_variable(maybe_dest_var, types.string());
                // let temp_data = self.add_temporary(typer.types.pointer_to(typer.types.void()));
    
                // self.instructions.push(Instruction::Local(temp.clone()));
                self.instructions.push(Instruction::Store(var_ref.clone(), Offset(0), Argument::IntegerLiteral(s.value.len() as i64)));

                let data_temp = self.add_temporary(types.pointer_to(types.void()));
                // self.instructions.push(Instruction::lea)

                self.instructions.push(Instruction::Store(var_ref.clone(), Offset(8), Argument::StringLiteral(s.value.clone())));
                // self.instructions.push(Instruction::Store(temp, Offset(0), Argument::IntegerLiteral(s.value.len() as i64)));

                // self.instructions.push(Instruction::Copy(temp_len.clone(), Argument::IntegerLiteral(s.value.len() as i64)));
                // self.instructions.push(Instruction::Pointer(temp_data.clone(), Argument::StringLiteral(s.value.clone())));
    
                let mut struct_args: Vec<Variable> = Vec::new();
                // struct_args.push(temp_len);
                // struct_args.push(temp_data);
    
                Argument::Variable(var_ref)
                // Argument::Struct(struct_args)
            }
            ast::Expression::BinaryExpr(bin_expr) => {
                let lhs = self.compile_expression(typer, &bin_expr.left, None);
                let rhs = self.compile_expression(typer, &bin_expr.right, None);
                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, typer.types.bool());
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
                let var_sym = typer.try_resolve_symbol(&ident.name, SymbolKind::Local, ident.parent)
                    .unwrap();
                let var_type = typer.try_resolve_symbol_type(&var_sym).unwrap();
                let var = Variable {
                    name: ident.name.clone(),
                    type_: var_type,
                };
                Argument::Variable(var)
            }
            ast::Expression::FunctionCall(call) => {
                let args: Vec<Argument> = call
                    .arguments
                    .iter()
                    .map(|x| self.compile_expression(typer, x, None))
                    .collect();
                let fx_sym = typer.try_resolve_symbol(&call.name_token.value, SymbolKind::Function, call.parent)
                    .unwrap();
                let fx_type = fx_sym.type_.unwrap();
                let ret_type = match &fx_type.definition {
                    TypeDefinition::Function(_, x) => x,
                    _ => panic!(),
                };
                let dest_ref = self.maybe_add_temp_variable(maybe_dest_var, Rc::clone(ret_type));
                self.instructions.push(Instruction::Call(
                    dest_ref.clone(),
                    call.name_token.value.clone(),
                    args,
                ));
                Argument::Variable(dest_ref)
            }
            ast::Expression::Void => Argument::IntegerLiteral(0),
            ast::Expression::PointerExpr(to_expr) => {
                let expr_type = typer.try_infer_type(&to_expr).unwrap();
                let ptr_type = typer.types.pointer_to(expr_type);
                let temp = self.maybe_add_temp_variable(maybe_dest_var, ptr_type);
                let arg = self.compile_expression(typer, &to_expr, Some(&temp));
    
                self.instructions.push(Instruction::Pointer(temp.clone(), arg));
            
                Argument::Variable(temp)
            }
        };
        return value;
    }

    fn maybe_add_temp_variable(&mut self, dest_var: Option<&Variable>, type_: Rc<Type>) -> Variable {
        return match dest_var {
            Some(v) => v.clone(),
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
                let fx_arg_sym = typer.try_resolve_symbol(&fx_arg.name_token.value, SymbolKind::Local, fx.body)
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
                .push(Instruction::Return(Argument::IntegerLiteral(0)));
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
                let var_sym = typer.try_resolve_symbol(&var.name_token.value, SymbolKind::Local, var.parent)
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
                    self.instructions.push(Instruction::Copy(var_ref, init_arg));
                }          
            }
            ast::Statement::Return(ret) => {
                let ret_reg = self.compile_expression(typer, &ret.expr, None);
                self.instructions.push(Instruction::Return(ret_reg));
            }
            ast::Statement::If(if_stmt) => {
                let label_after_block = self.add_label();
                let arg = self.compile_expression(typer, &if_stmt.condition, None);
    
                // jump over the true-block if the result isn't truthy.
                self.instructions.push(Instruction::JumpNotEqual(
                    label_after_block.clone(),
                    arg,
                    Argument::IntegerLiteral(1),
                ));
    
                let if_block = typer.ast.get_block(if_stmt.block);
    
                self.compile_block(typer, if_block);
                self.instructions.push(Instruction::Label(label_after_block));
            }
            ast::Statement::Expression(expr) => {
                self.compile_expression(typer, expr, None);
            },
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
        x = copy 6
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
        x = copy 6
        y = copy x
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
            %0 = add 1, 2
            x = copy %0
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
                %0 = add x, y
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
            x = copy 2
            %0 = add x, 1
            %1 = eq %0, 3
            y = copy %1
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
            %0 = eq 1, 2
            x = copy %0
            jne .LB0, x, 1
            y = copy 42
        .LB0:
            z = copy 3
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
        x = copy struct (5, "hello")
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_pointer_expr_to_literal() {
        let code = r###"
            var x = 1;
            var y = &x;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        println!("{bc}");

        let expected = r###"
        x = copy 1
        %0 = pointer x
        y = copy %0
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_pointer_expr_to_struct() {
        let code = r###"
            var x = "yee!";
            var y = &x;
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        println!("{bc}");

        let expected = r###"
        x = copy struct (4, "yee!")
        %0 = pointer x
        y = copy %0
        "###;

        assert_bytecode_matches(expected, &bc);
    }

    #[test]
    fn should_compile_call_with_pointer_argument() {
        let code = r###"
            fun takes_str(x: &string): void {}
            fun main(): void {
                takes_str(&"yee!");
            }
        "###;

        let bc = Bytecode::from_code(code).unwrap();
        let expected = r###"
        takes_str(x):
          ret 0
        main():
          %0 = pointer struct (4, "yee!")
          %1 = call takes_str(%0)
          ret 0
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
