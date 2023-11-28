use crate::bytecode::{self, Argument, ENTRYPOINT_NAME};
use crate::typer;
use crate::typer::Type;
use crate::util::Error;
use crate::util::OperatingSystem;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;
use crate::util::Offset;

#[derive(Debug, Clone, Copy)]
struct ConstId(i64);

impl std::fmt::Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!(".LC{}", self.0));
    }
}

#[derive(Debug, Clone)]
struct Constant {
    id: ConstId,
    value: String,
}

#[derive(Debug, Clone)]
struct Instruction(String);

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.0.fmt(f);
    }
}

#[derive(Debug)]
pub struct Assembly {
    comments: HashMap<usize, String>,
    constants: Vec<Constant>,
    instructions: Vec<Instruction>,
    os: &'static OperatingSystem,
}

impl Assembly {
    fn add_comment(&mut self, comment: &str) {
        let index = self.instructions.len() - 1;
        self.comments.insert(index, comment.to_string());
    }

    fn add_constant(&mut self, value: &str) -> ConstId {
        let id = ConstId(self.constants.len() as i64);
        let c = Constant {
            id: id,
            value: value.to_string(),
        };
        self.constants.push(c);
        return id;
    }

    fn emit_instruction(&mut self, name: &'static str, args: &[String]) {
        let arg_s = args.join(", ");
        let instr = Instruction(format!("  {} {}", name, arg_s));
        self.instructions.push(instr);
    }

    fn directive(&mut self, value: &str) {
        self.instructions.push(Instruction(value.to_string()));
    }

    fn function(&mut self, name: &str) {
        let instr = Instruction(format!("{}:", name));
        self.instructions.push(instr);
    }

    fn label(&mut self, name: &str) {
        self.function(name);
    }

    fn push(&mut self, reg: Register) {
        self.emit_instruction("push", &[reg.to_string()]);
    }

    fn pop(&mut self, reg: Register) {
        self.emit_instruction("pop", &[reg.to_string()]);
    }

    fn ret(&mut self) {
        self.emit_instruction("ret", &[]);
    }

    fn mov<A: Into<InstructionArgument>, B: Into<InstructionArgument>>(
        &mut self,
        dest: A,
        source: B,
    ) {
        self.emit_instruction("mov", &[dest.into().to_string(), source.into().to_string()]);
    }

    fn movzx<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("movzx", &[dest.to_string(), source.into().to_string()]);
    }

    fn add<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("add", &[dest.to_string(), source.into().to_string()]);
    }

    fn sub<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("sub", &[dest.to_string(), source.into().to_string()]);
    }

    fn imul<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("imul", &[dest.to_string(), source.into().to_string()]);
    }

    fn idiv<A: Into<InstructionArgument>>(&mut self, divisor: A) {
        self.emit_instruction("idiv", &[divisor.into().to_string()]);
    }

    fn call(&mut self, name: &str) {
        self.emit_instruction("call", &[name.to_string()]);
    }

    fn syscall(&mut self) {
        self.emit_instruction("syscall", &[]);
    }

    fn cmp<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("cmp", &[dest.to_string(), source.into().to_string()]);
    }

    fn jne(&mut self, to_label: &str) {
        self.emit_instruction("jne", &[to_label.to_string()]);
    }

    fn sete(&mut self, dest: Register) {
        self.emit_instruction("sete", &[dest.to_string()]);
    }

    fn lea<A: Into<InstructionArgument>>(&mut self, dest: Register, source: A) {
        self.emit_instruction("lea", &[dest.to_string(), source.into().to_string()]);
    }

    fn cqo(&mut self) {
         // sign extend RAX into RDX. mainly useful for signed divide.
        self.emit_instruction("cqo", &[]);
    }
}

impl std::fmt::Display for Assembly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::with_capacity(8192);

        for i in 0..self.instructions.len() {
            let instr = &self.instructions[i];
            let maybe_comment = match self.comments.get(&i) {
                Some(s) => format!("# {}", s.trim()),
                None => "".to_string(),
            };

            s.push_str(&format!("{:48}{}\n", instr.to_string(), maybe_comment));
        }

        for c in &self.constants {
            s.push_str(&format!("{}:\n", c.id));
            s.push_str(&format!("  .ascii \"{}\"\n", c.value));
        }

        return f.write_str(&s);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Register(&'static str);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.0.fmt(f);
    }
}

#[derive(Debug, Clone, Copy)]
struct RegisterIndirect(Register, Offset);

impl std::fmt::Display for RegisterIndirect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!("[{}{}]", self.0, self.1));
    }
}

const RAX: Register = Register("rax");
const RBX: Register = Register("rbx");
const RCX: Register = Register("rcx");
const RDX: Register = Register("rdx");
const RSI: Register = Register("rsi");
const RDI: Register = Register("rdi");
const RSP: Register = Register("rsp");
const RBP: Register = Register("rbp");
const R8: Register = Register("r8");
const R9: Register = Register("r9");
const R10: Register = Register("r10");
const R11: Register = Register("r11");
const R12: Register = Register("r12");
const R13: Register = Register("r13");
const R14: Register = Register("r14");
const R15: Register = Register("r15");
const RIP: Register = Register("rip");
const AL: Register = Register("al");

const INTEGER_ARGUMENT_REGISTERS: [Register; 6] = [RDI, RSI, RDX, RCX, R8, R9];

#[derive(Debug, Clone, Copy)]
enum InstructionArgument {
    Immediate(i64),
    Register(Register),
    Indirect(Register, Offset),
    Constant(ConstId),
}

impl std::fmt::Display for InstructionArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Immediate(x) => x.fmt(f),
            Self::Register(reg) => reg.fmt(f),
            Self::Indirect(reg, offset) => {
                f.write_str(&format!("qword ptr [{}{}]", reg, offset))
            }
            Self::Constant(id) => f.write_str(&format!("qword ptr [{}+{}]", RIP, id)),
        };
    }
}

impl Into<InstructionArgument> for Register {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Register(self);
    }
}

impl Into<InstructionArgument> for RegisterIndirect {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Indirect(self.0, self.1);
    }
}

impl Into<InstructionArgument> for i64 {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Immediate(self);
    }
}

impl Into<InstructionArgument> for ConstId {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Constant(self);
    }
}

fn indirect<T: Into<Offset>>(register: Register, offset: T) -> RegisterIndirect {
    return RegisterIndirect(register, offset.into());
}

#[derive(Debug, Clone)]
struct Stack {
    variables: Vec<bytecode::Variable>,
}

impl Stack {
    fn new() -> Self {
        return Self {
            variables: Vec::new(),
        };
    }

    fn get_offset_or_push(&mut self, var: &bytecode::Variable) -> Offset {
        let mut offset: i64 = 0;

        for k in 0..self.variables.len() {
            let maybe_var = &self.variables[k];

            offset += maybe_var.type_.size();

            if maybe_var.name == var.name {
                return Offset::Negative(offset);
            }
        }

        offset += var.type_.size();
        self.variables.push(var.clone());

        return Offset::Negative(offset);
    }
}

fn create_mov_source_for_dest<T: Into<InstructionArgument>>(
    dest: T,
    dest_type: &Type,
    source: &bytecode::Argument,
    stack: &mut Stack,
    asm: &mut Assembly,
) -> InstructionArgument {
    let dest = dest.into();
    
    let move_arg = match source {
        bytecode::Argument::Void => InstructionArgument::Immediate(0),
        bytecode::Argument::Variable(v, field_offset) => {
            let offset_to_variable = stack.get_offset_or_push(v);
            let is_dest_stack = matches!(dest, InstructionArgument::Indirect(RBP, _));
            let is_pointer_add = dest_type.is_pointer() && v.type_.is_pointer();
            let needs_deref = dest_type.is_scalar() && v.type_.is_pointer();
            let mut arg = InstructionArgument::Indirect(RBP, offset_to_variable.add(*field_offset));

            if is_pointer_add {
                asm.mov(RAX, indirect(RBP, offset_to_variable));
                asm.add(RAX, field_offset.to_i64());
                arg = InstructionArgument::Register(RAX);
            } else if is_dest_stack {
                // we can't mov directly between stack variables. emit
                // an intermediate mov into a register.
                asm.mov(RAX, arg);
                arg = InstructionArgument::Register(RAX);
            }

            if needs_deref {
                asm.mov(RAX, arg);
                asm.mov(RAX, indirect(RAX, 0));
                arg = InstructionArgument::Register(RAX);
            }

            return arg;
        }
        bytecode::Argument::Int(i) => InstructionArgument::Immediate(*i),
        _ => panic!("bad. got source = {:?}", source),
    };
    return move_arg;
}

fn align_16(value: i64) -> i64 {
    let mul = (value as f64) / 16.0;
    return (mul.ceil() as i64) * 16;
}

fn determine_stack_size_of_function(bc: &bytecode::Bytecode, at_index: usize) -> i64 {
    let fx_instr = &bc.instructions[at_index];
    let fx_args = match fx_instr {
        bytecode::Instruction::Function(_, x) => x,
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    let mut sum: i64 = 0;

    for arg in fx_args {
        sum += arg.type_.size() as i64;
    }

    for k in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[k];
        if let bytecode::Instruction::Local(var) = instr {
            sum += var.type_.size() as i64;
        }

        // we found the next function. let's stop.
        if let bytecode::Instruction::Function(_, _) = instr {
            break;
        }
    }
    return align_16(sum);
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, asm: &mut Assembly) -> usize {
    let fx_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fx_instr {
        bytecode::Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    asm.function(fx_name);
    asm.push(RBP);
    asm.mov(RBP, RSP);

    // will be updated later with the correct stack size when we exit the function body.
    let needs_stack_size = determine_stack_size_of_function(bc, at_index);

    asm.sub(RSP, needs_stack_size);

    let mut stack = Stack::new();

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let stack_offset = stack.get_offset_or_push(fx_arg);

        asm.mov(indirect(RBP, stack_offset), INTEGER_ARGUMENT_REGISTERS[i]);
        asm.add_comment(&format!("{}(): argument {} to stack", fx_name, fx_arg));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            bytecode::Instruction::Local(var) => {
                stack.get_offset_or_push(var);
            }
            bytecode::Instruction::Store(dest_var, dest_offset, source) => {
                // assert!(dest_offset.0 < dest_var.type_.size());

                let stack_offset = stack.get_offset_or_push(dest_var).add(*dest_offset);
                let mov_dest = indirect(RBP, stack_offset);
                let mov_source = create_mov_source_for_dest(mov_dest, &dest_var.type_, source, &mut stack, asm);

                asm.mov(mov_dest, mov_source);
            }
            bytecode::Instruction::AddressOf(dest_var, dest_offset, source) => {
                // assert!(field_offset.0 < dest_var.type_.size());

                let stack_offset = stack.get_offset_or_push(dest_var).add(*dest_offset);

                match source {
                    Argument::String(s) => {
                        let id = asm.add_constant(s);

                        asm.lea(RAX, id);
                        asm.mov(indirect(RBP, stack_offset), RAX);
                    }
                    Argument::Variable(var, source_offset) => {
                        let other_offset = stack.get_offset_or_push(var).add(*source_offset);
                        asm.lea(RAX, indirect(RBP, other_offset));
                        asm.mov(indirect(RBP, stack_offset), RAX);
                    }
                    _ => panic!(),
                }
            }
            bytecode::Instruction::Return(ret_arg) => {
                if fx_name == ENTRYPOINT_NAME {
                    asm.mov(RAX, asm.os.syscall_exit);
                    asm.add_comment("syscall: code exit");

                    let mov_source = create_mov_source_for_dest(RDI, &ret_arg.get_type(), ret_arg, &mut stack, asm);
                    asm.mov(RDI, mov_source);
                    asm.add_comment(&format!("syscall: argument {}", ret_arg));

                    asm.syscall();
                } else {
                    let mov_source = create_mov_source_for_dest(RAX, &ret_arg.get_type(), ret_arg, &mut stack, asm);
                    asm.mov(RAX, mov_source);
                }

                // will be updated later with the correct stacks size when we exit the function body.
                asm.add(RSP, needs_stack_size);
                asm.pop(RBP);
                asm.ret();
            }
            bytecode::Instruction::Add(dest_var, a, b) => {
                let mov_source_a = create_mov_source_for_dest(RAX, &Type::Int, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);
                asm.add_comment(&format!("add: lhs argument {}", a));
                let mov_source_b = create_mov_source_for_dest(RAX, &Type::Int, b, &mut stack, asm);
                asm.add(RAX, mov_source_b);
                asm.add_comment(&format!("add: rhs argument {}", b));
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset), RAX);
                asm.add_comment("add: result to stack");
            }
            bytecode::Instruction::Sub(dest_var, a, b) => {
                let mov_source_a = create_mov_source_for_dest(RAX, &Type::Int, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);
                let mov_source_b = create_mov_source_for_dest(RAX, &Type::Int, b, &mut stack, asm);
                asm.sub(RAX, mov_source_b);
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset), RAX);
            }
            bytecode::Instruction::Mul(dest_var, a, b) => {
                let mov_source_a = create_mov_source_for_dest(RAX, &Type::Int, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);
                let mov_source_b = create_mov_source_for_dest(RAX, &Type::Int, b, &mut stack, asm);
                asm.imul(RAX, mov_source_b);
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset), RAX);
            }
            bytecode::Instruction::Div(dest_var, a, b) => {
                let mov_source_a = create_mov_source_for_dest(RAX, &Type::Int, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);
                asm.cqo();
                let mov_source_b = create_mov_source_for_dest(R8, &Type::Int, b, &mut stack, asm);
                asm.mov(R8, mov_source_b);
                asm.idiv(R8);
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset), RAX);
            }
            bytecode::Instruction::Call(dest_var, fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];
                    let mov_source =
                        create_mov_source_for_dest(call_arg_reg, &fx_arg.get_type(), &fx_arg, &mut stack, asm);

                    asm.mov(call_arg_reg, mov_source);
                    asm.add_comment(&format!("{}(): argument {} into register", fx_name, fx_arg));
                }
                asm.call(fx_name);
                let target_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, target_offset), RAX);
                asm.add_comment(&format!("{}(): return value to stack", fx_name));
            }
            bytecode::Instruction::Label(name) => {
                asm.label(name);
            }
            bytecode::Instruction::Function(_, _) => {
                found_next_index = body_index;

                // this is the start of another function. we might encounter
                // several return statements and we must make sure to read
                // all of them. however, when we hit another function we
                // can surely stop.
                break;
            }
            bytecode::Instruction::IsEqual(dest_var, a, b) => {
                let dest_type = &dest_var.type_;
                let mov_source_a = create_mov_source_for_dest(RAX, dest_type, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);

                let mov_source_b = create_mov_source_for_dest(RAX, dest_type, b, &mut stack, asm);
                asm.cmp(RAX, mov_source_b);
                asm.sete(AL);
                asm.movzx(RAX, AL);

                let target_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, target_offset), RAX);
            }
            bytecode::Instruction::JumpNotEqual(to_label, a, b) => {
                let mov_source_a = create_mov_source_for_dest(RAX, &Type::Int, a, &mut stack, asm);
                asm.mov(RAX, mov_source_a);
                asm.add_comment(&format!("jump: argument {} to register", a));
                let mov_source_b = create_mov_source_for_dest(RAX, &Type::Int, b, &mut stack, asm);
                asm.cmp(RAX, mov_source_b);
                asm.jne(to_label);
                asm.add_comment(&format!("jump: if {} != {} then {}", a, b, to_label));
            }
        }
    }

    return found_next_index;
}

fn emit_builtins(asm: &mut Assembly) {
    asm.function("print");
    asm.push(RBP);
    asm.mov(RBP, RSP);

    asm.sub(RSP, 16);

    // RDI is a pointer to a 'string' struct. the first 8 bytes holds
    // the length of the string and the next 8 bytes is a pointer to the
    // data segment.

    // length to stack
    asm.mov(RAX, indirect(RDI, 0));
    asm.mov(indirect(RBP, -8), RAX);

    // data pointer to stack
    asm.mov(RAX, indirect(RDI, 8));
    asm.mov(indirect(RBP, -16), RAX);

    asm.mov(RAX, asm.os.syscall_print);
    asm.mov(RDI, 1);
    asm.mov(RSI, indirect(RBP, -16));
    asm.mov(RDX, indirect(RBP, -8));

    asm.syscall();
    asm.add(RSP, 16);
    asm.pop(RBP);
    asm.ret();

    // an exit function!
    asm.function("exit");
    asm.push(RBP);
    asm.mov(RBP, RSP);

    asm.sub(RSP, 16);
    asm.mov(indirect(RBP, -8), RDI);

    asm.mov(RAX, asm.os.syscall_exit);
    asm.mov(RDI, indirect(RBP, -8));

    asm.syscall();
    asm.add(RSP, 16);
    asm.pop(RBP);
    asm.ret();
}

pub fn emit_assembly(code: &str, os: &'static OperatingSystem) -> Result<Assembly, Error> {
    let bytecode = bytecode::Bytecode::from_code(code)?;
    let mut asm = Assembly {
        instructions: Vec::new(),
        comments: HashMap::new(),
        constants: Vec::new(),
        os: os,
    };

    asm.directive(".intel_syntax noprefix");
    asm.directive(&format!(".globl {}", ENTRYPOINT_NAME));

    emit_builtins(&mut asm);

    let mut index: usize = 0;

    while let Some(instr) = bytecode.instructions.get(index) {
        match instr {
            bytecode::Instruction::Function(_, _) => {
                index = emit_function(&bytecode, index, &mut asm);
            }
            _ => panic!("{:?}", instr),
        };
    }

    return Ok(asm);
}

pub fn emit_binary(
    code: &str,
    out_name: &str,
    os: &'static OperatingSystem,
) -> Result<String, Error> {
    let asm = emit_assembly(code, os)?;
    let compiler_args = [asm.os.compiler_args, &["-o", out_name, "-"]].concat();

    let mut child = std::process::Command::new(asm.os.compiler_bin)
        // let mut child = std::process::Command::new("x86_64-elf-gcc")
        .args(compiler_args)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let asm_as_string = asm.to_string();

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(asm_as_string.as_bytes()).unwrap();
    }

    let output = child.wait_with_output().unwrap();
    let str = std::str::from_utf8(&output.stdout).unwrap();

    return Ok(asm_as_string);
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::{x64::*, util::with_stdlib};

    #[test]
    fn should_return_code() {
        let s = r###"
            exit(5);
        "###;
        do_test(5, s);
    }

    #[test]
    fn should_call_fn() {
        let code = r###"
        fun add(x: int): int {
            return x;
        }
        var x = add(3);
        exit(x);
        "###;
        do_test(3, code);
    }

    #[test]
    fn should_call_factorial() {
        let code = r###"
            fun mul(x: int, y: int): int {
                if y == 0 {
                    return 0;
                }
                return x + mul(x, y - 1);
            }
            fun factorial(n: int): int {
                if n == 1 {
                    return 1;
                }
                return mul(n, factorial(n - 1));
            }
            var x = factorial(5);
            exit(x);
        "###;

        do_test(120, code);
    }

    #[test]
    fn should_call_print_with_variable_arg() {
        let code = r###"
        var x = "hello!";
        print(&x);
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    #[test]
    fn should_call_print_with_literal_arg() {
        let code = r###"
        print(&"hello!");
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    #[test]
    fn should_call_print_in_sub_procedure_with_string_passed_by_reference() {
        let code = r###"
        fun thing(x: &string): void {
          print(x);
        }
        thing(&"hello!");
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    #[test]
    fn should_call_print_with_inline_member_access() {
        let code = r###"
        type person = {
            name: string,
        };

        var x = person { name: "helmut" };
        print(&x.name);
        "###;
        let out = do_test(0, code);

        assert_eq!("helmut", out);
    }

    #[test]
    fn should_call_print_with_member_access_in_variable() {
        let code = r###"
        type person = {
            name: string,
        };

        var x = person { name: "helmut" };
        var y = x.name;
        print(&y);
        "###;
        let out = do_test(0, code);

        assert_eq!("helmut", out);
    }

    #[test]
    fn should_call_print_with_deep_member_access() {
        let code = r###"
        type C = {
            yee: string,
        };
        type B = {
            c: C,  
        };
        type A = {
            b: B,
        };

        var x = A { b: B { c: C { yee: "cowabunga!" } } };
        print(&x.b.c.yee);
        "###;
        let out = do_test(0, code);

        assert_eq!("cowabunga!", out);
    }

    #[test]
    fn should_call_print_with_derefefenced_variable() {
        let code = r###"
        type B = { value: string };
        type A = { b: B };

        fun takes(a: &A): void {
            print(a.b.value);
        }
        var x = A { b: B { value: "cowabunga!" } };
        takes(&x);
        "###;
        let out = do_test(0, code);

        assert_eq!("cowabunga!", out);
    }

    #[test]
    fn should_call_print_with_derefefenced_variable_with_offset() {
        let code = r###"
        type B = { yee: int, boi: int, value: string };
        type A = { b: B };

        fun takes(a: &A): void {
            print(a.b.value);
        }
        var x = A { b: B { yee: 420, boi: 69, value: "cowabunga!" } };
        takes(&x);
        "###;
        let out = do_test(0, code);

        assert_eq!("cowabunga!", out);
    }

    #[test]
    fn should_derefence_scalar_and_add() {
        let code = r###"
        type A = { x: int };

        fun takes(a: &A): int {
            return a.x + 1;
        }
        var x = A { x: 69 };
        var y = takes(&x);
        exit(y);
        "###;
        let out = do_test(70, code);
    }

    #[test]
    fn should_derefence_scalar_into_local_and_add() {
        let code = r###"
        type A = { x: int };

        fun takes(a: &A): int {
            var y: &int = a.x;
            return y + 1;
        }
        var x = A { x: 69 };
        var y = takes(&x);
        exit(y);
        "###;
        let out = do_test(70, code);
    }

    #[test]
    fn should_jump_with_else_if() {
        let code = r###"
        if 1 == 2 {
            print(&"bad!");
        } else if 5 == 5 {
            print(&"cowabunga!");
        }
        "###;
        let out = do_test(0, code);
        assert_eq!("cowabunga!", out);
    }

    #[test]
    fn should_jump_with_else() {
        let code = r###"
        if 1 == 2 {
            exit(42);
        } else {
            exit(69);
        }
        "###;
        let out = do_test(69, code);
    }

    #[test]
    fn should_jump_with_boolean_literal() {
        let code = r###"
        if false {
            exit(42);
        } else if true {
            exit(69);
        }
        "###;
        let out = do_test(69, code);
    }

    #[test]
    fn should_multiply() {
        let code = r###"
        var x = 3 * 3;
        exit(x);
        "###;
        let out = do_test(9, code);
    }

    #[test]
    fn should_multiply_negative_number() {
        let code = r###"
        var x = -4 * -4;
        exit(x);
        "###;
        let out = do_test(16, code);
    }

    #[test]
    fn should_divide() {
        let code = r###"
        var x = 6 / 2;
        exit(x);
        "###;
        let out = do_test(3, code);
    }

    #[test]
    fn should_divide_negative_number() {
        let code = r###"
        var x = -8 / -2;
        exit(x);
        "###;
        let out = do_test(4, code);
    }

    #[test]
    fn should_divide_with_remainder() {
        let code = r###"
        var x = 9 / 2;
        exit(x);
        "###;
        let out = do_test(4, code);
    }

    #[test]
    fn should_respect_operator_precedence() {
        let code = r###"
        var x = (1 + 2) * 3;
        exit(x);
        "###;
        let out = do_test(9, code);
    }

    #[test]
    fn should_do_math() {
        let code = r###"
        assert(5 == 5);
        assert(5 * 5 == 25);
        assert(-5 * -5 == 25);
        assert(5 + 3 * 5 == 20);
        assert(5 * -1 == -5);
        assert(5 / -1 == -5);
        assert((5 + 3) * 2 == 16);

        print(&"cowabunga!");
        "###;

        let out = do_test(0, &with_stdlib(code));
        assert_eq!("cowabunga!", out);
    }

    #[test]
    fn should_enter_falsy_while_condition() {
        let code = r###"
        while 1 == 1 {
            exit(42);
        }
        exit(3);
        "###;
        do_test(42, code);
    }

    #[test]
    fn should_not_enter_falsy_while_condition() {
        let code = r###"
        while 1 == 2 {
            exit(42);
        }
        exit(3);
        "###;
        do_test(3, code);
    }

    #[test]
    fn should_compile_not_equals() {
        let code = r###"
        assert(1 != 2);
        assert(1 == 1);
        "###;
        do_test(0, &with_stdlib(code));
    }

    #[test]
    fn should_compile_reassignment_to_local() {
        let code = r###"
        var x = 0;
        x = 5;
        exit(x);
        "###;
        do_test(5, &with_stdlib(code));
    }

    #[test]
    fn should_compile_reassignment_to_member() {
        let code = r###"
        type person = { age: int };
        var x = person { age: 3 };
        x.age = 7;
        exit(x.age);
        "###;
        do_test(7, &with_stdlib(code));
    }

    fn do_test(expected_code: i32, code: &str) -> String {
        let os = OperatingSystem::current();
        let bin_name = format!("_test_{}.out", random_str(8));

        emit_binary(&code, &bin_name, os).unwrap();

        let stdout = std::process::Stdio::piped();
        let out = std::process::Command::new(format!("./{bin_name}"))
            .stdout(stdout)
            .spawn()
            .unwrap()
            .wait_with_output()
            .unwrap();

        std::process::Command::new("rm")
            .args(["-rf", &bin_name])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        assert_eq!(expected_code, out.status.code().unwrap());

        let str = std::str::from_utf8(&out.stdout).unwrap();

        return str.to_string();
    }

    fn random_str(len: usize) -> String {
        let num_bytes = len / 2;
        let mut buf: Vec<u8> = (0..num_bytes).map(|_| 0).collect();
        std::fs::File::open("/dev/urandom")
            .unwrap()
            .read_exact(&mut buf)
            .unwrap();
        return buf.iter().map(|x| format!("{:x?}", x)).collect::<String>();
    }
}
