use std::collections::HashMap;
use std::io::Write;
use crate::bytecode::{self, Argument};
use crate::bytecode::PositiveOffset;
use crate::bytecode::NegativeOffset;
use crate::typer;
use crate::typer::Type;
use crate::util::Error;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
struct ConstId(i64);

impl std::fmt::Display for ConstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.write_str(&format!(".LC{}", self.0));
    }
}

impl Into<MovArgument> for ConstId {
    fn into(self) -> MovArgument {
        return MovArgument::Constant(self);
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
struct X86Assembly {
    instructions: Vec<Instruction>,
    comments: HashMap<usize, String>,
    constants: Vec<Constant>,
}

impl X86Assembly {
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

    fn directive(&mut self, value: &'static str) {
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
        return self.emit_instruction("push", &[reg.to_string()]);
    }

    fn pop(&mut self, reg: Register) {
        return self.emit_instruction("pop", &[reg.to_string()]);
    }

    fn ret(&mut self) {
        return self.emit_instruction("ret", &[]);
    }

    fn mov<A: Into<MovArgument>, B: Into<MovArgument>>(&mut self, dest: A, source: B) {
        return self.emit_instruction("mov", &[dest.into().to_string(), source.into().to_string()]);
    }

    fn movzx<A: Into<MovArgument>>(&mut self, dest: Register, source: A) {
        return self.emit_instruction("movzx", &[dest.to_string(), source.into().to_string()]);
    }

    fn add<A: Into<MovArgument>>(&mut self, dest: Register, source: A) {
        return self.emit_instruction("add", &[dest.to_string(), source.into().to_string()]);
    }

    fn sub<A: Into<MovArgument>>(&mut self, dest: Register, source: A) {
        return self.emit_instruction("sub", &[dest.to_string(), source.into().to_string()]);
    }

    fn call(&mut self, name: &str) {
        return self.emit_instruction("call", &[name.to_string()]);
    }

    fn syscall(&mut self) {
        return self.emit_instruction("syscall", &[]);
    }

    fn cmp<A: Into<MovArgument>>(&mut self, dest: Register, source: A) {
        return self.emit_instruction("cmp", &[dest.to_string(), source.into().to_string()]);
    }

    fn jne(&mut self, to_label: &str) {
        return self.emit_instruction("jne", &[to_label.to_string()]);
    }

    fn sete(&mut self, dest: Register) {
        return self.emit_instruction("sete", &[dest.to_string()]);
    }

    fn nop(&mut self) {
        return self.emit_instruction("nop", &[]);
    }

    fn lea<A: Into<MovArgument>>(&mut self, dest: Register, source: A) {
        return self.emit_instruction("lea", &[dest.to_string(), source.into().to_string()]);
    }
}

impl std::fmt::Display for X86Assembly {
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

#[derive(Debug, Clone, Copy)]
enum OSKind {
    Linux,
    MacOS,
    Windows,
}

#[derive(Debug, Clone, Copy)]
struct OS {
    kind: OSKind,
    syscall_print: i64,
    syscall_exit: i64,
}

impl OS {
    fn current() -> Self {
        let os_name = std::env::consts::OS;
        return match os_name {
            "macos" => OS {
                kind: OSKind::MacOS,
                // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
                // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
                syscall_print: 0x2000000 + 4,
                syscall_exit: 0x2000000 + 1,
            },
            "linux" => OS {
                kind: OSKind::Linux,
                // https://filippo.io/linux-syscall-table/
                syscall_print: 1,
                syscall_exit: 60,
            },
            _ => panic!("Unsupported OS: {}", os_name),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Register(&'static str);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.0.fmt(f);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegisterIndirect(Register, i64);

impl std::fmt::Display for RegisterIndirect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = if self.1 < 0 { "-" } else { "+" };
        return f.write_str(&format!("[{} {} {}]", self.0, op, self.1));
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

const INTEGER_ARGUMENT_REGISTERS: [Register; 6] = [
    RDI,
    RSI,
    RDX,
    RCX,
    R8,
    R9,
];

#[derive(Debug, Clone, Copy)]
enum MovArgument {
    Immediate(i64),
    Register(Register),
    Indirect(Register, i64),
    Constant(ConstId),
}

impl std::fmt::Display for MovArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Immediate(x) => x.fmt(f),
            Self::Register(reg) => reg.fmt(f),
            Self::Indirect(reg, offset) => {
                let op = if *offset < 0 { "-" } else { "+" };
                f.write_str(&format!("qword ptr [{} {} {}]", reg, op, offset.abs()))
            },
            Self::Constant(id) => {
                f.write_str(&format!("qword ptr [{} + {}]", RIP, id))
            }
        };
    }
}

impl Into<MovArgument> for Register {
    fn into(self) -> MovArgument {
        return MovArgument::Register(self);
    }
}

impl Into<MovArgument> for RegisterIndirect {
    fn into(self) -> MovArgument {
        return MovArgument::Indirect(self.0, self.1)
    }
}

impl Into<MovArgument> for i64 {
    fn into(self) -> MovArgument {
        return MovArgument::Immediate(self);
    }
}

fn indirect(register: Register, offset: i64) -> RegisterIndirect {
    return RegisterIndirect(register, offset);
}

const MACOS_COMPILER_ARGS: &[&str] = &[
    "-arch",
    "x86_64",
    "-masm=intel",
    "-x",
    "assembler",
    "-nostartfiles",
    "-nostdlib",
    "-e",
    "main",
];

const LINUX_COMPILER_ARGS: &[&str] = &[
    "-masm=intel",
    "-x",
    "assembler",
    "-nostartfiles",
    "-nolibc",
    "-e",
    "main",
];

fn get_compiler_arguments(out_name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    let args = match OS::current().kind {
        OSKind::Linux => LINUX_COMPILER_ARGS,
        OSKind::MacOS => MACOS_COMPILER_ARGS,
        _ => panic!("Unsupported OS: {}", std::env::consts::OS),
    };

    for arg in args {
        out.push(arg.to_string());
    }

    for arg in ["-o", out_name, "-"] {
        out.push(arg.to_string());
    }

    return out;
}

pub fn emit_binary(asm: &str, out_name: &str) -> String {
    let compiler_args = get_compiler_arguments(out_name);
    let mut child = std::process::Command::new("gcc")
        // let mut child = std::process::Command::new("x86_64-elf-gcc")
        .args(compiler_args)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(asm.as_bytes()).unwrap();
    }

    let output = child.wait_with_output().unwrap();
    let str = std::str::from_utf8(&output.stdout).unwrap();

    return str.to_string();
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

    fn get_offset_or_push(&mut self, var: &bytecode::Variable) -> NegativeOffset {
        let mut offset = 0;

        for k in 0..self.variables.len() {
            let maybe_var = &self.variables[k];
    
            offset -= maybe_var.type_.size();
    
            if maybe_var.name == var.name {
                return NegativeOffset(offset);
            }
        }
    
        offset -= var.type_.size();
        self.variables.push(var.clone());
    
        return NegativeOffset(offset);
    }
}

fn resolve_move_argument(arg: &bytecode::Argument, stack: &mut Stack) -> MovArgument {
    let move_arg = match arg {
        bytecode::Argument::Void => {
            MovArgument::Immediate(0)
        }
        bytecode::Argument::Variable(v) => {
            let offset = stack.get_offset_or_push(v);
            let to_arg = MovArgument::Indirect(RBP, offset.0);
            to_arg
        }
        bytecode::Argument::Integer(i) => {
            MovArgument::Immediate(*i)
        }
        _ => panic!()
    };
    return move_arg;
}

fn align_16(value: i64) -> i64 {
    let mul = (value as f64) / 16.0;
    return (mul.ceil() as i64) * 16;
}

fn determine_stack_size_of_function_at(bc: &bytecode::Bytecode, fx_instr: &bytecode::Instruction, at_index: usize) -> i64 {
    let fx_args = match fx_instr {
        bytecode::Instruction::Function(_, x) => x,
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    let mut sum: i64 = 0;

    for arg in fx_args {
        sum += arg.type_.size();
    }

    for k in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[k];
        if let bytecode::Instruction::Local(var) = instr {
            sum += var.type_.size();
        }

        // we found the next function. let's stop.
        if let bytecode::Instruction::Function(_, _) = instr {
            break;
        }
    }
    return align_16(sum);
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, asm: &mut X86Assembly) -> usize {
    let fx_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fx_instr {
        bytecode::Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    asm.function(fx_name);
    asm.push(RBP);
    asm.mov(RBP, RSP);

    // will be updated later with the correct stack size when we exit the function body.
    let needs_stack_size = determine_stack_size_of_function_at(bc, fx_instr, at_index);

    asm.sub(RSP, needs_stack_size);

    let mut stack = Stack::new();

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let stack_offset = stack.get_offset_or_push(fx_arg);

        asm.mov(indirect(RBP, stack_offset.0), INTEGER_ARGUMENT_REGISTERS[i]);
        asm.add_comment(&format!("{}(): argument {} to stack", fx_name, fx_arg));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            bytecode::Instruction::Local(var) => {
                stack.get_offset_or_push(var);
            }
            bytecode::Instruction::Store(dest_var, field_offset, arg) => {
                assert!(field_offset.0 < dest_var.type_.size());

                let source_arg = resolve_move_argument(arg, &mut stack);
                let stack_offset = stack.get_offset_or_push(dest_var).0 + field_offset.0;

                asm.mov(indirect(RBP, stack_offset), source_arg);
            }
            bytecode::Instruction::AddressOf(dest_var, field_offset, arg) => {
                assert!(field_offset.0 < dest_var.type_.size());

                let stack_offset = stack.get_offset_or_push(dest_var).0 + field_offset.0;

                match arg {
                    Argument::String(s) => {
                        let id = asm.add_constant(s);

                        asm.lea(RAX, id);
                        asm.mov(indirect(RBP, stack_offset), RAX);
                    }
                    Argument::Variable(var) => {
                        let other_offset = stack.get_offset_or_push(var);
                        asm.lea(RAX, indirect(RBP, other_offset.0));
                        asm.mov(indirect(RBP, stack_offset), RAX);
                    }
                    _ => panic!()
                }
                // asm.add_comment(&format!("{}", instr));
            }
            bytecode::Instruction::Return(ret_arg) => {
                let source_arg = resolve_move_argument(ret_arg, &mut stack);
                let os = OS::current();

                if fx_name == "main" {
                    asm.mov(RAX, os.syscall_exit);
                    asm.add_comment("syscall: code exit");

                    asm.mov(RDI, source_arg);
                    asm.add_comment(&format!("syscall: argument {}", ret_arg));

                    asm.syscall();
                } else {
                    asm.mov(RAX, source_arg);
                }

                // will be updated later with the correct stacks size when we exit the function body.
                asm.add(RSP, needs_stack_size);
                asm.pop(RBP);
                asm.ret();
            }
            bytecode::Instruction::Add(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                asm.mov(RAX, arg_a);
                asm.add_comment(&format!("add: lhs argument {}", a));
                let arg_b = resolve_move_argument(b, &mut stack);
                asm.add(RAX, arg_b);
                asm.add_comment(&format!("add: rhs argument {}", b));
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset.0), RAX);
                asm.add_comment("add: result to stack");
            }
            bytecode::Instruction::Sub(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                asm.mov(RAX, arg_a);
                let arg_b = resolve_move_argument(b, &mut stack);
                asm.sub(RAX, arg_b);
                let dest_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, dest_offset.0), RAX);
            }
            bytecode::Instruction::Call(dest_var, fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_move_arg = resolve_move_argument(&fx_arg, &mut stack);
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];

                    asm.mov(call_arg_reg, call_move_arg);
                    asm.add_comment(&format!("{}(): argument {} into register", fx_name, fx_arg));
                }
                asm.call(fx_name);
                let target_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, target_offset.0), RAX);
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
                let arg_a = resolve_move_argument(a, &mut stack);

                asm.mov(RAX, arg_a);

                let arg_b = resolve_move_argument(b, &mut stack);
                asm.cmp(RAX, arg_b);
                asm.sete(AL);
                asm.movzx(RAX, AL);

                let target_offset = stack.get_offset_or_push(dest_var);
                asm.mov(indirect(RBP, target_offset.0), RAX);
            }
            bytecode::Instruction::JumpNotEqual(to_label, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                asm.mov(RAX, arg_a);
                asm.add_comment(&format!("jump: argument {} to register", a));
                let arg_b = resolve_move_argument(b, &mut stack);
                asm.cmp(RAX, arg_b);
                asm.jne(to_label);
                asm.add_comment(&format!("jump: if {} != {} then {}", a, b, to_label));
            }
            bytecode::Instruction::Noop => {
                asm.nop();
            }
        }
    }

    return found_next_index;
}

fn emit_builtins(asm: &mut X86Assembly) {
    let os = OS::current();

    asm.function("print");
    asm.push(RBP);
    asm.mov(RBP, RSP);
    asm.sub(RSP, 16);

    // put the string pointer on the stack. this thing
    // points to the string struct, where the first
    // 8 bytes are the length of the string and the
    // next 8 are a pointer to the character data.
    asm.mov(indirect(RBP, -8), RDI);

    // length
    asm.mov(R10, indirect(RDI, 0));

    // pointer to data
    asm.mov(R11, indirect(RDI, 8));

    asm.mov(RAX, os.syscall_print);
    asm.mov(RDI, 1);
    asm.mov(RSI, R11);
    asm.mov(RDX, R10);

    asm.syscall();
    asm.add(RSP, 16);
    asm.pop(RBP);
    asm.ret();
}

fn emit_instructions(code: &str) -> Result<X86Assembly, Error> {
    let bytecode = bytecode::Bytecode::from_code(code)?;
    let mut asm = X86Assembly {
        instructions: Vec::new(),
        comments: HashMap::new(),
        constants: Vec::new(),
    };

    asm.directive(".intel_syntax noprefix");
    asm.directive(".globl main");

    emit_builtins(&mut asm);

    let mut index: usize = 0;

    while let Some(instr) = bytecode.instructions.get(index) {
        match instr {
            bytecode::Instruction::Function(_, _) => {
                index = emit_function(&bytecode, index, &mut asm);
            }
            _ => panic!(),
        };
    }

    return Ok(asm);
}

pub fn emit_assembly(code: &str) -> Result<String, Error> {
    let asm = emit_instructions(code)?;
    return Ok(asm.to_string());
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::x64::*;

    #[test]
    fn should_return_code() {
        let s = r###"
            fun main(): int {
                return 5;
            }
        "###;
        do_test(5, s);
    }

    #[test]
    fn should_call_fn() {
        let code = r###"
        fun add(x: int): int {
            return x;
        }
        fun main(): int {
            return add(3);
        }
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
            fun main(): int {
                return factorial(5);
            }
        "###;

        do_test(120, code);
    }

    #[test]
    fn should_call_print_with_variable_arg() {
        let code = r###"
        fun main(): int {
            var x = "hello!";
            print(x);
            return 0;
        }
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    #[test]
    fn should_call_print_with_literal_arg() {
        let code = r###"
        fun main(): int {
            print("hello!");
            return 0;
        }
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    #[test]
    fn should_call_print_in_sub_procedure_with_string_passed_by_reference() {
        let code = r###"
        fun thing(x: string): void {
            print(x);
        }

        fun main(): int {
            thing("hello!");
            return 0;
        }
        "###;
        let out = do_test(0, code);

        assert_eq!("hello!", out);
    }

    fn do_test(expected_code: i32, code: &str) -> String {
        let asm = emit_assembly(code).unwrap();
        println!("{asm}");

        let bin_name = format!("_test_{}.out", random_str(8));

        emit_binary(&asm, &bin_name);
        
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
