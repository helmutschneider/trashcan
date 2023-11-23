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

#[derive(Debug, Clone)]
struct Constant {
    id: ConstId,
    value: String,
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

#[derive(Debug, Clone, Copy)]
enum Register {
    RAX,
    RBX,
    RCX,
    RDX,
    RBP,
    RSP,
    RSI,
    RDI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,

    AL,
}

const INTEGER_ARGUMENT_REGISTERS: [Register; 6] = [
    Register::RDI,
    Register::RSI,
    Register::RDX,
    Register::RCX,
    Register::R8,
    Register::R9,
];

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::RAX => "rax",
            Self::RBX => "rbx",
            Self::RCX => "rcx",
            Self::RDX => "rdx",
            Self::RSP => "rsp",
            Self::RSI => "rsi",
            Self::RDI => "rdi",
            Self::RBP => "rbp",
            Self::R8 => "r8",
            Self::R9 => "r9",
            Self::R10 => "r10",
            Self::R11 => "r11",
            Self::R12 => "r12",
            Self::R13 => "r13",
            Self::R14 => "r14",
            Self::R15 => "r15",
            Self::AL => "al",
        };
        return f.write_str(s);
    }
}

#[derive(Debug, Clone, Copy)]
enum MovArgument {
    Register(Register),
    Integer(i64),
    IndirectAddress(Register, i64),
    Constant(ConstId),
}

impl std::fmt::Display for MovArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Register(reg) => reg.fmt(f),
            Self::Integer(x) => x.fmt(f),
            Self::IndirectAddress(reg, offset) => {
                let op = if *offset > 0 { "+" } else { "-" };
                f.write_str(&format!("qword ptr [{} {} {}]", reg, op, offset.abs()))
            },
            Self::Constant(id) => {
                f.write_str(&format!("qword ptr [rip + {}]", id))
            }
        };
    }
}

#[derive(Debug, Clone)]
enum Instruction {
    Directive(String),
    Function(String),
    Label(String),
    Push(Register),
    Pop(Register),
    Ret,
    Mov(MovArgument, MovArgument),
    Movzx(Register, MovArgument),
    Add(Register, MovArgument),
    Sub(Register, MovArgument),
    Call(String),
    Syscall,
    Cmp(Register, MovArgument),
    Jne(String),
    Sete(Register),
    Nop,
    Lea(Register, MovArgument),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Directive(dir) => dir.clone(),
            Self::Function(name) => format!("{}:", name),
            Self::Push(reg) => format!("  push {}", reg),
            Self::Pop(reg) => format!("  pop {}", reg),
            Self::Ret => "  ret".to_string(),
            Self::Mov(dest, source) => format!("  mov {}, {}", dest, source),
            Self::Movzx(dest, source) => format!("  movzx {}, {}", dest, source),
            Self::Add(dest, source) => format!("  add {}, {}", dest, source),
            Self::Sub(dest, source) => format!("  sub {}, {}", dest, source),
            Self::Call(fx) => format!("  call {}", fx),
            Self::Syscall => "  syscall".to_string(),
            Self::Label(name) => format!("{}:", name),
            Self::Cmp(a, b) => format!("  cmp {}, {}", a, b),
            Self::Jne(to_label) => format!("  jne {}", to_label),
            Self::Sete(reg) => format!("  sete {}", reg),
            Self::Nop => format!("  nop"),
            Self::Lea(reg, a) => format!("  lea {}, {}", reg, a),
        };
        return f.write_str(&s);
    }
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

    fn size(&self) -> i64 {
        let mut sum: i64 = 0;
        for var in &self.variables {
            sum += var.type_.size();
        }
        return sum;
    }
}

fn resolve_move_argument(
    arg: &bytecode::Argument,
    stack: &mut Stack,
    out: &mut X86Assembly,
) -> MovArgument {
    let move_arg = match arg {
        bytecode::Argument::Void => {
            MovArgument::Integer(0)
        }
        bytecode::Argument::Variable(v) => {
            let offset = stack.get_offset_or_push(v);
            let to_arg = MovArgument::IndirectAddress(Register::RBP, offset.0);
            to_arg
        }
        bytecode::Argument::Integer(i) => {
            MovArgument::Integer(*i)
        }
        _ => panic!()
    };
    return move_arg;
}

fn align_16(value: i64) -> i64 {
    let mul = (value as f64) / 16.0;
    return (mul.ceil() as i64) * 16;
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, out: &mut X86Assembly) -> usize {
    let fun_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fun_instr {
        bytecode::Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fun_instr),
    };

    // used by the loop below to find the add/sub RSP instructions.
    let fn_body_starts_at_index = out.instructions.len() + 1;

    out.instructions.push(Instruction::Function(fx_name.clone()));
    out.instructions.push(Instruction::Push(Register::RBP));
    out.instructions.push(Instruction::Mov(
        MovArgument::Register(Register::RBP),
        MovArgument::Register(Register::RSP),
    ));

    // will be updated later with the correct stack size when we exit the function body.
    let needs_stack_size = MovArgument::Integer(-42069);
    out.instructions.push(Instruction::Sub(Register::RSP, needs_stack_size));

    let mut stack = Stack::new();

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let stack_offset = stack.get_offset_or_push(fx_arg);

        out.instructions.push(Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, stack_offset.0),
            MovArgument::Register(INTEGER_ARGUMENT_REGISTERS[i]),
        ));
        out.add_comment(&format!("{}(): argument {} to stack", fx_name, fx_arg));
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

                let source_arg = resolve_move_argument(arg, &mut stack, out);
                let stack_offset = stack.get_offset_or_push(dest_var).0 + field_offset.0;
                let instr = Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, stack_offset),
                    source_arg
                );
                out.instructions.push(instr);
            }
            bytecode::Instruction::AddressOf(dest_var, field_offset, arg) => {
                assert!(field_offset.0 < dest_var.type_.size());

                let stack_offset = stack.get_offset_or_push(dest_var).0 + field_offset.0;

                match arg {
                    Argument::String(s) => {
                        let id = out.add_constant(s);

                        out.instructions.push(Instruction::Lea(
                            Register::RAX,
                            MovArgument::Constant(id)
                        ));
                        out.instructions.push(Instruction::Mov(
                            MovArgument::IndirectAddress(Register::RBP, stack_offset),
                            MovArgument::Register(Register::RAX)
                        ));
                    }
                    Argument::Variable(var) => {
                        let other_offset = stack.get_offset_or_push(var);
                        out.instructions.push(Instruction::Lea(
                            Register::RAX,
                            MovArgument::IndirectAddress(Register::RBP, other_offset.0)
                        ));
                        out.instructions.push(Instruction::Mov(
                            MovArgument::IndirectAddress(Register::RBP, stack_offset),
                            MovArgument::Register(Register::RAX)
                        ));
                    }
                    _ => panic!()
                }
                out.add_comment(&format!("{}", instr));
            }
            bytecode::Instruction::Return(ret_arg) => {
                let source_arg = resolve_move_argument(ret_arg, &mut stack, out);
                let os = OS::current();

                if fx_name == "main" {
                    out.instructions.push(Instruction::Mov(
                        MovArgument::Register(Register::RAX),
                        MovArgument::Integer(os.syscall_exit),
                    ));
                    out.add_comment("syscall: code exit");
                    out.instructions.push(Instruction::Mov(
                        MovArgument::Register(Register::RDI),
                        source_arg,
                    ));
                    out.add_comment(&format!("syscall: argument {}", ret_arg));
                    out.instructions.push(Instruction::Syscall);
                } else {
                    out.instructions.push(Instruction::Mov(
                        MovArgument::Register(Register::RAX),
                        source_arg,
                    ));
                }

                // will be updated later with the correct stacks size when we exit the function body.
                out.instructions.push(Instruction::Add(Register::RSP, needs_stack_size));
                out.instructions.push(Instruction::Pop(Register::RBP));
                out.instructions.push(Instruction::Ret);
            }
            bytecode::Instruction::Add(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack, out);
                out.instructions.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                out.add_comment(&format!("add: lhs argument {}", a));
                let arg_b = resolve_move_argument(b, &mut stack, out);
                out.instructions.push(Instruction::Add(Register::RAX, arg_b));
                out.add_comment(&format!("add: rhs argument {}", b));
                let dest_offset = stack.get_offset_or_push(dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset.0),
                    MovArgument::Register(Register::RAX),
                ));
                out.add_comment("add: result to stack");
            }
            bytecode::Instruction::Sub(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack, out);
                out.instructions.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                let arg_b = resolve_move_argument(b, &mut stack, out);
                out.instructions.push(Instruction::Sub(Register::RAX, arg_b));
                let dest_offset = stack.get_offset_or_push(dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset.0),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::Call(dest_var, fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_move_arg = resolve_move_argument(&fx_arg, &mut stack, out);
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];

                    out.instructions.push(Instruction::Mov(
                        MovArgument::Register(call_arg_reg),
                        call_move_arg,
                    ));
                    out.add_comment(&format!("{}(): argument {} into register", fx_name, fx_arg));
                }
                out.instructions.push(Instruction::Call(fx_name.clone()));
                let target_offset = stack.get_offset_or_push(dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset.0),
                    MovArgument::Register(Register::RAX),
                ));
                out.add_comment(&format!("{}(): return value to stack", fx_name));
            }
            bytecode::Instruction::Label(name) => {
                out.instructions.push(Instruction::Label(name.clone()));
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
                let arg_a = resolve_move_argument(a, &mut stack, out);

                out.instructions.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));

                let arg_b = resolve_move_argument(b, &mut stack, out);
                out.instructions.push(Instruction::Cmp(Register::RAX, arg_b));
                out.instructions.push(Instruction::Sete(Register::AL));
                out.instructions.push(Instruction::Movzx(
                    Register::RAX,
                    MovArgument::Register(Register::AL),
                ));

                let target_offset = stack.get_offset_or_push(dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset.0),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::JumpNotEqual(to_label, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack, out);
                out.instructions.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                out.add_comment(&format!("jump: argument {} to register", a));
                let arg_b = resolve_move_argument(b, &mut stack, out);
                out.instructions.push(Instruction::Cmp(Register::RAX, arg_b));
                out.instructions.push(Instruction::Jne(to_label.clone()));
                out.add_comment(&format!("jump: if {} != {} then {}", a, b, to_label));
            }
            bytecode::Instruction::Noop => {
                out.instructions.push(Instruction::Nop);
            }
        }
    }

    let needs_stack_size = align_16(stack.size());

    for k in fn_body_starts_at_index..out.instructions.len() {
        let instr = &mut out.instructions[k];
        if let Instruction::Sub(Register::RSP, MovArgument::Integer(value)) = instr {
            *value = needs_stack_size;
        }
        if let Instruction::Add(Register::RSP, MovArgument::Integer(value)) = instr {
            *value = needs_stack_size;
        }
        if let Instruction::Function(_) = instr {
            break;
        }
    }

    return found_next_index;
}

fn emit_builtins() -> Vec<Instruction> {
    let os = OS::current();
    let mut out: Vec<Instruction> = Vec::new();

    let builtin_print: &[Instruction] = &[
        Instruction::Function("print".to_string()),
        Instruction::Push(Register::RBP),
        Instruction::Mov(
            MovArgument::Register(Register::RBP),
            MovArgument::Register(Register::RSP),
        ),
        Instruction::Sub(Register::RSP, MovArgument::Integer(32)),

        // put the string pointer on the stack. this thing
        // points to the string struct, where the first
        // 8 bytes are the length of the string and the
        // next 8 are a pointer to the character data.
        Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, -8),
            MovArgument::Register(Register::RDI)
        ),

        // length
        Instruction::Mov(
            MovArgument::Register(Register::R10),
            MovArgument::IndirectAddress(Register::RDI, 0)
        ),

        // pointer to data
        Instruction::Mov(
            MovArgument::Register(Register::R11),
            MovArgument::IndirectAddress(Register::RDI, 8)
        ),

        Instruction::Mov(
            MovArgument::Register(Register::RAX),
            MovArgument::Integer(os.syscall_print)
        ),
        Instruction::Mov(
            MovArgument::Register(Register::RDI),
            MovArgument::Integer(1)
        ),
        Instruction::Mov(
            MovArgument::Register(Register::RSI),
            MovArgument::Register(Register::R11)
        ),
        Instruction::Mov(
            MovArgument::Register(Register::RDX),
            MovArgument::Register(Register::R10)
        ),

        Instruction::Syscall,
        Instruction::Add(Register::RSP, MovArgument::Integer(32)),
        Instruction::Pop(Register::RBP),
        Instruction::Ret,
    ];

    out.extend_from_slice(builtin_print);

    return out;
}

fn emit_instructions(code: &str) -> Result<X86Assembly, Error> {
    let bytecode = bytecode::Bytecode::from_code(code)?;
    let mut out = X86Assembly {
        instructions: Vec::new(),
        comments: HashMap::new(),
        constants: Vec::new(),
    };

    out.instructions.push(Instruction::Directive(".intel_syntax noprefix".to_string()));
    out.instructions.push(Instruction::Directive(".globl main".to_string()));

    let mut builtins = emit_builtins();

    out.instructions.append(&mut builtins);

    let mut index: usize = 0;

    while let Some(instr) = bytecode.instructions.get(index) {
        match instr {
            bytecode::Instruction::Function(_, _) => {
                index = emit_function(&bytecode, index, &mut out);
            }
            _ => panic!(),
        };
    }

    return Ok(out);
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
