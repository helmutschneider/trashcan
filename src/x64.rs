use std::collections::HashMap;
use std::io::Write;
use crate::bytecode;
use crate::typer;
use crate::util::Error;

#[derive(Debug)]
struct X86Assembly {
    instructions: Vec<Instruction>,
    comments: HashMap<usize, String>,
    constants: Vec<Const>,
}

#[derive(Debug)]
struct Const {
    id: i64,
    value: String,
}

impl Const {
    fn name_of(id: i64) -> String {
        return format!(".LC{}", id);
    }
}

impl X86Assembly {
    fn add_comment(&mut self, comment: &str) {
        let index = self.instructions.len() - 1;
        self.comments.insert(index, comment.to_string());
    }

    fn add_constant(&mut self, value: &str) -> i64 {
        let id = self.constants.len() as i64;
        let cons = Const {
            id: id,
            value: value.to_string(),
        };
        self.constants.push(cons);
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
            s.push_str(&format!("{}:\n", Const::name_of(c.id)));
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
    Const(i64),
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
            Self::Const(id) => {
                f.write_str(&format!("qword ptr [rip + {}]", Const::name_of(*id)))
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

fn index_to_stack_offset(stack_index: usize) -> i64 {
    return -((stack_index + 1) as i64) * 8;
}

fn get_stack_offset_or_push_var(stack: &mut HashMap<String, i64>, var: &bytecode::Variable) -> i64 {
    return match stack.get(&var.name) {
        Some(v) => *v,
        None => {
            let offset = index_to_stack_offset(stack.len());
            stack.insert(var.name.clone(), offset);
            offset
        }
    };
}

fn resolve_move_argument(
    arg: &bytecode::Argument,
    stack: &mut HashMap<String, i64>,
    out: &mut X86Assembly,
) -> MovArgument {
    let move_arg = match arg {
        bytecode::Argument::IntegerLiteral(i) => MovArgument::Integer(*i),
        bytecode::Argument::StringLiteral(s) => {
            let id = out.add_constant(&s);
            out.instructions.push(
                Instruction::Lea(Register::RAX, MovArgument::Const(id))
            );
            out.add_comment(&format!("load effective address of '{s}'"));
            MovArgument::Register(Register::RAX)
        }
        bytecode::Argument::Variable(v) => {
            let offset = get_stack_offset_or_push_var(stack, v);
            let to_arg = MovArgument::IndirectAddress(Register::RBP, offset);
            to_arg
        }
    };
    return move_arg;
}

fn maybe_emit_intermediate_move_for_copy(
    source: &bytecode::Argument,
    stack: &mut HashMap<String, i64>,
    out: &mut X86Assembly,
) -> MovArgument {
    // we can't move directly between two memory locations so
    // insert an intermediate step with RAX as a workaround.
    // currently we just mangle RAX which doesn't seem very
    // smart, but whatever.
    //   -johan, 2023-11-16
    if let bytecode::Argument::Variable(var) = source {
        let arg = resolve_move_argument(source, stack, out);
        let next_source = MovArgument::Register(Register::RAX);
        out.instructions.push(Instruction::Mov(next_source, arg));
        out.add_comment(&format!("intermediate copy of {var}"));
        return next_source;
    }

    return resolve_move_argument(source, stack, out);
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

    let mut stack: HashMap<String, i64> = HashMap::new();

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let stack_offset = get_stack_offset_or_push_var(&mut stack, fx_arg);

        out.instructions.push(Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, stack_offset),
            MovArgument::Register(INTEGER_ARGUMENT_REGISTERS[i]),
        ));
        out.add_comment(&format!("{}(): argument {} to stack", fx_name, fx_arg));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            bytecode::Instruction::Copy(dest_var, source) => {
                let source_arg = maybe_emit_intermediate_move_for_copy(source, &mut stack, out);
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    source_arg,
                ));
                out.add_comment(&format!("{} = {}", dest_var, source));
            }
            bytecode::Instruction::Local(var) => {

            }
            bytecode::Instruction::Store(var, offset, arg) => {
                
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
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
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
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
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
                let target_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
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

                let target_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
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
            bytecode::Instruction::Pointer(dest_var, a) => {
                println!("a = {}", a);

                let arg_a = resolve_move_argument(a, &mut stack, out);
                let target_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.instructions.push(Instruction::Lea(Register::RAX, arg_a));
                out.instructions.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
                    MovArgument::Register(Register::RAX)
                ));
            }
        }
    }

    let needs_stack_size = align_16((stack.len() as i64) * 8);

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
        Instruction::Sub(Register::RSP, MovArgument::Integer(16)),

        // put the string pointer on the stack.
        Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, -8),
            MovArgument::Register(Register::RDI)
        ),
        Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, -16),
            MovArgument::Register(Register::RSI)
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
            MovArgument::IndirectAddress(Register::RBP, -8)
        ),
        Instruction::Mov(
            MovArgument::Register(Register::RDX),
            MovArgument::IndirectAddress(Register::RBP, -16)
        ),

        Instruction::Syscall,
        Instruction::Add(Register::RSP, MovArgument::Integer(16)),
        Instruction::Pop(Register::RBP),
        Instruction::Ret,
    ];

    // out.extend_from_slice(builtin_print);

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
    fn should_call_print_and_write_to_stdout() {
        let code = r###"
        fun main(): int {
            var x = "hello!";
            print(&x);
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
