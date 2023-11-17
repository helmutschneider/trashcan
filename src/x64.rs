use std::collections::HashMap;
use std::io::Write;
use crate::bytecode;

#[derive(Debug, Clone, Copy)]
enum OSKind {
    Linux,
    MacOS,
    Windows,
}

#[derive(Debug, Clone, Copy)]
struct OS {
    kind: OSKind,
    exit_syscall: i64,
}

impl OS {
    fn current() -> Self {
        let os_name = std::env::consts::OS;
        return match os_name {
            "macos" => OS {
                kind: OSKind::MacOS,
                // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
                // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
                exit_syscall: 0x2000000 + 1,
            },
            "linux" => OS {
                kind: OSKind::Linux,
                // https://filippo.io/linux-syscall-table/
                exit_syscall: 60,
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
}

impl std::fmt::Display for MovArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Register(reg) => reg.fmt(f),
            Self::Integer(x) => x.fmt(f),
            Self::IndirectAddress(reg, offset) => {
                let op = if *offset > 0 { "+" } else { "-" };
                f.write_str(&format!("qword ptr [{} {} {}]", reg, op, offset.abs()))
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
    return match stack.get(&var.0) {
        Some(v) => *v,
        None => {
            let offset = index_to_stack_offset(stack.len());
            stack.insert(var.0.clone(), offset);
            offset
        }
    };
}

fn resolve_move_argument(
    arg: &bytecode::Argument,
    stack: &mut HashMap<String, i64>,
) -> MovArgument {
    let move_arg = match arg {
        bytecode::Argument::Integer(x) => MovArgument::Integer(*x),
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
    out: &mut Vec<Instruction>,
) -> MovArgument {
    // we can't move directly between two memory locations so
    // insert an intermediate step with RAX as a workaround.
    // currently we just mangle RAX which doesn't seem very
    // smart, but whatever.
    //   -johan, 2023-11-16
    if let bytecode::Argument::Variable(_) = source {
        let arg = resolve_move_argument(source, stack);
        let next_source = MovArgument::Register(Register::RAX);
        out.push(Instruction::Mov(next_source, arg));
        return next_source;
    }
    return resolve_move_argument(source, stack);
}

fn align_16(value: i64) -> i64 {
    let mul = (value as f64) / 16.0;
    return (mul.ceil() as i64) * 16;
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, out: &mut Vec<Instruction>) -> usize {
    let fun_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fun_instr {
        bytecode::Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fun_instr),
    };

    // used by the loop below to find the add/sub RSP instructions.
    let fn_body_starts_at_index = out.len() + 1;

    out.push(Instruction::Function(fx_name.clone()));
    out.push(Instruction::Push(Register::RBP));
    out.push(Instruction::Mov(
        MovArgument::Register(Register::RBP),
        MovArgument::Register(Register::RSP),
    ));

    // will be updated later with the correct stack size when we exit the function body.
    let needs_stack_size = MovArgument::Integer(-42069);
    out.push(Instruction::Sub(Register::RSP, needs_stack_size));

    let mut stack: HashMap<String, i64> = HashMap::new();

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let stack_offset = get_stack_offset_or_push_var(&mut stack, fx_arg);

        out.push(Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, stack_offset),
            MovArgument::Register(INTEGER_ARGUMENT_REGISTERS[i]),
        ));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            bytecode::Instruction::Copy(dest_var, source) => {
                let source_arg = maybe_emit_intermediate_move_for_copy(source, &mut stack, out);
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    source_arg,
                ));
            }
            bytecode::Instruction::Ret(ret_arg) => {
                let source_arg = resolve_move_argument(ret_arg, &mut stack);
                let os = OS::current();

                if fx_name == "main" {
                    out.push(Instruction::Mov(
                        MovArgument::Register(Register::RAX),
                        MovArgument::Integer(os.exit_syscall),
                    ));
                    out.push(Instruction::Mov(
                        MovArgument::Register(Register::RDI),
                        source_arg,
                    ));
                    out.push(Instruction::Syscall);
                } else {
                    out.push(Instruction::Mov(
                        MovArgument::Register(Register::RAX),
                        source_arg,
                    ));
                }

                // will be updated later with the correct stacks size when we exit the function body.
                out.push(Instruction::Add(Register::RSP, needs_stack_size));
                out.push(Instruction::Pop(Register::RBP));
                out.push(Instruction::Ret);
            }
            bytecode::Instruction::Add(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                let arg_b = resolve_move_argument(b, &mut stack);
                out.push(Instruction::Add(Register::RAX, arg_b));
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::Sub(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                let arg_b = resolve_move_argument(b, &mut stack);
                out.push(Instruction::Sub(Register::RAX, arg_b));
                let dest_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::Call(dest_var, fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_move_arg = resolve_move_argument(&fx_arg, &mut stack);
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];

                    out.push(Instruction::Mov(
                        MovArgument::Register(call_arg_reg),
                        call_move_arg,
                    ));
                }
                out.push(Instruction::Call(fx_name.clone()));

                let target_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::Label(name) => {
                out.push(Instruction::Label(name.clone()));
            }
            bytecode::Instruction::Function(_, _) => {
                found_next_index = body_index;

                // this is the start of another function. we might encounter
                // several return statements and we must make sure to read
                // all of them. however, when we hit another function we
                // can surely stop.
                break;
            }
            bytecode::Instruction::Eq(dest_var, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);

                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));

                let arg_b = resolve_move_argument(b, &mut stack);
                out.push(Instruction::Cmp(Register::RAX, arg_b));
                out.push(Instruction::Sete(Register::AL));
                out.push(Instruction::Movzx(
                    Register::RAX,
                    MovArgument::Register(Register::AL),
                ));

                let target_offset = get_stack_offset_or_push_var(&mut stack, dest_var);
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
                    MovArgument::Register(Register::RAX),
                ));
            }
            bytecode::Instruction::Jne(to_label, a, b) => {
                let arg_a = resolve_move_argument(a, &mut stack);
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    arg_a,
                ));
                let arg_b = resolve_move_argument(b, &mut stack);
                out.push(Instruction::Cmp(Register::RAX, arg_b));
                out.push(Instruction::Jne(to_label.clone()));
            }
            bytecode::Instruction::Noop => {
                out.push(Instruction::Nop);
            }
        }
    }

    let needs_stack_size = align_16((stack.len() as i64) * 8);

    for k in fn_body_starts_at_index..out.len() {
        let instr = &mut out[k];
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

fn emit_instructions(code: &str) -> Vec<Instruction> {
    let bytecode = bytecode::from_code(code);
    let mut out: Vec<Instruction> = Vec::new();

    out.push(Instruction::Directive(".intel_syntax noprefix".to_string()));
    out.push(Instruction::Directive(".globl main".to_string()));

    let mut index: usize = 0;

    while let Some(instr) = bytecode.instructions.get(index) {
        match instr {
            bytecode::Instruction::Function(_, _) => {
                index = emit_function(&bytecode, index, &mut out);
            }
            _ => panic!(),
        };
    }

    return out;
}

pub fn emit_assembly(code: &str) -> String {
    let instructions = emit_instructions(code);
    let mut str = String::with_capacity(1024);

    for instr in &instructions {
        str.push_str(&format!("{instr}\n"));
    }

    return str;
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

    fn do_test(expected: i32, code: &str) {
        let asm = emit_assembly(code);
        let bin_name = format!("_test_{}.out", random_str(8));

        emit_binary(&asm, &bin_name);

        let status = std::process::Command::new(format!("./{bin_name}"))
            .spawn()
            .unwrap()
            .wait()
            .unwrap()
            .code()
            .unwrap();

        std::process::Command::new("rm")
            .args(["-rf", &bin_name])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        assert_eq!(expected, status);
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
