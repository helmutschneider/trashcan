use std::collections::HashMap;
use std::{fmt::format, io::Write};

use crate::bytecode::LoadArgument;
use crate::{ast, bytecode};

const HELLO_WORLD: &'static str = r###"
.intel_syntax noprefix
.globl _start

_start:
  push rbp
  mov rbp, rsp
  
  mov rax, 0x2000001
  mov rdi, 69
  syscall

  pop rbp
  ret
"###;

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

#[derive(Debug, Clone, Copy)]
enum AddArgument {
    Register(Register),
    Integer(i64),
}

#[derive(Debug, Clone)]
enum Instruction {
    Directive(String),
    Function(String),
    Push(Register),
    Pop(Register),
    Ret,
    Mov(MovArgument, MovArgument),
    Add(Register, MovArgument),
    Call(String),
    Syscall,
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
            Self::Add(dest, source) => format!("  add {}, {}", dest, source),
            Self::Call(fx) => format!("  call {}", fx),
            Self::Syscall => "  syscall".to_string(),
            // _ => panic!("Unknown instruction '{}'.", self),
        };
        return f.write_str(&s);
    }
}

fn call_assembler_and_emit_binary(asm: &str) -> String {
    let mut child = std::process::Command::new("gcc")
        .args([
            "-arch",
            "x86_64",
            "-masm=intel",
            "-x",
            "assembler",
            "-o",
            "app",
            "-nostartfiles",
            "-e",
            "main",
            "-",
        ])
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

fn index_to_stack_offset(index: usize) -> i64 {
    return -((index + 1) as i64) * 8
}

fn resolve_stack_variable_offset(stack: &mut Vec<String>, name: &str) -> i64 {
    for i in 0..stack.len() {
        if name == stack[i] {
            return index_to_stack_offset(i);
        }
    }
    stack.push(name.to_string());
    return index_to_stack_offset(stack.len() - 1);
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, out: &mut Vec<Instruction>) -> usize {
    let fun_instr = &bc.instructions[at_index];
    let (fx_sym, fx_arg_registers) = match fun_instr {
        bytecode::Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fun_instr),
    };

    out.push(Instruction::Function(fx_sym.name.clone()));
    out.push(Instruction::Push(Register::RBP));
    out.push(Instruction::Mov(
        MovArgument::Register(Register::RBP),
        MovArgument::Register(Register::RSP),
    ));

    let mut body_index = at_index + 1;
    let mut stack: Vec<String> = Vec::new();

    for i in 0..fx_arg_registers.len() {
        let arg_reg = fx_arg_registers[i];
        let stack_offset = resolve_stack_variable_offset(&mut stack, &arg_reg.to_string());

        out.push(Instruction::Mov(
            MovArgument::IndirectAddress(Register::RBP, stack_offset),
            MovArgument::Register(INTEGER_ARGUMENT_REGISTERS[i]),
        ));
    }

    while let Some(instr) = bc.instructions.get(body_index) {
        match instr {
            bytecode::Instruction::Load(dest_reg, load_arg) => {
                let dest_offset = resolve_stack_variable_offset(&mut stack, &dest_reg.to_string());
                let source_arg = match load_arg {
                    LoadArgument::Integer(x) => MovArgument::Integer(*x),
                    LoadArgument::Register(reg) => {
                        let source_offset =
                            resolve_stack_variable_offset(&mut stack, &reg.to_string());
                        out.push(Instruction::Mov(
                            MovArgument::Register(Register::RAX),
                            MovArgument::IndirectAddress(Register::RBP, source_offset),
                        ));
                        MovArgument::Register(Register::RAX)
                    }
                    LoadArgument::Symbol(sym) => {
                        let source_offset = resolve_stack_variable_offset(&mut stack, &sym.name);
                        out.push(Instruction::Mov(
                            MovArgument::Register(Register::RAX),
                            MovArgument::IndirectAddress(Register::RBP, source_offset),
                        ));
                        MovArgument::Register(Register::RAX)
                    }
                };

                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    source_arg,
                ));
                body_index += 1;
            }
            bytecode::Instruction::Store(sym, source_reg) => {
                let source_offset =
                    resolve_stack_variable_offset(&mut stack, &source_reg.to_string());
                let dest_offset = resolve_stack_variable_offset(&mut stack, &sym.name);
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    MovArgument::IndirectAddress(Register::RBP, source_offset),
                ));
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    MovArgument::Register(Register::RAX),
                ));
                body_index += 1;
            }
            bytecode::Instruction::Return(ret_reg) => {
                let offset = resolve_stack_variable_offset(&mut stack, &ret_reg.to_string());
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    MovArgument::IndirectAddress(Register::RBP, offset),
                ));

                if fx_sym.name == "main" {
                    // macos syscall exit().
                    // https://opensource.apple.com/source/xnu/xnu-1504.3.12/bsd/kern/syscalls.master
                    // https://stackoverflow.com/questions/48845697/macos-64-bit-system-call-table
                    out.push(Instruction::Mov(
                        MovArgument::Register(Register::RAX),
                        MovArgument::Integer(0x2000000 + 1)
                    ));
                    out.push(Instruction::Mov(
                        MovArgument::Register(Register::RDI),
                        MovArgument::IndirectAddress(Register::RBP, offset)
                    ));
                    out.push(Instruction::Syscall);
                }

                out.push(Instruction::Pop(Register::RBP));
                out.push(Instruction::Ret);
                break;
            }
            bytecode::Instruction::Add(dest_reg, a, b) => {
                let offset_a = resolve_stack_variable_offset(&mut stack, &a.to_string());
                out.push(Instruction::Mov(
                    MovArgument::Register(Register::RAX),
                    MovArgument::IndirectAddress(Register::RBP, offset_a)
                ));
                let offset_b = resolve_stack_variable_offset(&mut stack, &b.to_string());

                let instr = Instruction::Add(
                    Register::RAX,
                    MovArgument::IndirectAddress(Register::RBP, offset_b)
                );
                out.push(instr);

                let dest_offset = resolve_stack_variable_offset(&mut stack, &dest_reg.to_string());
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, dest_offset),
                    MovArgument::Register(Register::RAX)
                ));

                body_index += 1;
            }
            bytecode::Instruction::FunctionCall(target_reg, fx_name, arg_registers) => {
                for i in 0..arg_registers.len() {
                    let arg_source_reg = arg_registers[i];
                    let arg_offset =
                        resolve_stack_variable_offset(&mut stack, &arg_source_reg.to_string());
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];

                    out.push(Instruction::Mov(
                        MovArgument::Register(call_arg_reg),
                        MovArgument::IndirectAddress(Register::RBP, arg_offset),
                    ));
                }
                out.push(Instruction::Call(fx_name.name.clone()));

                let target_offset =
                    resolve_stack_variable_offset(&mut stack, &target_reg.to_string());
                out.push(Instruction::Mov(
                    MovArgument::IndirectAddress(Register::RBP, target_offset),
                    MovArgument::Register(Register::RAX),
                ));

                body_index += 1;
            }
            _ => {
                body_index += 1;
            }
        }
    }

    return body_index + 1;
}

fn emit_instructions(code: &str) -> Vec<Instruction> {
    let bytecode = bytecode::from_code(code);

    println!("{bytecode}");

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
    use crate::x64::*;

    #[test]
    fn do_thing() {
        let s = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }
            fun main(): int {
                return add(1, 3);
            }
        "###;
        let asm = emit_assembly(s);
        call_assembler_and_emit_binary(&asm);

        println!("{asm}");
        assert!(false);
    }
}
