use crate::bytecode::{self, ENTRYPOINT_NAME};
use crate::bytecode::Instruction;
use crate::typer;
use crate::typer::Type;
use crate::util::Env;
use crate::util::{Error, Offset};
use crate::util::determine_stack_size_of_function;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

macro_rules! emit {
    ($asm:ident, $instr:literal) => {{
        ($asm).emit(&format!($instr))
    }};
    ($asm:ident, $instr:literal, $($arg:tt)*) => {{
        ($asm).emit(&format!($instr, $($arg)*))
    }};
}

struct Assembly<'a> {
    comments: HashMap<usize, String>,
    constants: Vec<bytecode::Const>,
    env: &'a Env,
    instructions: Vec<String>,
}

impl <'a> Assembly<'a> {
    fn add_comment<S: ToString>(&mut self, comment: S) {
        let index = self.instructions.len() - 1;
        self.comments.insert(index, comment.to_string());
    }

    fn emit(&mut self, instr: &str) {
        self.instructions.push(instr.to_string());
    }
}

impl <'a> std::fmt::Display for Assembly<'a> {
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
            let value = c.value.replace("\n", "\\n").replace("\t", "\\t");
            s.push_str(&format!("{}:\n", c.id));
            s.push_str(&format!("  .ascii \"{}\"\n", value));
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
struct X86StackOffset(i64);

impl std::fmt::Display for X86StackOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            return std::fmt::Result::Ok(());
        }
        let op = if self.0 < 0 { "-" } else { "+" };
        f.write_str(&format!("{}{}", op, self.0.abs()))
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

impl Into<Register> for &bytecode::Register {
    fn into(self) -> Register {
        let reg: Register = match *self {
            bytecode::REG_R0 => R8,
            bytecode::REG_R1 => R9,
            bytecode::REG_R2 => R10,
            bytecode::REG_R3 => R11,
            bytecode::REG_R4 => R12,
            bytecode::REG_R5 => R13,
            bytecode::REG_R6 => R14,
            bytecode::REG_RET => R15,
            _ => panic!("unknown register '{}'", self),
        };
        return reg;
    }
}

impl Into<X86StackOffset> for &Rc<bytecode::Memory> {
    fn into(self) -> X86StackOffset {
        let stack_offset = self.offset;

        // on x86 the stack grows downwards. this also means that for
        // struct types the first element will be stored at the lowest
        // address and so on.
        let offset = -stack_offset.0 - self.type_.size();

        return X86StackOffset(offset);
    }
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, asm: &mut Assembly) -> usize {
    let fx_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fx_instr {
        Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    emit!(asm, "{}:", fx_name);
    emit!(asm, "  push rbp");
    emit!(asm, "  mov rbp, rsp");

    let needs_stack_size = determine_stack_size_of_function(bc, at_index);

    emit!(asm, "  sub rsp, {}", needs_stack_size);

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let reg = INTEGER_ARGUMENT_REGISTERS[i];
        let off: X86StackOffset = fx_arg.into();

        match fx_arg.type_.size() {
            1 => emit!(asm, "  mov byte ptr [rbp{}], {}", off, reg),
            _ => emit!(asm, "  mov qword ptr [rbp{}], {}", off, reg),
        };
        
        asm.add_comment(format!("{} = {}", fx_arg, reg));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            Instruction::Add(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();

                emit!(asm, "  add {}, {}", r1, r2);
                asm.add_comment(format!("{} + {}", r1, r2));
            }
            Instruction::AddressOf(reg, mem) => {
                let reg: Register = reg.into();
                let off: X86StackOffset = mem.into();

                emit!(asm, "  lea {}, [rbp{}]", reg, off);
                asm.add_comment(format!("{} = &{}", reg, mem));
            }
            Instruction::AddressOfConst(reg, cons) => {
                let reg: Register = reg.into();
                emit!(asm, "  lea {}, [rip+{}]", reg, cons);
                asm.add_comment(format!("{} = &{}", reg, cons));
            }
            Instruction::Call(fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];
                    let off: X86StackOffset = fx_arg.into();

                    match fx_arg.type_.size() {
                        1 => emit!(asm, "  mov {}, byte ptr [rbp{}]", call_arg_reg, off),
                        _ => emit!(asm, "  mov {}, qword ptr [rbp{}]", call_arg_reg, off),
                    };
                }

                emit!(asm, "  call {}", fx_name);

                let call_arg_s = fx_args
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                asm.add_comment(format!("{}({})", fx_name, call_arg_s));
            }
            Instruction::Const(cons) => {
                asm.constants.push(cons.clone());
            }
            Instruction::Divide(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  mov rax, {}", r1);
                emit!(asm, "  cqo");
                emit!(asm, "  idiv {}", r2);
                emit!(asm, "  mov {}, rax", r1);
                asm.add_comment(format!("{} / {}", r1, r2));
            }
            Instruction::Equals(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  cmp {}, {}", r1, r2);
                emit!(asm, "  sete al");
                emit!(asm, "  movzx {}, al", r1);
                asm.add_comment(format!("{} = {} == {}", r1, r1, r2));
            }
            Instruction::Function(_, _) => {
                found_next_index = body_index;

                // this is the start of another function. we might encounter
                // several return statements and we must make sure to read
                // all of them. however, when we hit another function we
                // can surely stop.
                break;
            }
            Instruction::GreaterThan(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  cmp {}, {}", r1, r2);
                emit!(asm, "  setg al");
                emit!(asm, "  movzx {}, al", r1);
                asm.add_comment(format!("{} = {} > {}", r1, r1, r2));
            }
            Instruction::GreaterThanEquals(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  cmp {}, {}", r1, r2);
                emit!(asm, "  setge al");
                emit!(asm, "  movzx {}, al", r1);
                asm.add_comment(format!("{} = {} >= {}", r1, r1, r2));
            }
            Instruction::Jump(to_label) => {
                emit!(asm, "  jmp {}", to_label);
            }
            Instruction::JumpZero(to_label, reg) => {
                let reg: Register = reg.into();

                emit!(asm, "  cmp {}, 0", reg);
                emit!(asm, "  jz {}", to_label);
                asm.add_comment(format!("if {} == 0 jump {}", reg, to_label));
            }
            Instruction::Local(_) => {
                // we already know the stack size so no need to do anything here.
            }
            Instruction::StoreReg(addr, reg, type_) => {
                let r1: Register = (&addr.0).into();
                let reg: Register = reg.into();

                match type_.size() {
                    1 => emit!(asm, "  mov byte ptr [{}{}], {}", r1, addr.1, reg),
                    _ => emit!(asm, "  mov qword ptr [{}{}], {}", r1, addr.1, reg),
                };

                asm.add_comment(format!("{} = {}", addr, reg));
            }
            Instruction::StoreInt(addr, x) => {
                let r1: Register = (&addr.0).into();

                emit!(asm, "  mov qword ptr [{}{}], {}", r1, addr.1, x);
                asm.add_comment(format!("{} = {}", addr, x));
            }
            Instruction::Label(name) => {
                emit!(asm, "{}:", name);
            }
            Instruction::LessThan(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  cmp {}, {}", r1, r2);
                emit!(asm, "  setl al");
                emit!(asm, "  movzx {}, al", r1);
                asm.add_comment(format!("{} = {} < {}", r1, r1, r2));
            }
            Instruction::LessThanEquals(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "  cmp {}, {}", r1, r2);
                emit!(asm, "  setle al");
                emit!(asm, "  movzx {}, al", r1);
                asm.add_comment(format!("{} = {} <= {}", r1, r1, r2));
            }
            Instruction::LoadAddr(r1, addr, type_) => {
                let r1: Register = r1.into();
                let r2: Register = (&addr.0).into();

                match type_.size() {
                    1 => emit!(asm, "  mov {}, byte ptr [{}{}]", r1, r2, addr.1),
                    _ => emit!(asm, "  mov {}, qword ptr [{}{}]", r1, r2, addr.1),
                };
            }
            Instruction::LoadInt(reg, x) => {
                let reg: Register = reg.into();
                emit!(asm, "  mov {}, {}", reg, x);
            }
            Instruction::LoadMem(reg, mem) => {
                let reg: Register = reg.into();
                let off: X86StackOffset = mem.into();

                match mem.type_.size() {
                    1 => emit!(asm, "  mov {}, byte ptr [rbp{}]", reg, off),
                    _ => emit!(asm, "  mov {}, qword ptr [rbp{}]", reg, off),
                };
            }
            Instruction::LoadReg(r1, r2, type_) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                emit!(asm, "  mov {}, {}", r1, r2);
            }
            Instruction::Multiply(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();

                emit!(asm, "  imul {}, {}", r1, r2);
                asm.add_comment(format!("{} * {}", r1, r2));
            }
            Instruction::Not(r1) => {
                let r1: Register = r1.into();
                emit!(asm, "  not {}", r1);
                emit!(asm, "  and {}, 1", r1);
                asm.add_comment(format!("{} = !{}", r1, r1))
            }
            Instruction::Return => {
                let reg: Register = (&bytecode::REG_RET).into();

                if fx_name == ENTRYPOINT_NAME {
                    emit!(asm, "  mov rax, {}", asm.env.syscall_exit);
                    asm.add_comment("syscall: code exit");
                    emit!(asm, "  mov rdi, {}", reg);
                    emit!(asm, "  syscall");
                } else {
                    emit!(asm, "  mov rdi, {}", reg);
                }

                emit!(asm, "  add rsp, {}", needs_stack_size);
                emit!(asm, "  pop rbp");
                emit!(asm, "  ret");
            }
            Instruction::Subtract(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();

                emit!(asm, "  sub {}, {}", r1, r2);
                asm.add_comment(format!("{} - {}", r1, r2));
            }
        }
    }

    return found_next_index;
}

fn emit_builtins(asm: &mut Assembly) {
    emit!(asm, "print:");
    emit!(asm, "  push rbp");
    emit!(asm, "  mov rbp, rsp");
    emit!(asm, "  sub rsp, 16");

    // RDI is a pointer to a 'string' struct. the first 8 bytes holds
    // the length of the string and the next 8 bytes is a pointer to the
    // data segment.

    // length to stack
    emit!(asm, "  mov rax, qword ptr [rdi]");
    emit!(asm, "  mov qword ptr [rbp-8], rax");

    // data pointer to stack
    emit!(asm, "  mov rax, qword ptr [rdi+8]");
    emit!(asm, "  mov qword ptr [rbp-16], rax");

    emit!(asm, "  mov rax, {}", asm.env.syscall_print);
    emit!(asm, "  mov rdi, 1");
    emit!(asm, "  mov rsi, qword ptr [rbp-16]");
    emit!(asm, "  mov rdx, qword ptr [rbp-8]");
    emit!(asm, "  syscall");
    emit!(asm, "  add rsp, 16");
    emit!(asm, "  pop rbp");
    emit!(asm, "  ret");

    // an exit function!
    emit!(asm, "exit:");
    emit!(asm, "  push rbp");
    emit!(asm, "  mov rbp, rsp");
    emit!(asm, "  sub rsp, 16");
    emit!(asm, "  mov qword ptr [rbp-8], rdi");
    emit!(asm, "  mov rax, {}", asm.env.syscall_exit);
    emit!(asm, "  mov rdi, qword ptr [rbp-8]");
    emit!(asm, "  syscall");
    emit!(asm, "  add rsp, 16");
    emit!(asm, "  pop rbp");
    emit!(asm, "  ret");
}

pub fn emit_assembly(bc: &crate::bytecode::Bytecode, env: &Env) -> Result<String, Error> {
    let mut asm = Assembly {
        instructions: Vec::new(),
        comments: HashMap::new(),
        constants: Vec::new(),
        env: env,
    };

    emit!(asm, "  .intel_syntax noprefix");
    emit!(asm, "  .p2align 4, 0x90");
    emit!(asm, "  .globl {}", ENTRYPOINT_NAME);

    emit_builtins(&mut asm);

    let mut index: usize = 0;

    while let Some(instr) = bc.instructions.get(index) {
        match instr {
            Instruction::Function(_, _) => {
                index = emit_function(bc, index, &mut asm);
            }
            _ => panic!("{:?}", instr),
        };
    }

    return Ok(asm.to_string());
}
