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

#[derive(Debug)]
pub struct Assembly<'a> {
    comments: HashMap<usize, String>,
    constants: Vec<bytecode::Const>,
    env: &'a Env,
    instructions: Vec<String>,
}

impl <'a> Assembly<'a> {
    fn add_comment(&mut self, comment: &str) {
        let index = self.instructions.len() - 1;
        self.comments.insert(index, comment.to_string());
    }

    fn emit(&mut self, instr: &str) {
        self.instructions.push(instr.to_string());
    }

    fn emit_instruction(&mut self, name: &'static str, args: &[String]) {
        let arg_s = args.join(", ");
        let instr = format!("  {} {}", name, arg_s);
        self.instructions.push(instr);
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

    fn add<A: Into<Register>, B: Into<InstructionArgument>>(&mut self, dest: A, source: B) {
        self.emit_instruction("add", &[dest.into().to_string(), source.into().to_string()]);
    }

    fn sub<A: Into<Register>, B: Into<InstructionArgument>>(&mut self, dest: A, source: B) {
        self.emit_instruction("sub", &[dest.into().to_string(), source.into().to_string()]);
    }

    fn imul<A: Into<Register>, B: Into<Register>>(&mut self, dest: A, source: B) {
        self.emit_instruction(
            "imul",
            &[dest.into().to_string(), source.into().to_string()],
        );
    }

    fn call(&mut self, name: &str) {
        self.emit_instruction("call", &[name.to_string()]);
    }

    fn syscall(&mut self) {
        self.emit_instruction("syscall", &[]);
    }

    fn jmp(&mut self, to_label: &str) {
        self.emit_instruction("jmp", &[to_label.to_string()]);
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
        f.write_str(&format!(" {} {}", op, self.0.abs()))
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
    Indirect(Register, X86StackOffset),
    Constant(bytecode::ConstId),
}

impl std::fmt::Display for InstructionArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Self::Immediate(x) => x.fmt(f),
            Self::Register(reg) => reg.fmt(f),
            Self::Indirect(reg, offset) => f.write_str(&format!("qword ptr [{}{}]", reg, offset)),
            Self::Constant(id) => f.write_str(&format!("qword ptr [{}+{}]", RIP, id)),
        };
    }
}

impl Into<InstructionArgument> for Register {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Register(self);
    }
}

impl Into<InstructionArgument> for i64 {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Immediate(self);
    }
}

impl Into<InstructionArgument> for bytecode::ConstId {
    fn into(self) -> InstructionArgument {
        return InstructionArgument::Constant(self);
    }
}

impl Into<InstructionArgument> for &bytecode::Register {
    fn into(self) -> InstructionArgument {
        let reg: Register = self.into();
        return InstructionArgument::Register(reg);
    }
}

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

impl Into<X86StackOffset> for i64 {
    fn into(self) -> X86StackOffset {
        return X86StackOffset(self);
    }
}

impl Into<X86StackOffset> for Offset {
    fn into(self) -> X86StackOffset {
        return X86StackOffset(self.0);
    }
}

fn indirect<T: Into<X86StackOffset>>(register: Register, offset: T) -> InstructionArgument {
    return InstructionArgument::Indirect(register, offset.into());
}

fn emit_function(bc: &bytecode::Bytecode, at_index: usize, asm: &mut Assembly) -> usize {
    let fx_instr = &bc.instructions[at_index];
    let (fx_name, fx_args) = match fx_instr {
        Instruction::Function(a, b) => (a, b),
        _ => panic!("Expected function instruction, got: {}.", fx_instr),
    };

    emit!(asm, "{}:", fx_name);
    asm.push(RBP);
    asm.mov(RBP, RSP);

    let needs_stack_size = determine_stack_size_of_function(bc, at_index);

    asm.sub(RSP, needs_stack_size);

    for i in 0..fx_args.len() {
        let fx_arg = &fx_args[i];
        let reg = INTEGER_ARGUMENT_REGISTERS[i];

        asm.mov(indirect(RBP, fx_arg), reg);
        asm.add_comment(&format!("{} = {}", fx_arg, reg));
    }

    let mut found_next_index = bc.instructions.len();

    for body_index in (at_index + 1)..bc.instructions.len() {
        let instr = &bc.instructions[body_index];

        match instr {
            Instruction::Local(_) => {
                // we already know the stack size so no need to do anything here.
            }
            Instruction::StoreReg(addr, reg) => {
                let r1: Register = (&addr.0).into();
                asm.mov(indirect(r1, addr.1), reg);
                asm.add_comment(&format!("{} = {}", addr, reg));
            }
            Instruction::StoreInt(addr, x) => {
                let r1: Register = (&addr.0).into();
                asm.mov(indirect(r1, addr.1), *x);
                asm.add_comment(&format!("{} = {}", addr, x));
            }
            Instruction::LoadMem(reg, mem) => {
                asm.mov(reg, indirect(RBP, mem));
            }
            Instruction::LoadInt(reg, x) => {
                asm.mov(reg, *x);
            }
            Instruction::LoadAddr(r1, addr) => {
                let r2: Register = (&addr.0).into();
                asm.mov(r1, indirect(r2, addr.1));
            }
            Instruction::LoadReg(r1, r2) => {
                asm.mov(r1, r2);
            }
            Instruction::AddrOf(reg, mem) => {
                let reg: Register = reg.into();
                let off: X86StackOffset = mem.into();

                emit!(asm, "lea {}, [rbp{}]", reg, off);
                asm.add_comment(&format!("{} = &{}", reg, mem));
            }
            Instruction::AddrOfConst(reg, cons) => {
                let reg: Register = reg.into();
                emit!(asm, "lea {}, [rip+{}]", reg, cons);
                asm.add_comment(&format!("{} = &{}", reg, cons));
            }
            Instruction::Return => {
                if fx_name == ENTRYPOINT_NAME {
                    asm.mov(RAX, asm.env.syscall_exit);
                    asm.add_comment("syscall: code exit");

                    asm.mov(RDI, &bytecode::REG_RET);

                    asm.syscall();
                } else {
                    asm.mov(RAX, &bytecode::REG_RET);
                }

                // will be updated later with the correct stacks size when we exit the function body.
                asm.add(RSP, needs_stack_size);
                asm.pop(RBP);
                asm.ret();
            }
            Instruction::Add(r1, r2) => {
                asm.add(r1, r2);
                asm.add_comment(&format!("{} + {}", r1, r2));
            }
            Instruction::Sub(r1, r2) => {
                asm.sub(r1, r2);
                asm.add_comment(&format!("{} - {}", r1, r2));
            }
            Instruction::Mul(r1, r2) => {
                asm.imul(r1, r2);
                asm.add_comment(&format!("{} * {}", r1, r2));
            }
            Instruction::Div(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();

                emit!(asm, "mov rax, {}", r1);
                emit!(asm, "cqo");
                emit!(asm, "idiv {}", r2);
                emit!(asm, "mov {}, rax", r1);
                asm.add_comment(&format!("{} / {}", r1, r2));
            }
            Instruction::Call(fx_name, fx_args) => {
                for i in 0..fx_args.len() {
                    let fx_arg = &fx_args[i];
                    let call_arg_reg = INTEGER_ARGUMENT_REGISTERS[i];
                    asm.mov(call_arg_reg, indirect(RBP, fx_arg));
                }
                asm.call(fx_name);

                let call_arg_s = fx_args
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                asm.add_comment(&format!("{}({})", fx_name, call_arg_s));
            }
            Instruction::Label(name) => {
                emit!(asm, "{}:", name);
            }
            Instruction::Function(_, _) => {
                found_next_index = body_index;

                // this is the start of another function. we might encounter
                // several return statements and we must make sure to read
                // all of them. however, when we hit another function we
                // can surely stop.
                break;
            }
            Instruction::IsEqual(r1, r2) => {
                let r1: Register = r1.into();
                let r2: Register = r2.into();
                
                emit!(asm, "cmp {}, {}", r1, r2);
                emit!(asm, "sete al");
                emit!(asm, "movzx {}, al", r1);
                asm.add_comment(&format!("{} = {} == {}", r1, r1, r2));
            }
            Instruction::Jump(to_label) => {
                asm.jmp(&to_label);
            }
            Instruction::JumpZero(to_label, reg) => {
                let reg: Register = reg.into();

                emit!(asm, "cmp {}, 0", reg);
                emit!(asm, "jz {}", to_label);
                asm.add_comment(&format!("if {} == 0 jump {}", reg, to_label));
            }
            Instruction::Const(cons) => {
                asm.constants.push(cons.clone());
            }
        }
    }

    return found_next_index;
}

fn emit_builtins(asm: &mut Assembly) {
    emit!(asm, "print:");
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

    asm.mov(RAX, asm.env.syscall_print);
    asm.mov(RDI, 1);
    asm.mov(RSI, indirect(RBP, -16));
    asm.mov(RDX, indirect(RBP, -8));

    asm.syscall();
    asm.add(RSP, 16);
    asm.pop(RBP);
    asm.ret();

    // an exit function!
    emit!(asm, "exit:");
    asm.push(RBP);
    asm.mov(RBP, RSP);

    asm.sub(RSP, 16);
    asm.mov(indirect(RBP, -8), RDI);

    asm.mov(RAX, asm.env.syscall_exit);
    asm.mov(RDI, indirect(RBP, -8));

    asm.syscall();
    asm.add(RSP, 16);
    asm.pop(RBP);
    asm.ret();
}

pub fn emit_assembly(bc: &crate::bytecode::Bytecode, env: &Env) -> Result<String, Error> {
    let mut asm = Assembly {
        instructions: Vec::new(),
        comments: HashMap::new(),
        constants: Vec::new(),
        env: env,
    };

    emit!(asm, ".intel_syntax noprefix");
    emit!(asm, ".globl {}", ENTRYPOINT_NAME);

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
