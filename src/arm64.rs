use crate::bytecode;
use crate::bytecode::Bytecode;
use crate::bytecode::Const;
use crate::bytecode::Instruction;
use crate::bytecode::ENTRYPOINT_NAME;
use crate::util::determine_stack_size_of_function;
use crate::util::Env;
use crate::util::Error;

macro_rules! emit {
    ($asm:ident, $instr:literal) => {{
        ($asm).emit(&format!($instr))
    }};
    ($asm:ident, $instr:literal, $($arg:tt)*) => {{
        ($asm).emit(&format!($instr, $($arg)*))
    }};
}

struct ARM64Assembly<'a> {
    bytecode: &'a Bytecode,
    constants: Vec<Const>,
    env: &'a Env,
    instructions: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Register(&'static str);

const X0: Register = Register("x0");
const X1: Register = Register("x1");
const X2: Register = Register("x2");
const X3: Register = Register("x3");
const X4: Register = Register("x4");
const X5: Register = Register("x5");
const X6: Register = Register("x6");
const X7: Register = Register("x7");
const X8: Register = Register("x8");
const X9: Register = Register("x9");
const X10: Register = Register("x10");
const X11: Register = Register("x11");
const X12: Register = Register("x12");
const X13: Register = Register("x13");
const X14: Register = Register("x14");

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return self.0.fmt(f);
    }
}

impl Into<Register> for &crate::bytecode::Register {
    fn into(self) -> Register {
        return match *self {
            bytecode::REG_R0 => X8,
            bytecode::REG_R1 => X9,
            bytecode::REG_R2 => X10,
            bytecode::REG_R3 => X11,
            bytecode::REG_R4 => X12,
            bytecode::REG_R5 => X13,
            bytecode::REG_R6 => X14,
            bytecode::REG_RET => X0,
            _ => panic!("bad"),
        };
    }
}

const CALL_REGISTERS: &[Register] = &[X0, X1, X2, X3, X4, X5, X6, X7];

fn get_stack_offset(stack_size: i64, mem: &bytecode::Memory) -> i64 {
    return stack_size - mem.offset.0 - mem.type_.size();
}

impl<'a> ARM64Assembly<'a> {
    fn new(bc: &'a Bytecode, env: &'a Env) -> Self {
        return Self {
            bytecode: bc,
            constants: Vec::new(),
            env: env,
            instructions: Vec::new(),
        };
    }

    fn emit(&mut self, instr: &str) {
        self.instructions.push(instr.to_string());
    }

    fn emit_mov_immediate(&mut self, to_reg: Register, value: i64) {
        // ARM64 is a fixed-width ISA where each instruction is 32
        // bits long. this means that we can't fit a 64-bit immediate
        // value into a single instruction.

        let value = value as usize;
        emit!(self, "  mov {}, #{}", to_reg, value & 0xffff);
        for shl in [16, 32, 48] {
            let x = (value >> shl) & 0xffff;
            if x != 0 {
                emit!(self, "  movk {}, #{}, lsl #{}", to_reg, x, shl);
            }
        }
    }

    fn emit_builtins(&mut self) {
        emit!(self, "print:");
        emit!(self, "  sub sp, sp, #16");
        emit!(self, "  str x0, [sp, #8]");
        self.emit_mov_immediate(Register(self.env.syscall_register), self.env.syscall_print);
        emit!(self, "  mov x0, #1");
        emit!(self, "  ldr x1, [sp, #8]");
        emit!(self, "  ldr x1, [x1, #8]");
        emit!(self, "  ldr x2, [sp, #8]");
        emit!(self, "  ldr x2, [x2]");
        emit!(self, "  svc #0");
        emit!(self, "  add sp, sp, #16");
        emit!(self, "  ret");

        emit!(self, "exit:");
        emit!(self, "  sub sp, sp, #16");
        emit!(self, "  str x0, [sp, #8]");
        self.emit_mov_immediate(Register(self.env.syscall_register), self.env.syscall_exit);
        emit!(self, "  ldr x0, [sp, #8]");
        emit!(self, "  svc #0");
        emit!(self, "  add sp, sp, #16");
        emit!(self, "  ret");
    }

    fn emit_function(&mut self, index: usize) -> usize {
        let (fx_name, fx_args) = match &self.bytecode.instructions[index] {
            Instruction::Function(name, args) => (name, args),
            _ => panic!(),
        };

        let stack_size = determine_stack_size_of_function(self.bytecode, index)
            // reserve space for x29 and x30, which are the link register and frame pointer.
            + 16;

        emit!(self, "{}:", fx_name);
        emit!(self, "  sub sp, sp, #{}", stack_size);

        // link register and frame pointers. we need to store these so we can call functions
        // using the 'bl' instruction.
        emit!(self, "  stp x29, x30, [sp, #0]");

        for k in 0..fx_args.len() {
            let fx_arg = &fx_args[k];
            let reg = CALL_REGISTERS[k];
            let offset = get_stack_offset(stack_size, &fx_arg);
            emit!(self, "  str {}, [sp, #{}]", reg, offset);
        }

        let len = self.bytecode.instructions.len();
        let mut next_index: usize = len;

        for k in (index + 1)..len {
            let instr = &self.bytecode.instructions[k];

            if let Instruction::Function(_, _) = instr {
                next_index = k;
                break;
            }

            match instr {
                Instruction::Add(reg, r1) => {
                    let reg: Register = reg.into();
                    let r1: Register = r1.into();
                    emit!(self, "  add {}, {}, {}", reg, reg, r1);
                }
                Instruction::AddrOf(reg, mem) => {
                    let reg: Register = reg.into();
                    let offset = get_stack_offset(stack_size, mem);
                    emit!(self, "  add {}, sp, #{}", reg, offset);
                }
                Instruction::AddrOfConst(reg, cons) => {
                    let reg: Register = reg.into();
                    emit!(self, "  adr {}, {}", reg, cons);
                }
                Instruction::Call(name, args) => {
                    for k in 0..args.len() {
                        let arg = &args[k];
                        let to_reg: Register = CALL_REGISTERS[k].into();
                        let offset = get_stack_offset(stack_size, arg);
                        emit!(self, "  ldr {}, [sp, #{}]", to_reg, offset);
                    }

                    emit!(self, "  bl {}", name);
                }
                Instruction::Const(c) => {
                    self.constants.push(c.clone());
                }
                Instruction::Div(reg, r1) => {
                    let reg: Register = reg.into();
                    let r1: Register = r1.into();
                    emit!(self, "  sdiv {}, {}, {}", reg, reg, r1);
                }
                Instruction::Function(_, _) => panic!("got function!"),
                Instruction::Equals(r1, r2) => {
                    let r1: Register = r1.into();
                    let r2: Register = r2.into();

                    // https://developer.arm.com/documentation/den0042/a/Unified-Assembly-Language-Instructions/Instruction-set-basics/Conditional-execution?lang=en#CHDBEIHD
                    emit!(self, "  cmp {}, {}", r1, r2);
                    emit!(self, "  cset {}, eq", r1);
                }
                Instruction::Jump(to_label) => {
                    emit!(self, "  b {}", to_label);
                }
                Instruction::JumpZero(to_label, reg) => {
                    let reg: Register = reg.into();
                    emit!(self, "  cbz {}, {}", reg, to_label);
                }
                Instruction::Label(name) => {
                    emit!(self, "{}:", name);
                }
                Instruction::LessThan(r1, r2) => {
                    let r1: Register = r1.into();
                    let r2: Register = r2.into();

                    // https://developer.arm.com/documentation/den0042/a/Unified-Assembly-Language-Instructions/Instruction-set-basics/Conditional-execution?lang=en#CHDBEIHD
                    emit!(self, "  cmp {}, {}", r1, r2);
                    emit!(self, "  cset {}, lt", r1);
                }
                Instruction::LoadAddr(reg, addr) => {
                    let dest_reg: Register = reg.into();
                    let source_reg: Register = (&addr.0).into();
                    emit!(self, "  ldr {}, [{}, #{}]", dest_reg, source_reg, addr.1 .0);
                }
                Instruction::LoadInt(reg, x) => {
                    let reg: Register = reg.into();
                    self.emit_mov_immediate(reg, *x);
                }
                Instruction::LoadMem(reg, mem) => {
                    let reg: Register = reg.into();
                    let offset = get_stack_offset(stack_size, mem);
                    emit!(self, "  ldr {}, [sp, #{}]", reg, offset)
                }
                Instruction::LoadReg(reg, r1) => {
                    let reg: Register = reg.into();
                    let r1: Register = r1.into();
                    emit!(self, "  mov {}, {}", reg, r1);
                }
                Instruction::Local(_) => {
                    // do nothing.
                }
                Instruction::Mul(reg, r1) => {
                    let reg: Register = reg.into();
                    let r1: Register = r1.into();
                    emit!(self, "  mul {}, {}, {}", reg, reg, r1);
                }
                Instruction::Return => {
                    if fx_name == ENTRYPOINT_NAME {
                        // do an implicit exit syscall.
                        self.emit_mov_immediate(
                            Register(self.env.syscall_register),
                            self.env.syscall_exit,
                        );
                        emit!(self, "  mov x0, #0");
                        emit!(self, "  svc #0");
                    }

                    let reg: Register = (&bytecode::REG_RET).into();
                    emit!(self, "  mov x0, {}", reg);
                    emit!(self, "  ldp x29, x30, [sp, #0]");
                    emit!(self, "  add sp, sp, #{}", stack_size);
                    emit!(self, "  ret");
                }
                Instruction::StoreInt(addr, x) => {
                    let dest_reg: Register = (&addr.0).into();
                    self.emit_mov_immediate(X0, *x);
                    emit!(self, "  str x0, [{}, #{}]", dest_reg, addr.1 .0);
                }
                Instruction::StoreReg(addr, reg) => {
                    let dest_reg: Register = (&addr.0).into();
                    let reg: Register = reg.into();
                    emit!(self, "  str {}, [{}, #{}]", reg, dest_reg, addr.1 .0);
                }
                Instruction::Sub(reg, r1) => {
                    let reg: Register = reg.into();
                    let r1: Register = r1.into();
                    emit!(self, "  sub {}, {}, {}", reg, reg, r1);
                }
            }
        }

        return next_index;
    }
}

impl std::fmt::Display for ARM64Assembly<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::with_capacity(8192);

        for instr in &self.instructions {
            out.push_str(instr);
            out.push_str("\n");
        }

        for cons in &self.constants {
            out.push_str(&format!("{}:\n", cons.id));

            let escaped = cons.value.replace("\n", "\\n").replace("\t", "\\t");

            out.push_str(&format!("  .ascii \"{}\"\n", escaped));
        }

        return out.fmt(f);
    }
}

pub fn emit_assembly(bc: &Bytecode, env: &Env) -> Result<String, Error> {
    let mut asm = ARM64Assembly::new(bc, env);
    emit!(asm, ".globl {}", ENTRYPOINT_NAME);
    emit!(asm, ".p2align 2");
    asm.emit_builtins();

    let mut index: usize = 0;

    while let Some(_) = bc.instructions.get(index) {
        index = asm.emit_function(index);
    }

    // asm.emit_function(0);
    let asm_as_str = asm.to_string();
    return Ok(asm_as_str);
}
