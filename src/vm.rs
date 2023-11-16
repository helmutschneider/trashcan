use crate::bytecode::{self, Instruction};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Frame {
    pub program_counter: usize,
    pub memory: HashMap<String, i64>,
    pub registers: HashMap<usize, i64>,
}

#[derive(Debug)]
pub struct VM {
    pub is_running: bool,
    pub state: Vec<Frame>,
}

fn resolve_value_to_load(frame: &Frame, value: &bytecode::LoadArgument) -> i64 {
    let actual_value: i64 = match value {
        bytecode::LoadArgument::Integer(x) => *x,
        bytecode::LoadArgument::Register(r) => frame.registers[&r.0],
        bytecode::LoadArgument::Symbol(sym) => frame.memory[&sym.name],
    };
    return actual_value;
}

fn find_function_index(bc: &bytecode::Bytecode, name: &str) -> usize {
    for i in 0..bc.instructions.len() {
        let instr = &bc.instructions[i];

        if let bytecode::Instruction::Function(fn_sym, _) = instr {
            if fn_sym.name == name {
                return i;
            }
        }
    }

    panic!("Could not find function '{name}'");
}

impl VM {
    pub fn new() -> Self {
        return Self {
            is_running: false,
            state: Vec::new(),
        };
    }

    fn get_frame(&mut self) -> &Frame {
        let last_index = self.state.len() - 1;
        return &self.state[last_index];
    }

    fn get_frame_mut(&mut self) -> &mut Frame {
        let last_index = self.state.len() - 1;
        return &mut self.state[last_index];
    }

    fn execute_one(&mut self, bc: &bytecode::Bytecode) {
        let instr = {
            let frame = self.get_frame();
            &bc.instructions[frame.program_counter]
        };

        match instr {
            bytecode::Instruction::Function(sym, args) => {
                let frame = self.get_frame_mut();
                frame.program_counter += 1;
            }
            bytecode::Instruction::Store(sym, reg) => {
                let frame = self.get_frame_mut();
                let actual_value = frame.registers[&reg.0];
                frame.memory.insert(sym.name.clone(), actual_value);
                frame.program_counter += 1;
            }
            bytecode::Instruction::Load(reg, arg) => {
                let frame = self.get_frame_mut();
                let value = resolve_value_to_load(&frame, arg);
                frame.registers.insert(reg.0, value);
                frame.program_counter += 1;
            }
            bytecode::Instruction::FunctionCall(ret_reg, sym, arg_registers) => {
                let frame = self.get_frame();
                let fn_index = find_function_index(bc, &sym.name);
                let mut call_frame = Frame {
                    program_counter: fn_index,
                    memory: HashMap::new(),
                    registers: HashMap::new(),
                };

                let fx_args = match &bc.instructions[fn_index] {
                    bytecode::Instruction::Function(_, fx_args) => fx_args,
                    _ => panic!(),
                };

                for k in 0..fx_args.len() {
                    let arg_reg = &arg_registers[k];
                    let arg_value = frame.registers[&arg_reg.0];
                    call_frame.registers.insert(k, arg_value);
                }

                self.state.push(call_frame);
            }
            bytecode::Instruction::Add(reg, x, y) => {
                let frame = self.get_frame_mut();
                let x_val = frame.registers[&x.0];
                let y_val = frame.registers[&y.0];
                frame.registers.insert(reg.0, x_val + y_val);
                frame.program_counter += 1;
            }
            bytecode::Instruction::Return(ret_reg) => {
                let frame = self.get_frame();
                let ret_val = frame.registers[&ret_reg.0];
                let is_return_of_main: bool = {
                    let mut res = false;

                    for k in 0..frame.program_counter {
                        let instr_idx = frame.program_counter - k - 1;

                        if let Instruction::Function(fn_sym, _) = &bc.instructions[instr_idx] {
                            if fn_sym.name == "main" {
                                res = true;
                                break;
                            }
                        }
                    }
                    res
                };

                if !is_return_of_main {
                    self.state.pop();

                    let parent_frame = self.get_frame_mut();
                    let call_instr = &bc.instructions[parent_frame.program_counter];

                    if let Instruction::FunctionCall(target_reg, _, _) = call_instr {
                        parent_frame.registers.insert(target_reg.0, ret_val);
                    }
                    parent_frame.program_counter += 1;
                } else {
                    self.is_running = false;
                }

                // let parent_frame = &mut self.state[self.state.len() - 1];
            }
            _ => panic!("Unknown instruction: '{:?}'", instr),
        };
    }

    pub fn execute(&mut self, bc: &bytecode::Bytecode) {
        let main_index = find_function_index(bc, "main");
        let frame = Frame {
            program_counter: main_index,
            memory: HashMap::new(),
            registers: HashMap::new(),
        };
        self.state.push(frame);
        self.is_running = true;

        while self.is_running {
            self.execute_one(bc);
        }

        // self.execute_instructions();

        // let main_fn = bc.main;
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode;
    use crate::vm::*;

    #[test]
    fn should_execute_add() {
        let code = r###"
            fun main(): void {
                var x: int = 420 + 69;
            }
        "###;
        let bc = bytecode::from_code(code);
        let mut vm = VM::new();
        vm.execute(&bc);

        assert_eq!(420 + 69, vm.state[0].memory["x"]);
    }

    #[test]
    fn should_call_fn() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }

            fun main(): void {
                var x: int = add(420, 69);
            }
        "###;
        let bc = bytecode::from_code(code);
        println!("{bc}");
        let mut vm = VM::new();
        vm.execute(&bc);

        assert_eq!(420 + 69, vm.state[0].memory["x"]);
    }

    #[test]
    fn should_resolve_local_with_naming_collision() {
        let code = r###"
            fun add(x: int, y: int): int {
                return x + y;
            }

            fun main(): void {
                var y: int = 13;
                var x: int = add(420, 69) + y;
            }
        "###;
        let bc = bytecode::from_code(code);

        println!("{bc}");

        let mut vm = VM::new();
        vm.execute(&bc);

        assert_eq!(13, vm.state[0].memory["y"]);
        assert_eq!(420 + 69 + 13, vm.state[0].memory["x"]);
    }
}
