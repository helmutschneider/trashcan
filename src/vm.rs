use crate::bytecode::{self, Instruction};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Frame {
    pub program_counter: usize,
    pub memory: HashMap<String, i64>,
}

#[derive(Debug)]
pub struct VM {
    pub is_running: bool,
    pub state: Vec<Frame>,
}

fn resolve_argument_value(frame: &Frame, value: &bytecode::Argument) -> i64 {
    let actual_value: i64 = match value {
        bytecode::Argument::Integer(x) => *x,
        bytecode::Argument::Variable(v) => frame.memory[&v.0],
    };
    return actual_value;
}

fn find_function_index(bc: &bytecode::Bytecode, name: &str) -> usize {
    for i in 0..bc.instructions.len() {
        let instr = &bc.instructions[i];

        if let bytecode::Instruction::Function(fx_name, _) = instr {
            if fx_name == name {
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
            bytecode::Instruction::Copy(dest, source) => {
                let frame = self.get_frame_mut();
                let name = dest.0.clone();
                let value = resolve_argument_value(frame, source);
                frame.memory.insert(name, value);
                frame.program_counter += 1;
            }
            bytecode::Instruction::Call(ret_temp, call_name, call_args) => {
                let frame = self.get_frame();
                let fn_index = find_function_index(bc, &call_name);
                let mut call_frame = Frame {
                    program_counter: fn_index,
                    memory: HashMap::new(),
                };

                let fx_args = match &bc.instructions[fn_index] {
                    bytecode::Instruction::Function(_, fx_args) => fx_args,
                    _ => panic!(),
                };

                for k in 0..fx_args.len() {
                    let fx_arg = &fx_args[k];
                    let call_arg = &call_args[k];
                    let value = resolve_argument_value(frame, call_arg);
                    call_frame.memory.insert(fx_arg.0.clone(), value);
                }

                self.state.push(call_frame);
            }
            bytecode::Instruction::Add(temp, x, y) => {
                let frame = self.get_frame_mut();
                let x_val = resolve_argument_value(&frame, x);
                let y_val = resolve_argument_value(&frame, y);
                frame.memory.insert(temp.to_string(), x_val + y_val);
                frame.program_counter += 1;
            }
            bytecode::Instruction::Return(ret_reg) => {
                let frame = self.get_frame();
                let ret_val = resolve_argument_value(frame, ret_reg);
                let is_return_of_main: bool = {
                    let mut res = false;

                    for k in 0..frame.program_counter {
                        let instr_idx = frame.program_counter - k - 1;

                        if let Instruction::Function(fn_name, _) = &bc.instructions[instr_idx] {
                            if fn_name == "main" {
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

                    if let Instruction::Call(target_temp, _, _) = call_instr {
                        parent_frame.memory.insert(target_temp.to_string(), ret_val);
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

        let mut vm = VM::new();
        vm.execute(&bc);

        assert_eq!(13, vm.state[0].memory["y"]);
        assert_eq!(420 + 69 + 13, vm.state[0].memory["x"]);
    }
}
