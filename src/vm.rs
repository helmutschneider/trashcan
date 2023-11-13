struct Frame {
    registers: Vec<i64>,
}

struct VM {
    pc: usize,
    state: Vec<Frame>,
}

impl VM {
    fn new() -> Self {
        return Self {
            pc: 0,
            state: Vec::new(),
        };
    }

    fn execute(&mut self, bc: &crate::bytecode::Bytecode) {
        // let main_fn = bc.main;
    }
}
