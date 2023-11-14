mod tokenizer;
mod ast;
mod bytecode;
mod vm;

fn main() {
    let code = r###"
    var x: int = 6;
    var y: int = x;
"###;
    let bc = bytecode::from_code(code);

    println!("{}", bc);

    let mut vm = vm::VM::new();
    vm.execute(&bc);

    println!("{:?}", vm);
}
