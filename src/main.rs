mod ast;
mod bytecode;
mod tokenizer;
mod vm;
mod x64;

fn main() {
    let code = r###"
    fun main(): void {
        var x: int = 420 + 69;
    }
"###;
    let bc = bytecode::from_code(code);
    let mut vm = vm::VM::new();
    vm.execute(&bc);
}
