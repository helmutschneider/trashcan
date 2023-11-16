mod ast;
mod bytecode;
mod tokenizer;
mod vm;
mod x64;

fn main() {
    let s = r###"
    fun main(): void {
        var x: int = 420 + 69;
    }
"###;
    let asm = x64::emit_assembly(s);
}
