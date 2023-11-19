mod ast;
mod bytecode;
mod tokenizer;
mod typer;
mod util;
mod x64;

use crate::util::Error;

const USAGE_STR: &'static str = r###"
Usage:
  cargo run [filename.tc]

Options:
  -o <name>     Output binary name
  -b            Print bytecode
  -s            Print x64 assembly
"###;

fn usage() {
    eprintln!("{}", USAGE_STR.trim());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
        return;
    }

    let filename = &args[1];
    let code = std::fs::read_to_string(filename).unwrap();
    
    match typer::Typer::from_code(&code).and_then(|t| t.check()) {
        Ok(t) => t,
        Err(e) => {
            return;
        }
    };

    if args.contains(&"-b".to_string()) {
        let bc = bytecode::Bytecode::from_code(&code).unwrap();
        println!("{bc}");
        return;
    }

    let asm = match x64::emit_assembly(&code) {
        Ok(s) => s,
        Err(_) => {
            return;
        }
    };

    if args.contains(&"-s".to_string()) {
        println!("{asm}");
        return;
    }

    x64::emit_binary(&asm, "a.out");
}
