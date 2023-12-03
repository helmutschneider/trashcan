mod ast;
mod bytecode;
mod op;
mod tokenizer;
mod typer;
mod util;
mod x64;

use util::with_stdlib;

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
    let code = with_stdlib(&code);

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

    let os = util::OperatingSystem::current();

    if args.contains(&"-s".to_string()) {
        let asm = x64::emit_assembly(&code, os).unwrap();
        println!("{asm}");
        return;
    }

    let out_name = "a.out";
    x64::emit_binary(&code, out_name, os).unwrap();

    std::process::Command::new(format!("./{out_name}"))
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
}
