mod ast;
mod binexpr;
mod bytecode;
mod backend_tests;
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
  -o <name>     Set output binary name
  -b            Print bytecode
  -s            Print x64 assembly
  -a            Print the AST
  -no-std       Do not bundle stdlib functions with compiled code
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

    let with_std = !args.contains(&"-no-std".to_string());
    let filename = &args[1];
    let code = std::fs::read_to_string(filename).unwrap();
    let code = if with_std { with_stdlib(&code) } else { code };

    match typer::Typer::from_code(&code).and_then(|t| t.check()) {
        Ok(t) => t,
        Err(e) => {
            return;
        }
    };

    if args.contains(&"-a".to_string()) {
        let ast = ast::AST::from_code(&code).unwrap();
        println!("{ast}");
        return;
    }

    if args.contains(&"-b".to_string()) {
        let bc = bytecode::Bytecode::from_code(&code).unwrap();
        println!("{bc}");
        return;
    }

    let env = util::Env::current();

    if args.contains(&"-s".to_string()) {
        let asm = x64::emit_assembly(&code, env).unwrap();
        println!("{asm}");
        return;
    }

    let mut out_name = "a.out";

    if let Some((k, _)) = args.iter().enumerate().find(|&(_, arg)| arg == "-o") {
        out_name = &args[k + 1];
    }

    x64::emit_binary(&code, out_name, env).unwrap();

    std::process::Command::new(format!("./{out_name}"))
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
}
