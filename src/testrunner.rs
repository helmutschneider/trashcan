#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::util::Env;
    use crate::util::random_str;
    use crate::x64::emit_binary;

    fn run_one_test(code: &str, env: &'static Env) -> (i32, String) {
        let body_with_stdlib = crate::util::with_stdlib(&code);
        let bin_name = format!("test_{}", random_str(8));

        emit_binary(&body_with_stdlib, &bin_name, env)
            .unwrap();

        let stdout = std::process::Stdio::piped();
        let out = std::process::Command::new(format!("./{}", bin_name))
            .stdout(stdout)
            .spawn()
            .unwrap()
            .wait_with_output()
            .unwrap();

        std::process::Command::new("rm")
            .args(["-rf", &bin_name])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        let code = out.status.code().unwrap();
        let stdout = core::str::from_utf8(&out.stdout)
            .unwrap()
            .to_string();
        
        return (code, stdout);
    }

    fn run_all_tests(env: &'static Env) -> bool {
        let mut rdr = std::fs::read_dir("./tests")
            .unwrap();

        let mut is_all_ok = true;
        let mut buf = String::with_capacity(8192);

        while let Some(Ok(x)) = rdr.next() {
            let name = x.file_name();
            let name = name.to_str()
                .unwrap();
            let (name, _) = name.split_once(".")
                .unwrap();

            buf.clear();
            std::fs::File::open(x.path())
                .unwrap()
                .read_to_string(&mut buf)
                .unwrap();

            let (res, _) = run_one_test(&buf, env);
            let msg = if res == 0 { "ok" } else { "\x1B[31mFAILED\x1B[0m" };

            println!("test {} ({}, {}) ... {}", name, env.os, env.arch, msg);

            is_all_ok = is_all_ok && res == 0;
        }

        return is_all_ok;
    }

    #[test]
    fn assert_can_fail() {
        let env = Env::current();
        let code = r###"
        assert(0 == 1);
        "###;
        let (res, _) = run_one_test(code, env);
        assert_eq!(1, res);
    }

    #[test]
    fn assert_can_succeed() {
        let env = Env::current();
        let code = r###"
        assert(420 == 420);
        "###;
        let (res, _) = run_one_test(code, env);
        assert_eq!(0, res);
    }

    #[test]
    fn can_print() {
        let env = Env::current();
        let code = r###"
        print(&"cowabunga!\n");
        "###;
        let (res, out) = run_one_test(code, env);

        assert_eq!(0, res);
        assert_eq!("cowabunga!\n", out);
    }

    #[test]
    fn run_tests() {
        let env = Env::current();
        let ok = run_all_tests(env);

        assert!(ok);
    }
}
