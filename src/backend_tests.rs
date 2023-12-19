#[cfg(test)]
mod tests {
    use crate::util::LINUX_QEMU_ARM64;
    use crate::util::random_str;
    use crate::util::Env;

    fn run_test(env: &'static Env, program: &str) -> (i32, String) {
        let program = crate::util::with_stdlib(&program);
        let bin_name = format!("test_{}", random_str(8));

        env.emit_binary(&bin_name, &program)
            .unwrap();

        let out = env.run_binary(&bin_name);

        std::process::Command::new("rm")
            .args(["-rf", &bin_name])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        let code = out.status.code().unwrap();
        let stdout = core::str::from_utf8(&out.stdout).unwrap().to_string();

        return (code, stdout);
    }

    fn get_supported_envs() -> Vec<&'static Env> {
        let mut envs: Vec<&'static Env> = Vec::new();
        envs.push(Env::current());

        let has_qemu = std::process::Command::new("qemu-aarch64-static")
            .arg("--version")
            .stdout(std::process::Stdio::piped())
            .spawn()
            .is_ok();

        if has_qemu {
            envs.push(&LINUX_QEMU_ARM64);
        }

        return envs;
    }

    fn expect_code(expect_code: i32, program: &str) {
        let mut is_all_ok = true;

        for env in get_supported_envs() {
            let (code, _) = run_test(env, program);
            let ok = expect_code == code;
            let msg = if ok { "ok" } else { "\x1B[31mERROR\x1B[0m" };
            println!("  {} ... {}", env.name, msg);
            is_all_ok = is_all_ok && ok;
        }

        assert!(is_all_ok);
    }

    fn expect_stdout(expect_stdout: &str, program: &str) {
        let mut is_all_ok = true;

        for env in get_supported_envs() {
            let (code, stdout) = run_test(env, program);
            let ok = code == 0 && expect_stdout == stdout;
            let msg = if ok { "ok" } else { "\x1B[31mERROR\x1B[0m" };
            println!("  {} ... {}", env.name, msg);
            is_all_ok = is_all_ok && ok;
        }

        assert!(is_all_ok);
    }

    #[test]
    fn assert_can_fail() {
        let program = r###"
        assert(0 == 1);
        "###;

        expect_code(1, program);
    }

    #[test]
    fn ensure_can_succeed() {
        let code = r###"
        assert(420 == 420);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn ensure_can_print() {
        let program = r###"
        print(&"cowabunga!\n");
        "###;
        expect_stdout("cowabunga!\n", program);
    }

    #[test]
    fn should_call_identity_fn() {
        let program = r###"
        fun identity(x: int): int {
            return x;
          }
          var x = identity(420);
          assert(x == 420);          
        "###;
        expect_code(0, program);
    }

    #[test]
    fn should_call_factorial_fn() {
        let program = r###"
        fun factorial(n: int): int {
            if n == 1 {
              return 1;
            }
            return n * factorial(n - 1);
          }
          var x = factorial(5);
          assert(x == 120);          
        "###;
        expect_code(0, program);
    }

    #[test]
    fn should_return_code() {
        let s = r###"
            exit(5);
        "###;
        expect_code(5, s);
    }

    #[test]
    fn should_call_print_with_variable_arg() {
        let code = r###"
        var x = "hello!";
        print(&x);
        "###;
        expect_stdout("hello!", code);
    }

    #[test]
    fn should_call_print_with_literal_arg() {
        let code = r###"
        print(&"hello!");
        "###;
        expect_stdout("hello!", code);
    }

    #[test]
    fn should_call_print_in_sub_procedure_with_string_passed_by_reference() {
        let code = r###"
        fun thing(x: &string): void {
          print(x);
        }
        thing(&"hello!");
        "###;
        expect_stdout("hello!", code);
    }

    #[test]
    fn should_call_print_with_inline_member_access() {
        let code = r###"
        type person = {
            name: string,
        };

        var x = person { name: "helmut" };
        print(&x.name);
        "###;
        expect_stdout("helmut", code);
    }

    #[test]
    fn should_call_print_with_member_access_in_variable_on_stack() {
        let code = r###"
        type person = {
            name: string,
        };

        var x = person { name: "helmut" };
        var y = x.name;
        print(&y);
        "###;
        expect_stdout("helmut", code);
    }

    #[test]
    fn should_call_print_with_deep_member_access() {
        let code = r###"
        type C = {
            yee: string,
        };
        type B = {
            c: C,  
        };
        type A = {
            b: B,
        };

        var x = A { b: B { c: C { yee: "cowabunga!" } } };
        print(&x.b.c.yee);
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_call_print_with_derefefenced_variable() {
        let code = r###"
        type B = { value: string };
        type A = { b: B };

        fun takes(a: &A): void {
            print(a.b.value);
        }
        var x = A { b: B { value: "cowabunga!" } };
        takes(&x);
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_call_print_with_derefefenced_variable_with_offset() {
        let code = r###"
        type B = { yee: int, boi: int, value: string };
        type A = { b: B };

        fun takes(a: &A): void {
            print(a.b.value);
        }
        var x = A { b: B { yee: 420, boi: 69, value: "cowabunga!" } };
        takes(&x);
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_derefence_member_scalar_and_add() {
        let code = r###"
        type A = { x: int };

        fun takes(a: &A): int {
            return *a.x + 1;
        }
        var x = A { x: 69 };
        var y = takes(&x);
        exit(y);
        "###;
        expect_code(70, code);
    }

    #[test]
    fn should_derefence_member_scalar_into_local_and_add() {
        let code = r###"
        type A = { x: int };

        fun takes(a: &A): int {
            var y: int = *a.x;
            return y + 1;
        }
        var x = A { x: 69 };
        var y = takes(&x);
        exit(y);
        "###;
        expect_code(70, code);
    }

    #[test]
    fn should_jump_with_else_if() {
        let code = r###"
        if 1 == 2 {
            print(&"bad!");
        } else if 5 == 5 {
            print(&"cowabunga!");
        }
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_jump_with_else() {
        let code = r###"
        if 1 == 2 {
            exit(42);
        } else {
            exit(69);
        }
        "###;
        expect_code(69, code);
    }

    #[test]
    fn should_jump_with_boolean_literal() {
        let code = r###"
        if false {
            exit(42);
        } else if true {
            exit(69);
        }
        "###;
        expect_code(69, code);
    }

    #[test]
    fn should_multiply() {
        let code = r###"
        var x = 3 * 3;
        exit(x);
        "###;
        expect_code(9, code);
    }

    #[test]
    fn should_multiply_negative_number() {
        let code = r###"
        var x = -4 * -4;
        exit(x);
        "###;
        expect_code(16, code);
    }

    #[test]
    fn should_divide() {
        let code = r###"
        var x = 6 / 2;
        exit(x);
        "###;
        expect_code(3, code);
    }

    #[test]
    fn should_divide_negative_number() {
        let code = r###"
        var x = -8 / -2;
        exit(x);
        "###;
        expect_code(4, code);
    }

    #[test]
    fn should_divide_with_remainder() {
        let code = r###"
        var x = 9 / 2;
        exit(x);
        "###;
        expect_code(4, code);
    }

    #[test]
    fn should_respect_operator_precedence() {
        let code = r###"
        var x = (1 + 2) * 3;
        exit(x);
        "###;
        expect_code(9, code);
    }

    #[test]
    fn should_do_math() {
        let code = r###"
        assert(5 == 5);
        assert(5 * 5 == 25);
        assert(-5 * -5 == 25);
        assert(5 + 3 * 5 == 20);
        assert(5 * -1 == -5);
        assert(5 / -1 == -5);
        assert((5 + 3) * 2 == 16);

        print(&"cowabunga!");
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_enter_falsy_while_condition() {
        let code = r###"
        while 1 == 1 {
            exit(42);
        }
        exit(3);
        "###;
        expect_code(42, code);
    }

    #[test]
    fn should_not_enter_falsy_while_condition() {
        let code = r###"
        while 1 == 2 {
            exit(42);
        }
        exit(3);
        "###;
        expect_code(3, code);
    }

    #[test]
    fn should_compile_not_equals() {
        let code = r###"
        assert(1 != 2);
        assert(1 == 1);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_less_than() {
        let code = r###"
        assert(1 < 2);
        assert(1 < 1 == false);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_less_than_or_equals() {
        let code = r###"
        assert(1 <= 2);
        assert(2 <= 2);
        assert(3 <= 2 == false);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_greater_than() {
        let code = r###"
        assert(2 > 1);
        assert(2 > 2 == false);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_greater_than_or_equals() {
        let code = r###"
        assert(2 >= 1);
        assert(2 >= 2);
        assert(2 >= 3 == false);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_not() {
        let code = r###"
        assert(!false);
        assert(!(1 > 2));
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_not_to_the_first_bit_only() {
        let code = r###"
        assert(!false == true);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_reassignment_to_local() {
        let code = r###"
        var x = 0;
        x = 5;
        exit(x);
        "###;
        expect_code(5, code);
    }

    #[test]
    fn should_compile_reassignment_to_member() {
        let code = r###"
        type person = { age: int };
        var x = person { age: 3 };
        x.age = 7;
        exit(x.age);
        "###;
        expect_code(7, code);
    }

    #[test]
    fn should_compile_deref_from_local() {
        let code = r###"
        var x = 420;
        var y = &x;
        var z = *y;
        assert(z == 420);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_deref_from_pointer_in_argument() {
        let code = r###"
        fun takes(a: &A): int {
            return *a.x + *a.y + 1;
        }
        type A = { x: int, y: int };
        var a = A { x: 420, y: 69 };
        var b = takes(&a);
        assert(b == 490);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_copy_entire_struct_when_dereferencing_pointer() {
        let code = r###"
        fun do_print(a: &string): void {
            var b = *a;
            print(&b);
        }
        do_print(&"bunga!");
        "###;
        expect_stdout("bunga!", code);
    }

    #[test]
    fn should_copy_entire_struct_when_dereferencing_copied_pointer() {
        let code = r###"
        fun do_print(a: &string): void {
            var b = a;
            var c = *b;
            print(&c);
        }
        do_print(&"bunga!");
        "###;
        expect_stdout("bunga!", code);
    }

    #[test]
    fn should_compile_indirect_store_to_local() {
        let code = r###"
        var x = 420;
        var y = &x;
        *y = 3;
        assert(*y == 3);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_indirect_store_of_scalar_to_member() {
        let code = r###"
        type X = { a: int, b: int };
        var x = X { a: 420, b: 7 };
        var y = &x;
        *y.b = 5;
        assert(*y.a == 420);
        assert(*y.b == 5);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_indirect_store_of_struct_to_member() {
        let code = r###"
        type X = { a: int, b: string };
        var x = X { a: 420, b: "cowabunga!" };
        var y = &x;
        *y.b = "yee!";
        print(y.b);
        assert(*y.a == 420);
        "###;
        expect_stdout("yee!", code);
    }

    #[test]
    fn should_compile_indirect_store_with_nested_struct() {
        let code = r###"
        type B = { z: int, a: string };
        type A = { x: int, y: B };
        var a = A { x: 420, y: B { z: 69, a: "cowabunga!" } };
        var b = B { z: 3, a: "yee!" };
        var z = &a;
        *z.y = b;
        print(&a.y.a);
        "###;
        expect_stdout("yee!", code);
    }

    #[test]
    fn should_compile_copy_of_local_struct() {
        let code = r###"
        type A = { x: int, y: int };
        var a = A { x: 1, y: 2 };
        var b = a;
        assert(b.x == 1);
        assert(b.y == 2);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_copy_of_member_struct() {
        let code = r###"
type B = { x: int, y: int };
type A = { a: int, b: B };

var t1 = A { a: 420, b: B { x: 3, y: 5 } };
var t2 = B { x: 72, y: 69 };
t1.b = t2;
assert(t1.b.x == 72);
assert(t1.b.y == 69);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_element_access_to_int() {
        let code = r###"
        var x = [420, 69];
        var y = x[1];
        assert(y == 69);
        "###;
        expect_code(0, code);
    }

    #[test]
    fn should_compile_element_access_to_string() {
        let code = r###"
        var x = ["yee", "boi"];
        var y = x[1];
        print(&y);
        "###;
        expect_stdout("boi", code);
    }

    #[test]
    fn should_compile_element_access_to_deep_string() {
        let code = r###"
        var x = [[["yee", "cowabunga!"]], [["boi", "dude"]]];
        var y = x[1][0][1];
        print(&y);
        "###;
        expect_stdout("dude", code);
    }

    #[test]
    fn should_compile_element_access_with_expression() {
        let code = r###"
        var x = ["yee", "boi"];
        var k = 0;
        var y = x[k + 1];
        print(&y);
        "###;
        expect_stdout("boi", code);
    }

    #[test]
    fn should_compile_element_access_in_condition() {
        let code = r###"
        var x = [1, 2];
        if x[0] == 1 {
            print(&"boi");
        }
        "###;
        expect_stdout("boi", code);
    }

    #[test]
    fn should_compile_element_access_in_falsy_condition() {
        let code = r###"
        var x = [1, 2];
        if x[0] == 2 {
            print(&"boi");
        } else {
            print(&"cowabunga!");
        }
        "###;
        expect_stdout("cowabunga!", code);
    }

    #[test]
    fn should_compile_if_statement_with_boolean() {
        let code = r###"
        var x = true;
        if x {
            print(&"boi");
        } else {
            print(&"cowabunga!");
        }
        "###;
        expect_stdout("boi", code);
    }
}
