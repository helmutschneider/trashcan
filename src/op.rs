use std::collections::HashMap;

use crate::tokenizer::Token;
use crate::tokenizer::TokenKind;
use crate::tokenizer::tokenize;

#[derive(Debug, Clone, Copy, PartialEq)]
enum OpKind {
    None,
    Binary,
    UnaryPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Op(&'static str, OpKind, i64);

impl Op {
    // https://en.wikipedia.org/wiki/Order_of_operations#Programming_languages
    const MEMBER_ACCESS: Op = Op(".", OpKind::Binary,15);
    const UNARY_PLUS: Op = Op("+", OpKind::UnaryPrefix, 14);
    const UNARY_MINUS: Op = Op("-", OpKind::UnaryPrefix, 14);
    const MULTIPLY: Op = Op("*", OpKind::Binary, 13);
    const DIVIDE: Op = Op("/", OpKind::Binary, 13);
    const BINARY_PLUS: Op = Op("+", OpKind::Binary, 12);
    const BINARY_MINUS: Op = Op("-", OpKind::Binary, 12);

    const OPAREN: Op = Op("(", OpKind::None, 0);
    const CPAREN: Op = Op(")", OpKind::None, 0);

    const OPERATORS: &'static [Op] = &[
        Op::MEMBER_ACCESS,
        Op::UNARY_PLUS,
        Op::UNARY_MINUS,
        Op::MULTIPLY,
        Op::DIVIDE,
        Op::BINARY_PLUS,
        Op::BINARY_MINUS,
    ];
}

#[derive(Debug, Clone)]
enum Node {
    Operand(Token),
    Expr(Op, Vec<Node>),
    Fn(Box<Node>, Vec<Node>),
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Operand(s) => s.value.clone(),
            Self::Expr(op, operands) => {
                let operand_s = operands.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(" ");
                let s = format!("({} {})", op.0, operand_s);
                s
            }
            Self::Fn(fn_node, fn_args) => {
                let mut fn_args_s = fn_args.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                
                if !fn_args_s.is_empty() {
                    fn_args_s = format!(" {}", fn_args_s);
                }

                let s = format!("(call {}{})", fn_node, fn_args_s);
                s
            }
        };
        return f.write_str(&s);
    }
}

fn find_operator(token: &Token, kind: OpKind) -> Op {
    return Op::OPERATORS.iter()
        .find(|op| op.0 == token.value && op.1 == kind)
        .map(|op| *op)
        .unwrap();
}

fn token_is_operator(token: &Token) -> bool {
    return Op::OPERATORS.iter().any(|op| op.0 == token.value);
}

fn token_is_operand(token: &Token) -> bool {
    return Op::OPERATORS.iter().all(|op| op.0 != token.value);
}

fn last_operator_is(op: Op, operators: &[Op]) -> bool {
    return operators.last().map(|other_op| *other_op == op).unwrap_or(false);
}

fn has_operator_on_stack_with_higher_precedence(my_op: Op, operators: &[Op]) -> bool {
    if let Some(other_node) = operators.last() {
        if other_node.1 == OpKind::None {
            return false;
        }
        if my_op.1 == OpKind::UnaryPrefix {
            // unary operators can only pop other unary operators.
            return other_node.2 > my_op.2
                && other_node.1 == OpKind::UnaryPrefix;
        }
        return other_node.2 > my_op.2
            || (other_node.2 == my_op.2 && my_op.1 == OpKind::Binary);
    }
    return false;
}

fn pop_expr(operators: &mut Vec<Op>, operands: &mut Vec<Node>) {
    let op = operators.pop().unwrap();

    match op.1 {
        OpKind::Binary => {
            let rhs = operands.pop().unwrap();
            let lhs = operands.pop().unwrap();
            let node = Node::Expr(op, vec![lhs, rhs]);
            operands.push(node);
        }
        OpKind::UnaryPrefix => {
            let lhs = operands.pop().unwrap();
            let node = Node::Expr(op, vec![lhs]);
            operands.push(node);
        }
        _ => {}
    };
}

fn do_shunting_yard(value: &str) -> Node {
    let mut operators: Vec<Op> = Vec::new();
    let mut operands: Vec<Node> = Vec::new();
    let tokens = tokenize(value).unwrap();

    // a shunting-yard implementation with support for function calls.
    //   https://www.reedbeta.com/blog/the-shunting-yard-algorithm/#advanced-usage
    for k in 0..tokens.len() {
        let tok = &tokens[k];
        let look_for_binary_or_postfix = k > 0
            && tokens.get(k - 1).map(token_is_operand).unwrap()
            && tokens.get(k - 1).map(|t| t.value != "(").unwrap();
        let look_for_prefix = k == 0
            || tokens.get(k - 1).map(token_is_operator).unwrap();

        if tok.value == "(" {
            // if we encounter a open parenthesis in the postfix
            // position it's a function invokation. pop any operators
            // that have higher precedence than the function call node,
            // for example member access.
            if look_for_binary_or_postfix {
                while has_operator_on_stack_with_higher_precedence(Op::OPAREN, &operators) {
                    pop_expr(&mut operators, &mut operands);
                }

                // initialize a new operand and put it on the stack.
                // its arguments will be empty for now, but we will
                // put stuff in there when we encounter commas or the
                // closing parenthesis.
                let callee = operands.pop().unwrap();
                let fn_node = Node::Fn(Box::new(callee), Vec::new());
                operands.push(fn_node);
            }

            operators.push(Op::OPAREN);
        } else if tok.value == ")" {
            while !last_operator_is(Op::OPAREN, &operators) {
                pop_expr(&mut operators, &mut operands);
            }

            // the closing parethesis in the postfix position is the end
            // of a function call. if the function call had only one argument
            // we will never hit the ',' case, which means that we possibly
            // need to pop one argument off the stack.
            if look_for_binary_or_postfix {
                let operands_len = operands.len();

                if operands_len > 1 && matches!(operands.get(operands_len - 2), Some(Node::Fn(_, _))) {
                    let arg = operands.pop().unwrap();
                    if let Node::Fn(_, args) = operands.last_mut().unwrap() {
                        args.push(arg);
                    } else {
                        panic!("expected function node");
                    }
                }   
            }

            // pop the '('
            operators.pop().unwrap();
        } else if tok.value == "," {
            // push function arguments onto the last function node.
            while !last_operator_is(Op::OPAREN, &operators) {
                pop_expr(&mut operators, &mut operands);
            }
            let arg = operands.pop().unwrap();
            let fn_node = operands.last_mut().unwrap();

            if let Node::Fn(_, args) = fn_node {
                args.push(arg);
            } else {
                panic!("expected function node");
            }
        } else if token_is_operator(tok) {
            let op: Op;
            if look_for_prefix {
                op = find_operator(tok, OpKind::UnaryPrefix);
            } else if look_for_binary_or_postfix {
                op = find_operator(tok, OpKind::Binary);
            } else {
                panic!();
            }
            while has_operator_on_stack_with_higher_precedence(op, &operators) {
                pop_expr(&mut operators, &mut operands);
            }
            operators.push(op);
        } else {
            let n = Node::Operand(tok.clone());
            operands.push(n);
        }
    }

    while !operators.is_empty() {
        pop_expr(&mut operators, &mut operands);
    }

    println!("{:?}", operands);
    assert_eq!(1, operands.len());

    return operands[0].clone();
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::tokenize;
    use crate::op::do_shunting_yard;

    #[test]
    fn shunt_parse_1() {
        do_test("(+ 1 2)", "1 + 2");
    }

    #[test]
    fn shunt_parse_2() {
        do_test("(+ 1 2)", "(1 + 2)");
    }

    #[test]
    fn shunt_parse_3() {
        do_test("(+ 1 (* 2 3))", "1 + 2 * 3");
    }

    #[test]
    fn shunt_parse_4() {
        do_test("(- (- 5) (* (+ 3) 7))", "-5 - +3 * 7");
    }

    #[test]
    fn shunt_parse_5() {
        do_test("(- 5 (* 3 7))", "5 - 3 * 7");
    }

    #[test]
    fn shunt_parse_6() {
        do_test("(+ (. x y) 420)", "x.y + 420");
    }

    #[test]
    fn shunt_parse_7() {
        do_test("(call a)", "a()")
    }

    #[test]
    fn shunt_parse_8() {
        do_test("(call (. x name))", "x.name()");
    }
    
    #[test]
    fn shunt_parse_9() {
        do_test("(call a (+ 1 2))", "a(1 + 2)");
    }

    #[test]
    fn shunt_parse_10() {
        do_test("(call a 5 9)", "a(5, 9)");
    }

    #[test]
    fn shunt_parse_11() {
        do_test("(call a 5 (call b 42))", "a(5, b(42))");
    }

    #[test]
    fn shunt_parse_12() {
        do_test("(* (call a 5) 3)", "a(5) * 3");
    }

    #[test]
    fn shunt_parse_13() {
        do_test("(call a (* 5 3))", "a((5) * 3)");
    }

    fn do_test(expected: &str, to_parse: &str) {
        let n = do_shunting_yard(to_parse);
        assert_eq!(expected, n.to_string());
    }
}