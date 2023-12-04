use std::collections::HashMap;

use crate::tokenizer::Token;
use crate::tokenizer::TokenKind;
use crate::tokenizer::tokenize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpKind {
    None,
    Binary,
    UnaryPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpAssoc {
    None,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Op(&'static str, OpKind, OpAssoc, i64);

impl Op {
    // https://en.wikipedia.org/wiki/Order_of_operations#Programming_languages
    pub const MEMBER_ACCESS: Op = Op(".", OpKind::Binary, OpAssoc::Left, 15);
    pub const UNARY_PLUS: Op = Op("+", OpKind::UnaryPrefix, OpAssoc::Right, 14);
    pub const UNARY_MINUS: Op = Op("-", OpKind::UnaryPrefix, OpAssoc::Right, 14);
    pub const MULTIPLY: Op = Op("*", OpKind::Binary, OpAssoc::Left, 13);
    pub const DIVIDE: Op = Op("/", OpKind::Binary, OpAssoc::Left, 13);
    pub const BINARY_PLUS: Op = Op("+", OpKind::Binary, OpAssoc::Left, 12);
    pub const BINARY_MINUS: Op = Op("-", OpKind::Binary, OpAssoc::Left, 12);

    pub const OPAREN: Op = Op("(", OpKind::None, OpAssoc::None, 0);
    pub const CPAREN: Op = Op(")", OpKind::None, OpAssoc::None, 0);

    pub const OPERATORS: &'static [Op] = &[
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
pub enum Node {
    Operand(Token),
    Expr(OpNode, Vec<Node>),
    Fn(Box<Node>, Vec<Node>),
}

#[derive(Debug, Clone)]
pub struct OpNode {
    op: Op,
    arguments: Option<Vec<Node>>,
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Operand(s) => s.value.clone(),
            Self::Expr(op, operands) => {
                let operand_s = operands.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(" ");
                let s = format!("({} {})", op.op.0, operand_s);
                s
            }
            Self::Fn(fn_node, fn_args) => {
                let mut fn_args_s = fn_args.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                
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

fn last_operator_is(op: Op, operators: &[OpNode]) -> bool {
    return operators.last().map(|op| op.op == Op::OPAREN).unwrap_or(false);
}

fn has_operator_on_stack_with_higher_precedence(op_node: Op, operators: &[OpNode]) -> bool {
    if let Some(other_node) = operators.last() {
        if other_node.op.1 == OpKind::None {
            return false;
        }
        if op_node.1 == OpKind::UnaryPrefix {
            // unary operators can only pop other unary operators.
            return other_node.op.3 > op_node.3
                && other_node.op.1 == OpKind::UnaryPrefix;
        }
        return other_node.op.3 > op_node.3
            || (other_node.op.3 == op_node.3 && op_node.2 == OpAssoc::Left);
    }
    return false;
}

fn pop_expr(operators: &mut Vec<OpNode>, operands: &mut Vec<Node>) {
    let op_node = operators.pop().unwrap();

    match op_node.op.1 {
        OpKind::Binary => {
            let rhs = operands.pop().unwrap();
            let lhs = operands.pop().unwrap();
            let node = Node::Expr(op_node, vec![lhs, rhs]);
            operands.push(node);
        }
        OpKind::UnaryPrefix => {
            let lhs = operands.pop().unwrap();
            let node = Node::Expr(op_node, vec![lhs]);
            operands.push(node);
        }
        _ => {}
    };
}

fn do_shunting_yard(value: &str) -> Node {
    let mut operators: Vec<OpNode> = Vec::new();
    let mut operands: Vec<Node> = Vec::new();
    let tokens = tokenize(value).unwrap();

    for k in 0..tokens.len() {
        let tok = &tokens[k];
        let look_for_binary_or_postfix = k > 0
            && tokens.get(k - 1).map(token_is_operand).unwrap();
        let look_for_prefix = k == 0
            || tokens.get(k - 1).map(token_is_operator).unwrap();

        if tok.value == "(" {
            if look_for_binary_or_postfix {
                let op_node = OpNode {
                    op: Op::OPAREN,
                    arguments: Some(Vec::new()),
                };
                operators.push(op_node);
            } else {
                let op_node = OpNode {
                    op: Op::OPAREN,
                    arguments: None,
                };
                operators.push(op_node);
            }
        } else if tok.value == ")" {
            let mut did_push_argument = false;
            while !last_operator_is(Op::OPAREN, &operators) {
                pop_expr(&mut operators, &mut operands);
                did_push_argument = true;
            }
            let mut oparen = operators.pop().unwrap();

            if let Some(args) = &mut oparen.arguments {
                if did_push_argument {
                    let arg = operands.pop().unwrap();
                    args.push(arg);
                }
                let call_node = operands.pop().unwrap();
                let fn_node = Node::Fn(Box::new(call_node), args.clone());

                operands.push(fn_node);
            }
        } else if tok.value == "," {
            while !last_operator_is(Op::OPAREN, &operators) {
                pop_expr(&mut operators, &mut operands);
            }
            let arg = operands.pop().unwrap();
            let oparen_node = operators.last_mut().unwrap();

            if let Some(args) = &mut oparen_node.arguments {
                args.push(arg);
            } else {
                panic!("previous lparen was not a function call.");
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
            let op_node = OpNode {
                op: op,
                arguments: None,
            };
            operators.push(op_node);
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
        do_test("(+ 1 (call (. x name)))", "1 + x.name()")
    }
    
    #[test]
    fn shunt_parse_9() {
        do_test("(call a (+ 1 2))", "a(1 + 2)")
    }

    fn do_test(expected: &str, to_parse: &str) {
        let n = do_shunting_yard(to_parse);
        assert_eq!(expected, n.to_string());
    }
}