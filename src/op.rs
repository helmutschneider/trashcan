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
    Expr(Op, Vec<Node>),
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Operand(s) => s.value.clone(),
            Self::Expr(op, operands) => {
                let operand_s = operands.iter().map(|o| o.to_string()).collect::<Vec<String>>().join(" ");
                let s = format!("({} {})", op.0, operand_s);
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

fn last_operator_is_oparen(operators: &[Op]) -> bool {
    return operators.last().map(|op| *op == Op::OPAREN).unwrap_or(false);
}

fn has_operator_on_stack_with_higher_precedence(op: Op, operators: &[Op]) -> bool {
    if let Some(other_op) = operators.last() {
        if other_op.1 == OpKind::None {
            return false;
        }
        if op.1 == OpKind::UnaryPrefix {
            // unary operators can only pop other unary operators.
            return other_op.3 > op.3
                && other_op.1 == OpKind::UnaryPrefix;
        }
        return other_op.3 > op.3
            || (other_op.3 == op.3 && op.2 == OpAssoc::Left);
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
    }
}

fn do_shunting_yard(value: &str) -> Node {
    let mut operators: Vec<Op> = Vec::new();
    let mut operands: Vec<Node> = Vec::new();
    let tokens = tokenize(value).unwrap();

    for k in 0..tokens.len() {
        let tok = &tokens[k];
        let can_allow_binary = k > 0
            && tokens.get(k - 1).map(token_is_operand).unwrap()
            && tokens.get(k - 1).map(|t| t.value != "(").unwrap();
        let can_allow_postfix = can_allow_binary;
        let can_allow_prefix = k == 0
            || tokens.get(k - 1).map(token_is_operator).unwrap()
            || tokens.get(k - 1).map(|t| t.value == "(").unwrap();

        if tok.value == "(" {
            operators.push(Op::OPAREN);
        } else if tok.value == ")" {
            while !last_operator_is_oparen(&operators) {
                pop_expr(&mut operators, &mut operands);
            }
            operators.pop().unwrap();
        } else if token_is_operator(tok) {
            let op: Op;
            if can_allow_prefix {
                op = find_operator(tok, OpKind::UnaryPrefix);
            } else if can_allow_binary {
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

    fn do_test(expected: &str, to_parse: &str) {
        let n = do_shunting_yard(to_parse);
        assert_eq!(expected, n.to_string());
    }
}