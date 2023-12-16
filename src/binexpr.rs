use crate::ast::ASTBuilder;
use crate::ast::BinaryExpr;
use crate::ast::ElementAccess;
use crate::ast::Expression;
use crate::ast::FunctionCall;
use crate::ast::MemberAccess;
use crate::ast::StatementId;
use crate::ast::UnaryPrefix;
use crate::tokenizer::Token;
use crate::tokenizer::TokenKind;
use crate::util::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
enum OperatorKind {
    None,
    Binary,
    UnaryPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Operator {
    token: TokenKind,
    precedence: i64,
    kind: OperatorKind,
}

impl Operator {
    const OPEN_PARENTHESIS_CALL: Operator = Operator::new(TokenKind::OpenParenthesis, 15, OperatorKind::None);
    const OPEN_PARENTHESIS: Operator = Operator::new(TokenKind::OpenParenthesis, 0, OperatorKind::None);
    const OPEN_BRACKET: Operator = Operator::new(TokenKind::OpenBracket, 15, OperatorKind::None);

    // https://en.wikipedia.org/wiki/Order_of_operations#Programming_languages
    const OPERATORS: &'static [Operator] = &[
        // member access
        Operator::new(TokenKind::Dot, 16, OperatorKind::Binary),
        
        // unary plus
        Operator::new(TokenKind::Plus, 14, OperatorKind::UnaryPrefix),

        // unary minus
        Operator::new(TokenKind::Minus, 14, OperatorKind::UnaryPrefix),

        // deref
        Operator::new(TokenKind::Asterisk, 14, OperatorKind::UnaryPrefix),
        
        // pointer
        Operator::new(TokenKind::Ampersand, 14, OperatorKind::UnaryPrefix),

        // not
        Operator::new(TokenKind::Exclamation, 14, OperatorKind::UnaryPrefix),

        // mul
        Operator::new(TokenKind::Asterisk, 13, OperatorKind::Binary),

        // div
        Operator::new(TokenKind::Slash, 13, OperatorKind::Binary),

        // add
        Operator::new(TokenKind::Plus, 12, OperatorKind::Binary),

        // sub
        Operator::new(TokenKind::Minus, 12, OperatorKind::Binary),

        Operator::new(TokenKind::LessThan, 10, OperatorKind::Binary),
        Operator::new(TokenKind::LessThanEquals, 10, OperatorKind::Binary),
        Operator::new(TokenKind::GreaterThan, 10, OperatorKind::Binary),
        Operator::new(TokenKind::GreaterThanEquals, 10, OperatorKind::Binary),

        Operator::new(TokenKind::EqualsEquals, 9, OperatorKind::Binary),
        Operator::new(TokenKind::ExclamationEquals, 9, OperatorKind::Binary),
        Operator::new(TokenKind::Equals, 2, OperatorKind::Binary),
    ];

    const fn new(token: TokenKind, precedence: i64, kind: OperatorKind) -> Self {
        return Self {
            token: token,
            precedence: precedence,
            kind: kind,
        };
    }
}

fn token_is_operator(token: TokenKind) -> bool {
    return Operator::OPERATORS.iter().any(|op| op.token == token);
}

pub fn token_is_binary_or_postfix_operator_or_paren(token: TokenKind) -> bool {
    return token == TokenKind::OpenParenthesis
        || token == TokenKind::OpenBracket
        || find_operator(token, OperatorKind::Binary).is_some();
}

pub fn token_is_prefix_operator_or_paren(token: TokenKind) -> bool {
    return token == TokenKind::OpenParenthesis
        || find_operator(token, OperatorKind::UnaryPrefix).is_some();
}

fn find_operator(token: TokenKind, kind: OperatorKind) -> Option<Operator> {
    return Operator::OPERATORS.iter()
        .find(|op| op.token == token && kind == op.kind)
        .cloned();
}

#[derive(Debug, Clone)]
struct OpNode {
    operator: Operator,
    token: Token,
}

fn pop_expr(parent: StatementId, operators: &mut Vec<OpNode>, operands: &mut Vec<Expression>) {
    // assert!(out_queue.len() >= 2);
    assert!(operators.len() >= 1);
    let op_node = operators.pop().unwrap();

    let expr = match op_node.operator.kind {
        OperatorKind::None => panic!(),
        OperatorKind::Binary => {
            match op_node.token.kind {
                TokenKind::Dot => {
                    let rhs = operands.pop().unwrap();
                    let lhs = operands.pop().unwrap();
                    let ident = match rhs {
                        Expression::Identifier(x) => x,
                        _ => panic!(),
                    };
                    let member_access = MemberAccess {
                        left: Box::new(lhs),
                        right: ident,
                        parent: parent,
                    };
                    Expression::MemberAccess(member_access)
                }
                _ => {
                    let rhs = operands.pop().unwrap();
                    let lhs = operands.pop().unwrap();
                    let bin_expr = BinaryExpr {
                        left: Box::new(lhs),
                        operator: op_node.token,
                        right: Box::new(rhs),
                        parent: parent,
                    };
                    Expression::BinaryExpr(bin_expr)
                }
            }
        }
        OperatorKind::UnaryPrefix => {
            let rhs = operands.pop().unwrap();
            let unary = UnaryPrefix {
                operator: op_node.token,
                expr: Box::new(rhs),
                parent: parent,
            };
            Expression::UnaryPrefix(unary)
        }
    };

    operands.push(expr);
}

fn has_operator_on_stack_with_greater_precedence(op: Operator, operators: &[OpNode]) -> bool {
    if let Some(other_op) = operators.last() {
        if other_op.operator.kind == OperatorKind::None {
            return false;
        }
        return (other_op.operator.precedence > op.precedence)
            || (other_op.operator.precedence == op.precedence
                && op.kind == OperatorKind::Binary);
    }
    return false;
}

fn last_operator_matches(token: TokenKind, operators: &[OpNode]) -> bool {
    if let Some(op) = operators.last() {
        return op.token.kind == token;
    }
    return false;
}

fn is_end_of_expression(kind: TokenKind) -> bool {
    return kind == TokenKind::Semicolon
        || kind == TokenKind::OpenBrace;
}

pub fn do_shunting_yard(ast: &mut ASTBuilder, parent: StatementId, first_operand: Option<Expression>, index_at_start_of_expression: usize, is_reading_control_flow_condition: bool) -> Result<Expression, Error> {
    // a shunting yard implementation, slightly adapted from the wikipedia
    // example. since we already have a starting left hand when we get here
    // the next token must always be an operator. instead of evaluating
    // the expression we construct AST nodes.
    //
    //   https://en.wikipedia.org/wiki/Shunting_yard_algorithm
    //
    //   -johan, 2023-11-27

    let mut operands: Vec<Expression> = Vec::new();
    let mut operators: Vec<OpNode> = Vec::new();

    if let Some(x) = first_operand {
        operands.push(x);
    }

    // a shunting-yard implementation with support for function calls.
    //   https://www.reedbeta.com/blog/the-shunting-yard-algorithm/#advanced-usage
    while !is_end_of_expression(ast.peek()?) {
        let peek_kind = ast.peek()?;
        let is_first_token_of_expression = ast.token_index == index_at_start_of_expression;
        let index = ast.token_index;
        let look_for_binary_or_postfix = !is_first_token_of_expression
            && ast.tokens.get(index - 1)
                .map(|tok| !token_is_operator(tok.kind) && tok.kind != TokenKind::OpenParenthesis)
                .unwrap_or(false);

        let look_for_unary_prefix = is_first_token_of_expression
            || ast.tokens.get(index - 1)
                .map(|tok| token_is_operator(tok.kind) || tok.kind == TokenKind::OpenParenthesis)
                .unwrap_or(false);
        
        if peek_kind == TokenKind::OpenParenthesis {
            let token = ast.consume_one_token()?;
            let is_function_call = look_for_binary_or_postfix;
            let node = OpNode {
                operator: if is_function_call {
                    Operator::OPEN_PARENTHESIS_CALL
                } else {
                    Operator::OPEN_PARENTHESIS
                },
                token: token,
            };

            // if we encounter a open parenthesis in the postfix
            // position it's a function invokation. pop any operators
            // that have higher precedence than the function call node,
            // for example member access.
            if is_function_call {
                while has_operator_on_stack_with_greater_precedence(node.operator, &operators) {
                    pop_expr(parent, &mut operators, &mut operands);
                }

                // initialize a new operand and put it on the stack.
                // its arguments will be empty for now, but we will
                // put stuff in there when we encounter commas or the
                // closing parenthesis.
                let callee = operands.pop().unwrap();
                let ident = match callee {
                    Expression::Identifier(x) => x,
                    _ => panic!("function expression was not an identifier.")
                };
                let fx = FunctionCall {
                    name_token: ident.token,
                    arguments: Vec::new(),
                    parent: parent,
                };
                operands.push(Expression::FunctionCall(fx));
            }

            operators.push(node);
        } else if peek_kind == TokenKind::CloseParenthesis {
            ast.consume_one_token()?;

            while !last_operator_matches(TokenKind::OpenParenthesis, &operators) {
                pop_expr(parent, &mut operators, &mut operands);
            }

            // the closing parethesis in the postfix position is the end
            // of a function call. if the function call had only one argument
            // we will never hit the ',' case, which means that we possibly
            // need to pop one argument off the stack.
            let prev_oparen_is_function_call = operators.last()
                .map(|op| op.operator == Operator::OPEN_PARENTHESIS_CALL)
                .unwrap_or(false);

            let operands_len = operands.len();
            let has_argument_on_stack = operands_len > 1 && matches!(operands.get(operands_len - 2), Some(Expression::FunctionCall(_)));

            if prev_oparen_is_function_call && has_argument_on_stack {
                let arg = operands.pop().unwrap();

                if let Expression::FunctionCall(fx) = operands.last_mut().unwrap() {
                    fx.arguments.push(arg);
                } else {
                    panic!("expected function expression");
                }
            }
            
            // pop the open parenthesis
            operators.pop().unwrap();
        } else if peek_kind == TokenKind::Comma {
            // push function argument onto the last function node.

            ast.consume_one_token()?;

            while !last_operator_matches(TokenKind::OpenParenthesis, &operators) {
                pop_expr(parent, &mut operators, &mut operands);
            }

            let arg = operands.pop().unwrap();

            if let Expression::FunctionCall(fx) = operands.last_mut().unwrap() {
                fx.arguments.push(arg);
            } else {
                panic!("expected function expression");
            }
        } else if peek_kind == TokenKind::OpenBracket && look_for_binary_or_postfix {
            let token = ast.consume_one_token()?;
            let node = OpNode {
                operator: Operator::OPEN_BRACKET,
                token: token,
            };

            while has_operator_on_stack_with_greater_precedence(node.operator, &operators) {
                pop_expr(parent, &mut operators, &mut operands);
            }

            let operand = operands.pop().unwrap();
            let elem_access = ElementAccess {
                left: Box::new(operand),
                right: Box::new(Expression::Void),
                parent: parent,
            };

            operands.push(Expression::ElementAccess(elem_access));
            operators.push(node);
        } else if peek_kind == TokenKind::CloseBracket && look_for_binary_or_postfix {
            ast.consume_one_token()?;

            while !last_operator_matches(TokenKind::OpenBracket, &operators) {
                pop_expr(parent, &mut operators, &mut operands);
            }

            let arg = operands.pop().unwrap();

            if let Expression::ElementAccess(elem_access) = operands.last_mut().unwrap() {
                elem_access.right = Box::new(arg);
            }

            // pop the open bracket
            operators.pop().unwrap();
        } else if token_is_operator(peek_kind) {
            let token = ast.consume_one_token()?;
            let op: Operator;

            if look_for_unary_prefix {
                op = find_operator(token.kind, OperatorKind::UnaryPrefix)
                    .unwrap();
            } else if look_for_binary_or_postfix {
                op = find_operator(token.kind, OperatorKind::Binary)
                    .unwrap();
            } else {
                panic!("could not find operator for token '{}'", token.kind);
            }

            while has_operator_on_stack_with_greater_precedence(op, &operators) {
                pop_expr(parent, &mut operators, &mut operands);
            }

            let node = OpNode {
                token: token,
                operator: op,
            };
            operators.push(node);
        } else {
            let maybe_rhs = ast.expect_expression(parent, true, is_reading_control_flow_condition)?;
            operands.push(maybe_rhs);
        }
    }

    while !operators.is_empty() {
        pop_expr(parent, &mut operators, &mut operands);
    }

    assert_eq!(1, operands.len());
    let expr = &operands[0];
    return Ok(expr.clone());
}
