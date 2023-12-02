![Build status](https://github.com/helmutschneider/trashcan/workflows/build/badge.svg)

# 🗑️ the trashcan programming language 🗑️

The trashcan language is a toy programming language that
compiles to native x86_64 assembly code. Its feature set
is purposefully limited to types that fit into the registers
of the CPU.

Features:
  - types: void, bool, int, user-defined structs (must be passed by reference)
  - math: add, sub, mul, div
  - control flow: if, else if, else, while
  - OS support: linux, macos
  - functions
  - type inference for locals

Here is a code example:
```
fun add(x: int, y: int): int {
  return x + y;
}

var x = add(1, 2);
var y = x * 2;
```

... which compiles into the following bytecode:

```
add(x: int, y: int):
  local %0, int
  add %0, x, y
  ret %0
__trashcan__main():
  local x, int
  call x, add(1, 2)
  local y, int
  mul y, x, 2
  ret void
```

... which compiles into the following x86 assembly (on linux):

```asm
.intel_syntax noprefix
.globl __trashcan__main
add:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov qword ptr [rbp - 8], rdi                  # x = rdi
  mov qword ptr [rbp - 16], rsi                 # y = rsi
  mov rax, qword ptr [rbp - 8]
  add rax, qword ptr [rbp - 16]
  mov qword ptr [rbp - 24], rax                 # %0 = x + y
  mov rax, qword ptr [rbp - 24]
  add rsp, 32
  pop rbp
  ret
__trashcan__main:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov rdi, 1
  mov rsi, 2
  call add
  mov qword ptr [rbp - 8], rax                  # x = add(1, 2)
  mov rax, qword ptr [rbp - 8]
  imul rax, 2
  mov qword ptr [rbp - 16], rax                 # y = x * 2
  mov rax, 60                                   # syscall: code exit
  mov rdi, 0                                    # syscall: argument void
  syscall
  add rsp, 16
  pop rbp
  ret
```

## Dependencies
  - rust
  - gcc or clang

## Usage
```
cargo run [example.tc]
```
