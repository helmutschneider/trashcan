![Build status](https://github.com/helmutschneider/trashcan/workflows/build/badge.svg)

# 🗑️ trashcan: a toy programming language 🗑️

The trashcan language is an extremely simple programming language
that compiles to native x86_64 machine code. Its feature set is
purposefully limited to types that fit into the registers of the CPU.

Features:
  - integers
  - addition, subtraction
  - functions
  - if-statements
  - exit code is the return value of "main"
  - linux/macos support

A code example:
```
fun add(x: int, y: int): int {
  return x + y;
}

fun main(): int {
  return add(1, 2);
}
```

Which compiles down to the following very crude MacOS x86 assembly:
```asm
.intel_syntax noprefix
.globl main
add:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov qword ptr [rbp - 8], rdi
  mov qword ptr [rbp - 16], rsi
  mov rax, qword ptr [rbp - 8]
  add rax, qword ptr [rbp - 16]
  mov qword ptr [rbp - 24], rax
  mov rax, qword ptr [rbp - 24]
  add rsp, 32
  pop rbp
  ret
main:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov rdi, 1
  mov rsi, 2
  call add
  mov qword ptr [rbp - 8], rax
  mov rax, 33554433
  mov rdi, qword ptr [rbp - 8]
  syscall
  add rsp, 16
  pop rbp
  ret
```
