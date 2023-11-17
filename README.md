![Build status](https://github.com/helmutschneider/trashcan/workflows/build/badge.svg)

# 🗑️ the trashcan programming language 🗑️

The trashcan language is a toy programming language that
compiles to native x86_64 assembly code. Its feature set
is purposefully limited to types that fit into the registers
of the CPU.

Features:
  - integers
  - addition, subtraction
  - functions
  - if-statements
  - exit code is the return value of "main"
  - linux/macos support

Here is a code example:
```
fun add(x: int, y: int): int {
  return x + y;
}

fun main(): int {
  return add(1, 2);
}
```

... which compiles into the following bytecode:

```
add(x, y):
  %0 = add x, y
  ret %0
main():
  %0 = call add(1, 2)
  ret %0
```

... which compiles into the following x86 assembly:

```asm
.intel_syntax noprefix                          
.globl main                                     
add:                                            
  push rbp                                      
  mov rbp, rsp                                  
  sub rsp, 32                                   
  mov qword ptr [rbp - 8], rdi                  # add(): argument x to stack
  mov qword ptr [rbp - 16], rsi                 # add(): argument y to stack
  mov rax, qword ptr [rbp - 8]                  # add: lhs argument x
  add rax, qword ptr [rbp - 16]                 # add: rhs argument y
  mov qword ptr [rbp - 24], rax                 # add: result to stack
  mov rax, qword ptr [rbp - 24]                 
  add rsp, 32                                   
  pop rbp                                       
  ret                                           
main:                                           
  push rbp                                      
  mov rbp, rsp                                  
  sub rsp, 16                                   
  mov rdi, 1                                    # add(): argument 1 into register
  mov rsi, 2                                    # add(): argument 2 into register
  call add                                      
  mov qword ptr [rbp - 8], rax                  # add(): return value to stack
  mov rax, 33554433                             # syscall: code exit
  mov rdi, qword ptr [rbp - 8]                  # syscall: argument %0
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
