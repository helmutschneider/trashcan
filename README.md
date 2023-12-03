![Build status](https://github.com/helmutschneider/trashcan/workflows/build/badge.svg)

# 🗑️ the trashcan programming language 🗑️

The trashcan language is a toy programming language that
compiles to native x86_64 assembly code. Its feature set
is purposefully limited to types that fit into the registers
of the CPU.

Features:
  - types: void, bool, int, array, user-defined structs (must be passed by reference)
  - math: add, sub, mul, div
  - control flow: if, else if, else, while
  - OS support: linux, macos
  - functions
  - type inference for locals

## Dependencies
  - rust
  - gcc or clang

## Usage
```
# compile a program to an executable...
cargo run filename.tc

# compile a program to bytecode...
cargo run filename.tc -b

# compile a program to assembly...
cargo run filename.tc -s
```

## Example program
```shell
fun do_spooky_add(a: int, b: int): int {
  return a + b;
}

var strings = ["hello!\n", "world!\n"];
var numbers = [420, 69];
var sum = 0;
var k = 0;

while k != numbers.length {
  sum = do_spooky_add(sum, numbers[k]);
  var name = strings[k];
  print(&name);

  k = k + 1;
}

assert(sum == 489);
assert(k == 2);

print(&"cowabunga!\n");
```
