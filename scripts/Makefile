CC=gcc -arch x86_64 -O0

all: clean asm llvm

clean:
	rm -rf *.ll *.s

asm: clean
	$(CC) -S -masm=intel app.c

llvm: clean
	$(CC) -S -emit-llvm app.c
