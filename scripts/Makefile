CC=gcc -arch x86_64 -masm=intel -S -O0

all: clean asm llvm

clean:
	rm -rf *.ll *.s

asm: clean
	$(CC) app.c

llvm: clean
	$(CC) -emit-llvm app.c
