#include <stdio.h>

static int add(int x, int y) {
    return x + y;
}

int main(void) {
    int x = add(420, 69);

    printf("Added! %d\n", x);

    return 0;
}
