#include <iostream>
#include "main.h"

int main()
{
    int a = 1;
    int b = 2;

    int c = add(a, b);
    int d = sub(a, b);

    std::cout << c << " " << d << std::endl;

    return 0;
}