# Templates for different code structures
from extractors.feature_extracter import EXP_MATH_FUNC

LOOP_TEMPLATE = """
void loopFunction{idx}() {{
    int sum = 0;
    int iterator = 0;
    for (iterator = 0; iterator < {iters}; iterator++) {{
        sum += iterator;
    }}
}}
"""

IF_TEMPLATE = """
void ifFunction{idx}() {{
    int x = 1, y = 2;
    int i=0;
    for(i=0;i<10;i++){{
    if (x > y) {{
        x -= y;
    }} else {{
        y -= x;
    }}
    }}
}}
"""

MEMORY_TEMPLATE = """
void memoryFunction{idx}() {{
    int size = {size};
    int *arr = (int *)malloc({size} * sizeof(int));
    if (arr) {{
        for (int i = 0; i < {size}; i++) arr[i] = i * 2;
        free(arr);
    }}
}}
"""

BITWISE_TEMPLATE = """
void bitwiseFunction{idx}() {{
    int reg = 0b00001111;
    int i=0;
    for(i=0;i<10;i++){{
    reg = reg | 1 << 5;  // Set bit 5
    reg = reg & ~(1 << 1); // Clear bit 1
    reg = reg ^ (1 << 2) ; // Toggle bit 2
    }}
}}
"""

POINTER_ARITHMETIC = """
void pointerArithmeticFunction{idx}() {{
    int a = 0;
    int i = 0;
    int arr[10] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
    int *ptr = arr;
    for(i=0;i<10;i++)
    {{
    ptr++;
    ptr--;
    }}
}}
"""
STRUCT_UNION_TEMPLATE = """
typedef struct {{
    int id;
    float voltage;
}} SensorData{idx};

void structFunction{idx}() {{
    SensorData{idx} s = {{1, 3.3}};
}}
"""

FUNCTION_POINTER_TEMPLATE = """
typedef void (*Callback{idx})(int);

void myFunction{idx}(int x) {{
    return;
}}

void functionPointerFunction{idx}() {{
    Callback{idx} cb = myFunction{idx};
    cb(10);
}}
"""

EXPONENTIAL_MATH_TEMPLATE = """
void exponentialMathFunction{idx}() {{
    double base = 2.0;
    double exponent = 3.0;
    double result = pow(base, exponent);

    int i = 0;
    for (i = 0; i < 10; i++) {{
        result = pow(result, 1.1);  // Apply exponential function iteratively
    }}
}}
"""

MAIN_TEMPLATE = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function declarations
{declarations}

int main() {{
    {function_calls}
    return 0;
}}
"""

func_template_map = {
    "loop": LOOP_TEMPLATE,
    "if": IF_TEMPLATE,
    "memory": MEMORY_TEMPLATE,
    "bitwise": BITWISE_TEMPLATE,
    "pointerArithmetic": POINTER_ARITHMETIC,
    "struct": STRUCT_UNION_TEMPLATE,
    "functionPointer": FUNCTION_POINTER_TEMPLATE,
    "exponentialMath": EXPONENTIAL_MATH_TEMPLATE
}
