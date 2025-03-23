import argparse
import random
import os

from utils.c_func_templates import *

# Function to generate a random C program
def generate_c_codes(file_index, yaml_data):

    functions = []
    declarations = []
    calls = []

    for idx, (func_name, args) in enumerate(yaml_data.items()):
        if func_name in func_template_map.keys():
            repeat_count = args['repeat_count'] if not args['repeat_count'] == 'random' else random.randint(1, 10)
            return_type = args['returns']
            filled_args = {k: (random.randint(1, 10) if v == "random" else v) for k, v in args.items()}
            for count in range(repeat_count):
                functions.append(func_template_map[func_name].format(idx=count, **filled_args))
                declarations.append(f"{return_type} {func_name}Function{count}();")
                calls.append(f"{func_name}Function{count}();")

    main_code = MAIN_TEMPLATE.format(declarations="\n".join(declarations), function_calls="\n    ".join(calls))
    full_code = main_code + "\n".join(functions) + "\n"

    filename = f"generated_c_files/generated_codes_{file_index}.c"
    with open(filename, "w") as f:
        f.write(full_code)

    os.chmod(filename, 0o644)
    os.chmod(filename, 0o755)
