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
            # Generate a more diverse repeat count
            if args['repeat_count'] == 'random':
                # Mix bias toward low, medium, and high ranges
                bucket = random.choices(
                    population=['low', 'mid', 'high'],
                    weights=[3, 3, 4],  # bias toward high counts
                    k=1
                )[0]

                if bucket == 'low':
                    repeat_count = random.randint(1000, 5000)
                elif bucket == 'mid':
                    repeat_count = random.randint(7000, 10000)
                else:
                    repeat_count = random.randint(18000, 20000)
            else:
                repeat_count = args['repeat_count']

            return_type = args['returns']

            # print(f"saipavan {repeat_count}")
            for count in range(repeat_count):
                filled_args = {
                    k: (random.randint(1, 10000) if v == "random" else v)
                    for k, v in args.items()
                    if k not in ['repeat_count', 'returns']
                }
                functions.append(func_template_map[func_name].format(idx=count, **filled_args))
                declarations.append(f"{return_type} {func_name}Function{count}();")
                calls.append(f"{func_name}Function{count}();")

    main_code = MAIN_TEMPLATE.format(
        declarations="\n".join(declarations),
        function_calls="\n    ".join(calls)
    )
    full_code = main_code + "\n".join(functions) + "\n"

    os.makedirs("generated_files", exist_ok=True)
    filename = f"generated_files/generated_codes_{file_index}.c"
    with open(filename, "w") as f:
        f.write(full_code)

    os.chmod(filename, 0o644)
    os.chmod(filename, 0o755)