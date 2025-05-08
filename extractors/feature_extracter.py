# import csv
# import os
#
import os
# import sys
#
# import numpy as np
#
os.environ["LIBCLANG_PATH"] = "/opt/homebrew/Cellar/llvm/19.1.7_1/lib/libclang.dylib"

import clang.cindex
clang.cindex.Config.set_library_file(os.environ["LIBCLANG_PATH"])

from extractors.power_extractor import compile_c_program, run_with_power_logging

index = clang.cindex.Index.create()
#
# # Path to the directory containing C files
# C_FILES_DIR = "generated_files"  # Change this to your folder containing C files
# OUTPUT_DIR = "output"
# # CSV file to store results
# OUTPUT_CSV = "power_dataset_optimized.csv"
#
# # Feature counters
# features = {
#     "num_functions": 0,
#     "cyclomatic_complexity": 1,  # CC starts at 1 (default case)
#     "num_loops": 0,
#     "num_function_calls": 0,
#     "num_memory_allocations": 0,
#     "num_pointer_operations": 0,
#     "num_exp_math_operations": 0,
#     "power_mw": None
# }
#
# # Memory allocation functions
# MEMORY_FUNCS = {"malloc", "calloc", "realloc", "free"}
# LOOP_NODES = {clang.cindex.CursorKind.FOR_STMT, clang.cindex.CursorKind.WHILE_STMT, clang.cindex.CursorKind.DO_STMT}
# CONDITIONAL_NODES = {clang.cindex.CursorKind.IF_STMT, clang.cindex.CursorKind.CASE_STMT, clang.cindex.CursorKind.DEFAULT_STMT}
# EXP_MATH_FUNC = {"pow", "exp", "log", "sin", "cos", "tan", "sqrt", "__builtin_pow", "__builtin_exp"}
#
#
# def analyze_ast(node):
#     """ Recursively analyze AST nodes and extract required features. """
#     global features
#
#     if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
#         features["num_functions"] += 1
#
#     elif node.kind in LOOP_NODES:
#         features["num_loops"] += 1
#         features["cyclomatic_complexity"] += 1  # Each loop adds complexity
#
#     elif node.kind in CONDITIONAL_NODES:
#         features["cyclomatic_complexity"] += 1  # Each branch adds complexity
#
#     elif node.kind == clang.cindex.CursorKind.CALL_EXPR:
#         function_name = node.spelling
#         features["num_function_calls"] += 1
#         if function_name in MEMORY_FUNCS:
#             features["num_memory_allocations"] += 1
#
#     elif node.kind == clang.cindex.CursorKind.UNARY_OPERATOR:
#         # Check if it's a pointer operation (* or & operator)
#         tokens = [t.spelling for t in node.get_tokens()]
#         if "*" in tokens or "&" in tokens:
#             features["num_pointer_operations"] += 1
#
#     elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
#         # Extract the operator symbol
#         tokens = list(node.get_tokens())
#         if len(tokens) >= 2:
#             operator = tokens[1].spelling  # Operator is typically the second token
#             if operator in {"*", "/", "%", "+", "-", "<<", ">>", "&", "|", "^"}:
#                 features["num_exp_math_operations"] += 1
#
#     elif node.kind == clang.cindex.CursorKind.CALL_EXPR:
#         func_name = node.spelling or node.displayname
#         if func_name in EXP_MATH_FUNC:
#             features["num_exp_math_operations"] += 1
#
#     for child in node.get_children():
#         analyze_ast(child)
#
#
# def extract_features_from_c_file(input_c_file):
#     """ Parse the C file and extract features. """
#     global features
#     features = {key: 0 for key in features}  # Reset feature counts
#
#     # Parse the file using Clang
#     translation_unit = index.parse(
#         input_c_file,
#         args=[
#             '-x', 'c',
#             '-std=c11',
#             f'-isysroot', '/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk',
#             '-I/opt/homebrew/opt/llvm/include',
#             '-I/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk/usr/include'
#         ]
#     )
#     analyze_ast(translation_unit.cursor)
#
#     return features
#
# def extract_features():
#     with open(os.path.join(OUTPUT_DIR, OUTPUT_CSV), "w", newline="") as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(features.keys())
#         c_file_count = 0
#         for c_file in os.listdir(C_FILES_DIR):
#             if c_file.endswith(".c"):
#                 c_path = os.path.join(C_FILES_DIR, c_file)
#                 features_map = extract_features_from_c_file(c_path)
#                 exe_file = compile_c_program(c_path)
#
#                 if exe_file:
#                     runs = 3
#                     power_mw_list = []
#                     for run in range(runs):
#                         _, power_mw = run_with_power_logging(exe_file)
#                         power_mw_list.append(power_mw)
#                     power_average = np.mean(power_mw_list)
#                     features_map['power_mw'] = int(power_average)
#                 c_file_count += 1
#                 loader = _get_loader(c_file_count)
#                 sys.stdout.write(f"\rFiles Processed: {c_file_count} - Yet to complete, Please wait...")  # Overwrites the same line
#                 sys.stdout.flush()
#                 csv_writer.writerow(features_map.values())
#         sys.stdout.write(f"\rFiles Processing completed. Total Files Processed: {c_file_count}")
#
# def _get_loader(file_count):
#     if (file_count % 4) == 0:
#         return "-"
#     elif (file_count % 4) == 1:
#         return "\\"
#     elif (file_count % 4) == 2:
#         return "/"
#     elif (file_count % 4) == 3:
#         return "-"
#
#



import csv
import os
import sys
import numpy as np

# os.environ["LIBCLANG_PATH"] = "/opt/homebrew/Cellar/llvm/19.1.7_1/lib/libclang.dylib"
#
# import clang.cindex
# clang.cindex.Config.set_library_file(os.environ["LIBCLANG_PATH"])
#
# from extractors.power_extractor import compile_c_program, run_with_power_logging
#
# index = clang.cindex.Index.create()

# Path to the directory containing C files
C_FILES_DIR = "generated_files"  # Change this to your folder containing C files
OUTPUT_DIR = "output"
# CSV file to store results
OUTPUT_CSV = "power_dataset_last.csv"

# Feature counters
def init_features():
    return {
        "num_functions": 0,
        "cyclomatic_complexity": 1,  # Starts at 1
        "num_loops": 0,
        "num_function_calls": 0,
        "num_memory_allocations": 0,
        "num_pointer_operations": 0,
        "num_exp_math_operations": 0,
        "instruction_count": 0,
        "arithmetic_instr_pct": 0.0,
        "memory_instr_pct": 0.0,
        "control_instr_pct": 0.0,
        "control_path_depth": 0,
        "power_mw": None
    }

MEMORY_FUNCS = {"malloc", "calloc", "realloc", "free"}
LOOP_NODES = {clang.cindex.CursorKind.FOR_STMT, clang.cindex.CursorKind.WHILE_STMT, clang.cindex.CursorKind.DO_STMT}
CONDITIONAL_NODES = {clang.cindex.CursorKind.IF_STMT, clang.cindex.CursorKind.CASE_STMT, clang.cindex.CursorKind.DEFAULT_STMT}
CONTROL_NODES = LOOP_NODES | CONDITIONAL_NODES
EXP_MATH_FUNC = {"pow", "exp", "log", "sin", "cos", "tan", "sqrt", "__builtin_pow", "__builtin_exp"}

ARITH_OPS = {"+", "-", "*", "/", "%"}
MEM_OPS = {"*", "[", "]"}  # Simple heuristic for memory ops
CONTROL_KEYWORDS = {"if", "for", "while", "switch"}

def analyze_ast(node, features, current_depth=0):
    features["control_path_depth"] = max(features["control_path_depth"], current_depth)
    features["instruction_count"] += 1

    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        features["num_functions"] += 1

    elif node.kind in LOOP_NODES:
        features["num_loops"] += 1
        features["cyclomatic_complexity"] += 1

    elif node.kind in CONDITIONAL_NODES:
        features["cyclomatic_complexity"] += 1

    elif node.kind == clang.cindex.CursorKind.CALL_EXPR:
        function_name = node.spelling or node.displayname
        features["num_function_calls"] += 1
        if function_name in MEMORY_FUNCS:
            features["num_memory_allocations"] += 1
        if function_name in EXP_MATH_FUNC:
            features["num_exp_math_operations"] += 1

    elif node.kind == clang.cindex.CursorKind.UNARY_OPERATOR:
        tokens = [t.spelling for t in node.get_tokens()]
        if any(op in tokens for op in MEM_OPS):
            features["num_pointer_operations"] += 1

    elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
        tokens = list(node.get_tokens())
        if len(tokens) >= 2:
            operator = tokens[1].spelling
            if operator in ARITH_OPS:
                features["num_exp_math_operations"] += 1

    # Recurse
    for child in node.get_children():
        analyze_ast(child, features, current_depth + (1 if node.kind in CONTROL_NODES else 0))

def post_process_instruction_mix(features):
    total = features["instruction_count"]
    if total == 0:
        return
    features["arithmetic_instr_pct"] = round(features["num_exp_math_operations"] / total * 100, 2)
    features["memory_instr_pct"] = round(features["num_memory_allocations"] / total * 100, 2)
    features["control_instr_pct"] = round(features["cyclomatic_complexity"] / total * 100, 2)

def extract_features_from_c_file(input_c_file):
    features = init_features()
    translation_unit = index.parse(
        input_c_file,
        args=[
            '-x', 'c',
            '-std=c11',
            f'-isysroot', '/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk',
            '-I/opt/homebrew/opt/llvm/include',
            '-I/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk/usr/include'
        ]
    )
    analyze_ast(translation_unit.cursor, features)
    post_process_instruction_mix(features)
    return features

def extract_features():
    with open(os.path.join(OUTPUT_DIR, OUTPUT_CSV), "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(init_features().keys())
        c_file_count = 0
        for c_file in os.listdir(C_FILES_DIR):
            if c_file.endswith(".c"):
                c_path = os.path.join(C_FILES_DIR, c_file)
                features_map = extract_features_from_c_file(c_path)
                exe_file = compile_c_program(c_path)

                if exe_file:
                    runs = 3
                    power_mw_list = []
                    for run in range(runs):
                        _, power_mw = run_with_power_logging(exe_file)
                        power_mw_list.append(power_mw)
                    power_average = np.mean(power_mw_list)
                    features_map['power_mw'] = int(power_average)

                c_file_count += 1
                loader = _get_loader(c_file_count)
                sys.stdout.write(f"\rFiles Processed: {c_file_count} {loader} Yet to complete, Please wait...")
                sys.stdout.flush()
                csv_writer.writerow(features_map.values())
        sys.stdout.write(f"\rFiles Processing completed. Total Files Processed: {c_file_count}\n")

def _get_loader(file_count):
    return ["-", "\\", "/", "|"][file_count % 4]
