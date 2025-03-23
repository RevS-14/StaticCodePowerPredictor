import argparse

from extractors.feature_extracter import extract_features
from generators.file_generator import generate_c_codes
from utils.func_template_schema import yaml_schema
from strictyaml import load

FUNCTION_TEMPLATE = "resources/func_template.yaml"

def __main__():
    parser = argparse.ArgumentParser(description="Process user information.")
    parser.add_argument("-f", "--files_count", type=int, help="req c file count", default=10)
    parser.add_argument("-k", "--keep_files", type=bool, help="Keep the generated file", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()

    func_template_yaml = FUNCTION_TEMPLATE
    file_count = args.files_count
    with open(func_template_yaml) as file:
        yaml_data = load(file.read(), yaml_schema).data

    # Generate multiple C programs
    print(f"Generating {file_count} c files...")
    for i in range(file_count):  # Change this number for more programs
        generate_c_codes(i, yaml_data)

    print("Extracting features...")
    extract_features()

    print("Dataset successfully created. Location: output/")

    if args.verbose:
        print("Verbose mode enabled.")