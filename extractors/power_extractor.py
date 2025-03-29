import os
import subprocess
import time
import csv
from concurrent.futures import ProcessPoolExecutor

# Paths
C_FILES_DIR = "../generated_files"
OUTPUT_CSV = "power_dataset_5k.csv"
NUM_WORKERS = os.cpu_count()  # Use all available CPU cores


def compile_c_program(c_file):
    """Compiles a C file using clang and returns the executable name."""
    exe_file = c_file.replace(".c", "")
    compile_cmd = ["clang", c_file, "-o", exe_file, "-lm"]  # Enable optimizations & multi-threaded build

    try:
        subprocess.run(compile_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return exe_file
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {c_file}: {e}")
        return None


def extract_power_consumption(log_file):
    """Parses the power log file and extracts average power consumption."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        power_values = [
            float(line.split(":")[1].split(" ")[1])
            for line in lines if "Combined Power" in line
        ]

        return sum(power_values) / len(power_values) if power_values else None
    except Exception as e:
        print(f"Error extracting power consumption: {e}")
        return None


def run_with_power_logging(exe_file):
    """Runs the compiled executable while capturing power consumption."""
    power_log_file = f"power_log.txt"

    power_cmd = f"sudo powermetrics --samplers cpu_power -i 500 -n 5 | tee {power_log_file} > /dev/null"  # Reduce interval & samples
    power_proc = subprocess.Popen(power_cmd, shell=True, preexec_fn=os.setpgrp)

    time.sleep(2)

    start_time = time.time()
    try:
        subprocess.run(["./" + exe_file], check=True, timeout=10)  # Prevent infinite loops
    except subprocess.CalledProcessError as e:
        print(f"Execution failed for {exe_file}: {e}")
        return None, None
    except subprocess.TimeoutExpired:
        print(f"Execution timed out for {exe_file}")
        return None, None
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Stop power logging efficiently
    power_proc.terminate()
    subprocess.run(["sudo", "pkill", "-f", "powermetrics"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    power_mw = extract_power_consumption(power_log_file)
    return execution_time, power_mw


def process_c_file(c_file):
    """Processes a single C file: compile, execute, measure power, and return results."""
    c_path = os.path.join(C_FILES_DIR, c_file)
    exe_file = compile_c_program(c_path)

    count = os.getpid()  # Use process ID to avoid filename conflicts
    if exe_file:
        execution_time, power_mw = run_with_power_logging(exe_file)
        if execution_time is not None and power_mw is not None:
            return c_file, execution_time, power_mw
    return None


def main():
    """Runs multiple C files in parallel and writes results to CSV."""
    c_files = [f for f in os.listdir(C_FILES_DIR) if f.endswith(".c")]

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["file_name", "execution_time_ms", "power_mW"])  # CSV header

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = executor.map(process_c_file, c_files)

        samples_collected = 1
        for result in results:
            if result:
                csv_writer.writerow(result)
                print(f" Samples collected: {samples_collected}", flush=True)
                samples_collected += 1


if __name__ == "__main__":
    main()