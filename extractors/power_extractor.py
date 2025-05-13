import os
import re
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

def extract_power_consumption(log_file, target_freq=1000, alpha=2.0):
    """Extracts CPU power values, normalizes with frequency, and ensures consistency."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        cpu_power_values = []
        cpu_freq_values = []

        for line in lines:
            cpu_power_match = re.search(r"CPU Power:\s*([\d.]+)\s*mW", line)
            cpu_freq_match = re.search(r"CPU \d+ frequency:\s*([\d.]+)\s*MHz", line)

            if cpu_power_match:
                cpu_power_values.append(float(cpu_power_match.group(1)))

            if cpu_freq_match:
                cpu_freq_values.append(float(cpu_freq_match.group(1)))

        if not cpu_power_values or not cpu_freq_values:
            raise ValueError("Missing CPU power or frequency values in the log.")

        # Compute averages
        avg_cpu_power = sum(cpu_power_values) / len(cpu_power_values)
        avg_cpu_freq = sum(cpu_freq_values) / len(cpu_freq_values)

        # Normalize power using frequency scaling
        norm_cpu_power = avg_cpu_power * (target_freq / avg_cpu_freq) ** alpha

        return int(norm_cpu_power)

    except Exception as e:
        print(f"Error extracting CPU power: {e}")
        return None

def extract_power_consumption_old(log_file):
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


def run_with_power_logging_old(exe_file):
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


import subprocess
import os
import time
import signal


def run_with_power_logging(exe_file):
    """Runs the compiled executable while capturing net power consumption (active - idle)."""

    idle_log_before = "idle_power_log_before.txt"
    active_log = "active_power_log.txt"
    idle_log_after = "idle_power_log_after.txt"


    def record_power(filename, duration=3):
        cmd = f"sudo powermetrics --samplers cpu_power -i 200 > {filename}"
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setpgrp)
        time.sleep(duration)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(1)  # Allow logs to flush

    # Step 1: Record idle power
    #
    # print("[*] Recording idle power before start...")
    record_power(idle_log_before, duration=3)

    # Step 2: Start power logging and program execution
    # print("[*] Running program and recording active power...")
    power_cmd = f"sudo powermetrics --samplers cpu_power -i 100 > {active_log}"
    power_proc = subprocess.Popen(power_cmd, shell=True, preexec_fn=os.setpgrp)
    time.sleep(1.5)  # Allow powermetrics to start

    start_time = time.time()
    try:
        subprocess.run(["./" + exe_file], check=True, timeout=10)
    except subprocess.CalledProcessError as e:
        # print(f"[!] Execution failed for {exe_file}: {e}")
        os.killpg(os.getpgid(power_proc.pid), signal.SIGTERM)
        return None, None
    except subprocess.TimeoutExpired:
        # print(f"[!] Execution timed out for {exe_file}")
        os.killpg(os.getpgid(power_proc.pid), signal.SIGTERM)
        return None, None
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # in ms

    # Stop power logging
    os.killpg(os.getpgid(power_proc.pid), signal.SIGTERM)
    time.sleep(1)  # Let powermetrics flush

    # print("[*] Recording idle power after start...")
    record_power(idle_log_after, duration=3)

    # Step 3: Compute net power
    idle_power_before = extract_power_consumption(idle_log_before)
    active_power = extract_power_consumption(active_log)
    idle_power_after = extract_power_consumption(idle_log_after)

    if idle_power_before is not None and active_power is not None and idle_power_after is not None:
        baseline_power =(idle_power_before + idle_power_after)/2
        net_power = max(active_power - baseline_power, 0)
    else:
        net_power = None

    return execution_time, net_power

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