import os
import subprocess
import sys

input_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.lower().endswith('.wav'):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        print(f"Processing {in_path} -> {out_path}")
        subprocess.run([
            sys.executable,  # Use the current Python interpreter
            "childrenize.py",
            in_path,
            out_path,
            "-t", "1.1",
            "-f", "288",
            "-s", "1.2"
        ])