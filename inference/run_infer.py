import argparse
import subprocess
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run infer_json.py with nohup")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save output")
    parser.add_argument("--cuda_idx", type=str, default="0", help="CUDA device index")
    parser.add_argument("--log_path", type=str, default="../infer_parallel.log", help="Path to log file")
    parser.add_argument("--run_dir", type=str, default=".", help="Directory to run the command from")

    args = parser.parse_args()

    command = [
        "nohup",
        "python", "-u", "infer_json.py",
        "--json_path", args.json_path,
        "--output_dir", args.output_dir,
        "--cuda_idx", args.cuda_idx,
    ]

    log_path = os.path.abspath(args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=args.run_dir
        )

    print(f"Started infer_json.py with nohup.")
    print(f"→ PID: {process.pid}")
    print(f"→ Log: {log_path}")
