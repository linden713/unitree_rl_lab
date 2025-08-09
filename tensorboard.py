#!/usr/bin/env python3
import subprocess
import sys
import os

log_dir = "/home/lch/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity"

if not os.path.exists(log_dir):
    print(f"Error: Log directory does not exist: {log_dir}")
    sys.exit(1)

print(f"Starting TensorBoard for: {log_dir}")
print("Open http://localhost:6006 in your browser")

try:
    subprocess.run(["tensorboard", "--logdir", log_dir, "--port", "6006"], check=True)
except KeyboardInterrupt:
    print("\nTensorBoard stopped.")
except subprocess.CalledProcessError as e:
    print(f"Error running tensorboard: {e}")
except FileNotFoundError:
    print("Error: tensorboard command not found. Install with: pip install tensorboard")