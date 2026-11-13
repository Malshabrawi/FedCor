#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, even when launched from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 720 training runs, executed sequentially. Stops on the first failure.
# CUDA is selected automatically when available; otherwise CPU is used.
# Use an active Python environment with the project dependencies installed.

# FMNIST — original ablations
python3 experiments/run.py --dataset fmnist --variants baseline --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants dynamic --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants system --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants full --run_root=runs/main_fmnist --execute

# FMNIST — additional comparisons and epoch control
python3 experiments/run.py --dataset fmnist --variants oort --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants afl --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants random --run_root=runs/main_fmnist --execute
python3 experiments/run.py --dataset fmnist --variants full_fixed_epochs --run_root=runs/main_fmnist --execute

# CIFAR-10 — original ablations
python3 experiments/run.py --dataset cifar10 --variants baseline --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants dynamic --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants system --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants full --run_root=runs/main_cifar10 --execute

# CIFAR-10 — additional comparisons and epoch control
python3 experiments/run.py --dataset cifar10 --variants oort --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants afl --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants random --run_root=runs/main_cifar10 --execute
python3 experiments/run.py --dataset cifar10 --variants full_fixed_epochs --run_root=runs/main_cifar10 --execute

# Sensitivity — window size, threshold scaling and system penalty
python3 experiments/sensitivity.py --dataset fmnist --execute
python3 experiments/sensitivity.py --dataset cifar10 --execute
