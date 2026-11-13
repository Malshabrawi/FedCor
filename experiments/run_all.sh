#!/usr/bin/env bash

# Run from the repository root, even when launched from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 282 training runs, executed sequentially. Continues to later commands after failures.
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

# Reduced sensitivity — Full only, seven nondefault settings, all three partitions.
# Reuse baseline and default Full from main_fmnist above: 42 additional runs (seeds 1 and 2).
python3 experiments/sensitivity.py --dataset fmnist --partitions dir spc1 spc2 --seeds 1 2 --execute
# Optional CIFAR-10 extension (42 additional runs):
# python3 experiments/sensitivity.py --dataset cifar10 --partitions dir spc1 spc2 --seeds 1 2 --execute
