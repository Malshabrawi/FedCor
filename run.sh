#!/usr/bin/env bash
# Authenticate once with wandb login, or export WANDB_API_KEY in your terminal.
# Never put an API key in this file.
export WANDB_PROJECT="${WANDB_PROJECT:-FedDynaCor}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/experiments/run_all.sh" "$@"
