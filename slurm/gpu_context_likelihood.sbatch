#!/bin/bash
#SBATCH --job-name=bayes-lik
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/bayes_efficiency_mila" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MANIFEST="${1:-configs/bayes_pilot_example.json}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p logs
"${PYTHON_BIN}" -m bayes_efficiency_mila estimate-context-likelihood --manifest "${MANIFEST}"
