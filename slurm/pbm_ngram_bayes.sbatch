#!/bin/bash
#SBATCH --job-name=pbm-bayes
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/bayes_efficiency_mila" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 /path/to/default_naturalistic_merged_006_023 [output_root]" >&2
  exit 2
fi

BUNDLE_ROOT="$1"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
OUTPUT_ROOT="${2:-${SCRATCH_ROOT}/communicative_efficiency_runs/pbm_ngram_bayes/${JOB_ID}}"
PYTHON_BIN="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-pbm_ngram_bayes_full79_train}"
CANDIDATE_DATASETS="${CANDIDATE_DATASETS:-Brown,Manchester,Providence}"
TRAIN_DATASETS="${TRAIN_DATASETS:-ALL}"
CONTEXT_COLUMN="${CONTEXT_COLUMN:-context_k3}"
ORDER="${ORDER:-3}"
ALPHA="${ALPHA:-0.1}"
CONDITION_TAIL_WORDS="${CONDITION_TAIL_WORDS:-16}"
SOURCE_MODELS="${SOURCE_MODELS:-real,random,unigram,bigram,trigram}"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m bayes_efficiency_mila prepare-pbm-ngram-bayes-manifest \
  --bundle-root "${BUNDLE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --candidate-datasets "${CANDIDATE_DATASETS}" \
  --train-datasets "${TRAIN_DATASETS}" \
  --context-column "${CONTEXT_COLUMN}" \
  --order "${ORDER}" \
  --alpha "${ALPHA}" \
  --condition-tail-words "${CONDITION_TAIL_WORDS}" \
  --source-models "${SOURCE_MODELS}"

MANIFEST="${OUTPUT_ROOT}/manifests/pbm_ngram_bayes_manifest.json"
"${PYTHON_BIN}" -m bayes_efficiency_mila validate-ngram-manifest --manifest "${MANIFEST}" --check-inputs
"${PYTHON_BIN}" -m bayes_efficiency_mila score-ngram-bayes --manifest "${MANIFEST}"

cat > "${OUTPUT_ROOT}/cleanup_after_rsync.sh" <<EOF
#!/bin/bash
set -euo pipefail
rm -rf -- '${OUTPUT_ROOT}'
EOF
chmod +x "${OUTPUT_ROOT}/cleanup_after_rsync.sh"

echo "Outputs written under ${OUTPUT_ROOT}"
echo "After rsyncing needed outputs, clean scratch with:"
echo "  bash '${OUTPUT_ROOT}/cleanup_after_rsync.sh'"
