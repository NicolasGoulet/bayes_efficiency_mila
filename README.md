# bayes_efficiency_mila

Bayes-style communicative-efficiency scoring pilots for child-language
analyses.

This repo is intentionally separate from direct Mistral surprisal scoring. Its
job is to build and audit decomposition-style scores such as:

```text
p(u | c) = p(c | u) * p(u) / p(c)
```

For comparisons within the same fixed context `c`, `p(c)` is a constant, so a
pilot score can use:

```text
log2 p(u | c) = log2 p(u) + log2 p(c | u) + constant_for_c
```

The CPU code in this scaffold combines already-estimated prior and likelihood
tables. It does not estimate neural likelihoods yet.

## Repo Boundary

- `communicative_efficiency`: brain/reporting/data-linking repo.
- `compute_surprisal_mila`: direct neural target surprisal scoring.
- `generate_baselines_mila`: generated baseline utterance creation.
- `bayes_efficiency_mila`: Bayes decomposition pilots and likelihood scoring.

## Compute Lanes

CPU-first:

- combine `p(u)` prior tables and `p(c | u)` likelihood tables
- validate key coverage and sign conventions
- build compact Bayes pilot tables
- compare Bayes-style scores with direct Mistral scores

CPU smoke / GPU production:

- reverse discourse models for `p(c | u)`
- neural compatibility scorers

Mila GPU:

- large neural estimates of `p(c | u)`
- large candidate-cloud Bayes scoring

## Quick Start

Validate a manifest:

```bash
python3 -m bayes_efficiency_mila validate-manifest --manifest configs/bayes_pilot_example.json
```

Build a CPU pilot table:

```bash
python3 -m bayes_efficiency_mila combine --manifest configs/bayes_pilot_example.json
```

Run the PBM Bayes n-gram production pilot on Mila after the strict-naturalistic
bundle has been extracted under scratch. By default, training uses all 79
children in the bundle while candidates are PBM real + n-gram baseline
utterances.

```bash
cd "$HOME/communicative_efficiency_repos/bayes_efficiency_mila"
sbatch --output="$SCRATCH/pbm-bayes-%j.out" \
  slurm/pbm_ngram_bayes.sbatch \
  "$SCRATCH/communicative_efficiency_data/big_cleaned_dataset/default_naturalistic_merged_006_023"
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Data Policy

Do not commit large prior tables, likelihood tables, candidate clouds, scored
outputs, or model checkpoints. Transfer large inputs and outputs with `rsync`
or cluster storage.

On Mila, keep the permanent Git checkout in `$HOME` beside the other modular
repos. Put production manifests, temporary files, candidate clouds, and run
outputs under `$SCRATCH`, then remove scratch job directories after outputs have
been rsynced back or are no longer needed.
