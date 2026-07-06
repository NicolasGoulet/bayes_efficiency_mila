# TODO.md

Production checklist for `bayes_efficiency_mila`.

This repo adds Bayes-style decomposition analyses to the existing
communicative-efficiency project. It does not replace direct Mistral
`p(u | c)` surprisal; it adds a separate decomposition:

```text
p(u | c) = p(c | u) * p(u) / p(c)
```

## Core Contract

- [x] Keep direct contextual Mistral scoring separate from Bayes decomposition
      scoring.
- [x] Label Bayes pilot scores as unnormalized unless `p(c)` is estimated.
- [x] Preserve `context_id`, `utterance_id`, `source_model`, age, child,
      dataset, context, and effort provenance wherever available.
- [x] Keep Git limited to code, tests, docs, Slurm scripts, tiny synthetic
      fixtures, and manifest templates. Move real prior/likelihood tables,
      candidate clouds, and scored outputs with `rsync`.
- [x] On Mila, keep the permanent Git checkout in `$HOME` beside the other
      modular repos; write job outputs, temporary data, candidate clouds, and
      rsynced full datasets under `$SCRATCH`, then remove scratch job
      directories after retrieval.
- [x] Provide Slurm scripts that `cd` to repo root and set `PYTHONPATH=src`.
- [ ] Add production manifests pointing to compact exported prior/likelihood
      inputs from Mila or the main brain repo.
- [ ] Add a manifest audit command that checks join-key uniqueness, required
      columns, score sign conventions, and duplicate candidate rows.

## CPU Bayes Tables

- [x] Combine existing prior and likelihood score tables into an
      unnormalized Bayes table.
- [x] Write JSON audit sidecars with row counts, duplicate-key counts, missing
      prior/likelihood rows, and checksums.
- [x] Implement CPU n-gram estimation of `p(u)` from training utterances.
- [x] Implement CPU reverse/discourse n-gram estimation of `p(c | u)` from
      observed context/utterance pairs.
- [x] Add a one-command CPU pilot that trains `p(u)`, trains `p(c | u)`, scores
      candidate rows, and writes a decomposition table.
- [x] Add tests for probability signs:
      `bayes_log2_score_unnormalized = log2_p_u + log2_p_c_given_u` and
      `bayes_bits_unnormalized = -bayes_log2_score_unnormalized`.

## GPU / Neural Likelihood

- [ ] Implement a real neural `p(c | u)` estimator only after the CPU n-gram
      decomposition is validated.
- [ ] Keep neural likelihood outputs in separate columns from n-gram likelihood
      outputs.
- [ ] Write likelihood model configs, checkpoints, score audits, and exact
      scoring prompts/configuration.
- [ ] Add GPU Slurm scripts only after the estimator produces real outputs in
      a tested smoke run.

## Scientific Comparisons

- [ ] Compare direct Mistral `sum_bits` with Bayes-decomposed scores on the
      same real-child utterance rows.
- [ ] Compare real child utterances against random, n-gram, LSTM, and response
      cloud candidates using the same Bayes score columns.
- [ ] Report whether `p(u)` prior, `p(c | u)` compatibility, or both drive
      developmental effects.
- [ ] Treat the Bayes decomposition as an added analysis family, not as a
      replacement for existing Route 1/Route 2 results.
- [ ] Add PBM cleaned-data integration manifests using existing
      `compute_surprisal_mila/data/{Brown,Manchester,Providence}/*/chi.csv`
      as the first real-data test layer before full strict-naturalistic data.

## Verification Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/bayes_efficiency_mila_pycache PYTHONPATH=src python3 -m unittest discover -s tests
bash -n slurm/*.sbatch
PYTHONPATH=src python3 -m bayes_efficiency_mila validate-manifest --manifest configs/bayes_pilot_example.json
```
