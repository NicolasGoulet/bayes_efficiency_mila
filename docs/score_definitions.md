# Score Definitions

## Direct Contextual Scoring

Direct contextual scoring estimates target surprisal:

```text
-log2 p(u | c)
```

where `c` is the caretaker context and `u` is the child utterance. This is the
family used by the Mistral scoring repo when target-token surprisal is computed
only on the utterance tokens.

## Bayes Decomposition

The Bayes formulation decomposes contextual probability as:

```text
p(u | c) = p(c | u) * p(u) / p(c)
```

The CPU pilot table computes:

```text
bayes_log2_score_unnormalized = log2_p_u + log2_p_c_given_u
bayes_bits_unnormalized = -bayes_log2_score_unnormalized
```

This is unnormalized because `p(c)` is not estimated. It is valid as a
within-context ranking or decomposition diagnostic when the compared
candidates share the same context.

## Required Audit

Every production table should report:

- input row counts
- joined row counts
- missing prior rows
- missing likelihood rows
- duplicate join keys
- score column min/max
- checksum of input and output files
