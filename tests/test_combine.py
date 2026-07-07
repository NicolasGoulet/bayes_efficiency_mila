from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bayes_efficiency_mila.big_cleaned import prepare_pbm_ngram_bayes_manifest
from bayes_efficiency_mila.combine import run_combine
from bayes_efficiency_mila.manifest import BayesManifest
from bayes_efficiency_mila.ngram_bayes import NgramBayesManifest, run_ngram_bayes


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BayesCombineTests(unittest.TestCase):
    def test_prepare_pbm_ngram_bayes_manifest_from_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = root / "bundle"
            child_dir = bundle / "preprocessed_data" / "Brown" / "Adam"
            child_dir.mkdir(parents=True)
            write_csv(
                bundle / "manifest.csv",
                [
                    {
                        "dataset": "Brown",
                        "child_id": "Adam",
                        "child_scoring_ready": "1",
                        "child_scoring_csv": "preprocessed_data/Brown/Adam/chi.surprisal_scoring.csv",
                    }
                ],
            )
            write_csv(
                child_dir / "chi.surprisal_scoring.csv",
                [
                    {
                        "dataset": "Brown",
                        "child_id": "Adam",
                        "source_group": "Brown",
                        "session_id": "1",
                        "age_months": "27.1",
                        "file": "Adam/a.cha",
                        "line_no": "10",
                        "utt_id": "1",
                        "context_k3": "do you want milk?",
                        "chi_utterance_clean": "more milk",
                        "random_model_utterance_bin6": "go home",
                        "unigram_model_utterance_bin6": "want milk",
                        "bigram_model_utterance_bin6": "more cookie",
                        "trigram_model_utterance_bin6": "more milk",
                    }
                ],
            )

            audit = prepare_pbm_ngram_bayes_manifest(
                bundle_root=bundle,
                output_root=root / "run",
                run_id="unit-pbm",
                candidate_datasets={"Brown"},
                train_datasets={"Brown"},
            )

            self.assertEqual(audit["train_row_count"], 1)
            self.assertEqual(audit["candidate_row_count"], 5)
            manifest = NgramBayesManifest.from_path(audit["manifest_json"])
            self.assertEqual(manifest.run_id, "unit-pbm")
            self.assertEqual(manifest.join_keys, ("row_uid", "source_model"))

    def test_combines_prior_and_likelihood_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prior = root / "prior.csv"
            likelihood = root / "likelihood.csv"
            output = root / "scores.csv.gz"
            manifest_path = root / "manifest.json"

            write_csv(
                prior,
                [
                    {
                        "context_id": "c1",
                        "utterance_id": "u1",
                        "source_model": "real",
                        "log2_p_u": "-3.0",
                        "age_bin": "024-029",
                    },
                    {
                        "context_id": "c1",
                        "utterance_id": "u2",
                        "source_model": "trigram",
                        "log2_p_u": "-5.0",
                        "age_bin": "024-029",
                    },
                ],
            )
            write_csv(
                likelihood,
                [
                    {
                        "context_id": "c1",
                        "utterance_id": "u1",
                        "source_model": "real",
                        "log2_p_c_given_u": "-7.0",
                    }
                ],
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "unit",
                        "prior_csv": "prior.csv",
                        "likelihood_csv": "likelihood.csv",
                        "output_csv": "scores.csv.gz",
                        "join_keys": ["context_id", "utterance_id", "source_model"],
                        "carry_columns": ["age_bin"],
                    }
                ),
                encoding="utf-8",
            )

            audit = run_combine(BayesManifest.from_path(manifest_path))

            self.assertEqual(audit["row_count"], 1)
            self.assertEqual(audit["missing_likelihood_rows"], 1)
            self.assertTrue(output.exists())

            with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["bayes_log2_score_unnormalized"], "-10.0")
            self.assertEqual(rows[0]["bayes_bits_unnormalized"], "10.0")
            self.assertEqual(rows[0]["age_bin"], "024-029")

    def test_ngram_bayes_scores_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.csv"
            candidates = root / "candidates.csv"
            output = root / "ngram_bayes.csv.gz"
            manifest_path = root / "manifest.json"

            write_csv(
                train,
                [
                    {
                        "context_id": "c1",
                        "utterance_id": "t1",
                        "source_model": "real",
                        "context_text": "do you want",
                        "utterance_clean": "more milk",
                        "target_utterance_clean": "",
                    },
                    {
                        "context_id": "c2",
                        "utterance_id": "t2",
                        "source_model": "real",
                        "context_text": "where did it go",
                        "utterance_clean": "go there",
                        "target_utterance_clean": "",
                    },
                ],
            )
            write_csv(
                candidates,
                [
                    {
                        "context_id": "c1",
                        "utterance_id": "u1",
                        "source_model": "real",
                        "context_text": "do you want",
                        "target_utterance_clean": "more milk",
                    },
                    {
                        "context_id": "c1",
                        "utterance_id": "u2",
                        "source_model": "empty",
                        "context_text": "do you want",
                        "target_utterance_clean": "...",
                    },
                ],
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "ngram-unit",
                        "train_csv": "train.csv",
                        "candidate_csv": "candidates.csv",
                        "output_csv": "ngram_bayes.csv.gz",
                        "utterance_column": "utterance_clean",
                        "candidate_utterance_column": "target_utterance_clean",
                        "context_column": "context_text",
                        "join_keys": ["context_id", "utterance_id", "source_model"],
                        "carry_columns": ["context_text", "target_utterance_clean"],
                        "order": 2,
                        "alpha": 0.1,
                    }
                ),
                encoding="utf-8",
            )

            audit = run_ngram_bayes(NgramBayesManifest.from_path(manifest_path))

            self.assertEqual(audit["row_count"], 1)
            self.assertEqual(audit["candidate_rows_skipped_empty"], 1)
            with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertLess(float(rows[0]["log2_p_u"]), 0.0)
            self.assertLess(float(rows[0]["log2_p_c_given_u"]), 0.0)
            self.assertAlmostEqual(
                float(rows[0]["bayes_bits_unnormalized"]),
                -float(rows[0]["bayes_log2_score_unnormalized"]),
            )


if __name__ == "__main__":
    unittest.main()
