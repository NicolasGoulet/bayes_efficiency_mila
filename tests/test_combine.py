from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bayes_efficiency_mila.combine import run_combine
from bayes_efficiency_mila.manifest import BayesManifest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BayesCombineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
