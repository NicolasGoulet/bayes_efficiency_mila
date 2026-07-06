"""Manifest parsing for Bayes decomposition pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BayesManifest:
    run_id: str
    prior_csv: Path
    likelihood_csv: Path
    output_csv: Path
    join_keys: tuple[str, ...]
    prior_log2_column: str = "log2_p_u"
    likelihood_log2_column: str = "log2_p_c_given_u"
    carry_columns: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "BayesManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate
            return (base / candidate).resolve()

        required = ["run_id", "prior_csv", "likelihood_csv", "output_csv", "join_keys"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")

        join_keys = tuple(payload["join_keys"])
        if not join_keys:
            raise ValueError("join_keys must not be empty")

        return cls(
            run_id=str(payload["run_id"]),
            prior_csv=resolve(payload["prior_csv"]),
            likelihood_csv=resolve(payload["likelihood_csv"]),
            output_csv=resolve(payload["output_csv"]),
            join_keys=join_keys,
            prior_log2_column=str(payload.get("prior_log2_column", "log2_p_u")),
            likelihood_log2_column=str(payload.get("likelihood_log2_column", "log2_p_c_given_u")),
            carry_columns=tuple(payload.get("carry_columns", ())),
            raw=payload,
        )

    def validate_existing_inputs(self) -> None:
        missing = [str(path) for path in (self.prior_csv, self.likelihood_csv) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))

    @property
    def audit_json(self) -> Path:
        if self.output_csv.suffix == ".gz":
            stem = self.output_csv.with_suffix("").with_suffix("")
        else:
            stem = self.output_csv.with_suffix("")
        return stem.with_name(stem.name + ".audit.json")
