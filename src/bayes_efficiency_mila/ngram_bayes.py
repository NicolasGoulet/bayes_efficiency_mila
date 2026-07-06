"""CPU n-gram Bayes decomposition scoring."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json

BOS = "<bos>"
EOS = "<eos>"
CTX = "<ctx>"


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    import re

    value = "" if text is None else str(text)
    if lowercase:
        value = value.lower()
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", value)


@dataclass(frozen=True)
class NgramBayesManifest:
    run_id: str
    train_csv: Path
    candidate_csv: Path
    output_csv: Path
    utterance_column: str
    candidate_utterance_column: str
    context_column: str
    join_keys: tuple[str, ...]
    carry_columns: tuple[str, ...] = ()
    order: int = 3
    alpha: float = 0.1
    condition_tail_words: int = 16
    lowercase: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "NgramBayesManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate
            return (base / candidate).resolve()

        required = [
            "run_id",
            "train_csv",
            "candidate_csv",
            "output_csv",
            "utterance_column",
            "candidate_utterance_column",
            "context_column",
            "join_keys",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")
        order = int(payload.get("order", 3))
        if order < 1:
            raise ValueError("order must be >= 1")
        alpha = float(payload.get("alpha", 0.1))
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        condition_tail_words = int(payload.get("condition_tail_words", 16))
        if condition_tail_words < 0:
            raise ValueError("condition_tail_words must be >= 0")
        return cls(
            run_id=str(payload["run_id"]),
            train_csv=resolve(payload["train_csv"]),
            candidate_csv=resolve(payload["candidate_csv"]),
            output_csv=resolve(payload["output_csv"]),
            utterance_column=str(payload["utterance_column"]),
            candidate_utterance_column=str(payload["candidate_utterance_column"]),
            context_column=str(payload["context_column"]),
            join_keys=tuple(payload["join_keys"]),
            carry_columns=tuple(payload.get("carry_columns", ())),
            order=order,
            alpha=alpha,
            condition_tail_words=condition_tail_words,
            lowercase=bool(payload.get("lowercase", True)),
            raw=payload,
        )

    def validate_existing_inputs(self) -> None:
        missing = [str(path) for path in (self.train_csv, self.candidate_csv) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))

    @property
    def audit_json(self) -> Path:
        if self.output_csv.suffix == ".gz":
            stem = self.output_csv.with_suffix("").with_suffix("")
        else:
            stem = self.output_csv.with_suffix("")
        return stem.with_name(stem.name + ".audit.json")


class AdditiveNgramScorer:
    def __init__(self, *, order: int, alpha: float):
        self.order = order
        self.alpha = alpha
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = {EOS}

    def update(self, prefix: list[str], targets: list[str]) -> None:
        history = [BOS] * max(1, self.order - 1) + prefix
        for token in targets + [EOS]:
            self.vocab.add(token)
            for context_size in range(1, max(1, self.order)):
                context = tuple(history[-context_size:])
                self.counts[context][token] += 1
            history.append(token)

    def log2_probability(self, prefix: list[str], targets: list[str]) -> float:
        history = [BOS] * max(1, self.order - 1) + prefix
        total_log2 = 0.0
        vocab_size = max(1, len(self.vocab))
        for token in targets + [EOS]:
            context = tuple(history[-max(1, self.order - 1) :])
            counts = self.counts.get(context)
            if not counts:
                # Back off progressively, then to add-alpha unigram-like mass.
                for context_size in range(max(1, self.order - 2), 0, -1):
                    counts = self.counts.get(tuple(history[-context_size:]))
                    if counts:
                        break
            count = counts.get(token, 0) if counts else 0
            denom = (sum(counts.values()) if counts else 0) + self.alpha * vocab_size
            prob = (count + self.alpha) / denom
            total_log2 += math.log2(prob)
            history.append(token)
        return total_log2


def _condition_prefix(utterance_tokens: list[str], *, tail_words: int) -> list[str]:
    if tail_words:
        utterance_tokens = utterance_tokens[-tail_words:]
    return utterance_tokens + [CTX]


def train_scorers(manifest: NgramBayesManifest) -> tuple[AdditiveNgramScorer, AdditiveNgramScorer, dict[str, int]]:
    prior = AdditiveNgramScorer(order=manifest.order, alpha=manifest.alpha)
    likelihood = AdditiveNgramScorer(order=manifest.order, alpha=manifest.alpha)
    train_rows = 0
    usable_rows = 0
    for row in iter_csv_dicts(manifest.train_csv):
        train_rows += 1
        utterance = tokenize_words(row.get(manifest.utterance_column, ""), lowercase=manifest.lowercase)
        context = tokenize_words(row.get(manifest.context_column, ""), lowercase=manifest.lowercase)
        if not utterance:
            continue
        usable_rows += 1
        prior.update([], utterance)
        likelihood.update(_condition_prefix(utterance, tail_words=manifest.condition_tail_words), context)
    if usable_rows == 0:
        raise ValueError("No usable training utterances for n-gram Bayes scoring.")
    return prior, likelihood, {"train_rows": train_rows, "usable_train_rows": usable_rows}


def score_rows(
    manifest: NgramBayesManifest,
    prior: AdditiveNgramScorer,
    likelihood: AdditiveNgramScorer,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    scored = 0
    skipped_empty = 0
    for row in iter_csv_dicts(manifest.candidate_csv):
        utterance = tokenize_words(
            row.get(manifest.candidate_utterance_column, ""),
            lowercase=manifest.lowercase,
        )
        context = tokenize_words(row.get(manifest.context_column, ""), lowercase=manifest.lowercase)
        if not utterance:
            skipped_empty += 1
            continue
        log2_p_u = prior.log2_probability([], utterance)
        log2_p_c_given_u = likelihood.log2_probability(
            _condition_prefix(utterance, tail_words=manifest.condition_tail_words),
            context,
        )
        bayes_log2 = log2_p_u + log2_p_c_given_u
        output: dict[str, Any] = {
            "bayes_run_id": manifest.run_id,
            **{column: row.get(column, "") for column in manifest.join_keys},
            "log2_p_u": log2_p_u,
            "log2_p_c_given_u": log2_p_c_given_u,
            "bayes_log2_score_unnormalized": bayes_log2,
            "bayes_bits_unnormalized": -bayes_log2,
            "utterance_token_count": len(utterance),
            "context_token_count": len(context),
        }
        for column in manifest.carry_columns:
            output[column] = row.get(column, "")
        scored += 1
        rows.append(output)
    return rows, {"candidate_rows_scored": scored, "candidate_rows_skipped_empty": skipped_empty}


def output_fieldnames(manifest: NgramBayesManifest) -> list[str]:
    fields = [
        "bayes_run_id",
        *manifest.join_keys,
        "log2_p_u",
        "log2_p_c_given_u",
        "bayes_log2_score_unnormalized",
        "bayes_bits_unnormalized",
        "utterance_token_count",
        "context_token_count",
        *manifest.carry_columns,
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            deduped.append(field)
    return deduped


def run_ngram_bayes(manifest: NgramBayesManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    prior, likelihood, train_audit = train_scorers(manifest)
    rows, score_audit = score_rows(manifest, prior, likelihood)
    row_count = write_csv_dicts(
        manifest.output_csv,
        rows,
        fieldnames=output_fieldnames(manifest),
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.run_id,
        "row_count": row_count,
        "train_csv": str(manifest.train_csv),
        "candidate_csv": str(manifest.candidate_csv),
        "output_csv": str(manifest.output_csv),
        "train_sha256": sha256_file(manifest.train_csv),
        "candidate_sha256": sha256_file(manifest.candidate_csv),
        "output_sha256": sha256_file(manifest.output_csv),
        "order": manifest.order,
        "alpha": manifest.alpha,
        "condition_tail_words": manifest.condition_tail_words,
        **train_audit,
        **score_audit,
    }
    write_json(manifest.audit_json, audit)
    return audit
