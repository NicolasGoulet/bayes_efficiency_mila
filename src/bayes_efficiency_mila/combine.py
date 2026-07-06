"""Combine prior and likelihood tables into Bayes-style pilot scores."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from .manifest import BayesManifest


def _key(row: dict[str, str], join_keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in join_keys)


def _to_float(row: dict[str, str], column: str, *, key: tuple[str, ...]) -> float:
    value = row.get(column, "")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {column} at key {key}: {value!r}") from exc


def _load_unique_rows(
    path,
    *,
    join_keys: tuple[str, ...],
) -> tuple[dict[tuple[str, ...], dict[str, str]], int, int]:
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    duplicate_count = 0
    total = 0
    for row in iter_csv_dicts(path):
        total += 1
        key = _key(row, join_keys)
        if key in rows:
            duplicate_count += 1
            continue
        rows[key] = row
    return rows, total, duplicate_count


def combine_rows(manifest: BayesManifest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_rows, prior_total, prior_duplicates = _load_unique_rows(
        manifest.prior_csv,
        join_keys=manifest.join_keys,
    )
    likelihood_rows, likelihood_total, likelihood_duplicates = _load_unique_rows(
        manifest.likelihood_csv,
        join_keys=manifest.join_keys,
    )

    output_rows: list[dict[str, Any]] = []
    missing_likelihood = 0
    for key, prior in prior_rows.items():
        likelihood = likelihood_rows.get(key)
        if likelihood is None:
            missing_likelihood += 1
            continue
        log2_p_u = _to_float(prior, manifest.prior_log2_column, key=key)
        log2_p_c_given_u = _to_float(
            likelihood,
            manifest.likelihood_log2_column,
            key=key,
        )
        bayes_log2_score = log2_p_u + log2_p_c_given_u
        row: dict[str, Any] = {
            "bayes_run_id": manifest.run_id,
            **{column: prior.get(column, likelihood.get(column, "")) for column in manifest.join_keys},
            "log2_p_u": log2_p_u,
            "log2_p_c_given_u": log2_p_c_given_u,
            "bayes_log2_score_unnormalized": bayes_log2_score,
            "bayes_bits_unnormalized": -bayes_log2_score,
        }
        for column in manifest.carry_columns:
            row[column] = prior.get(column, likelihood.get(column, ""))
        output_rows.append(row)

    missing_prior = len(set(likelihood_rows) - set(prior_rows))
    audit = {
        "prior_rows": prior_total,
        "likelihood_rows": likelihood_total,
        "prior_duplicate_keys": prior_duplicates,
        "likelihood_duplicate_keys": likelihood_duplicates,
        "joined_rows": len(output_rows),
        "missing_likelihood_rows": missing_likelihood,
        "missing_prior_rows": missing_prior,
    }
    return output_rows, audit


def output_fieldnames(manifest: BayesManifest) -> list[str]:
    fields = [
        "bayes_run_id",
        *manifest.join_keys,
        "log2_p_u",
        "log2_p_c_given_u",
        "bayes_log2_score_unnormalized",
        "bayes_bits_unnormalized",
        *manifest.carry_columns,
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            deduped.append(field)
    return deduped


def run_combine(manifest: BayesManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    rows, audit = combine_rows(manifest)
    row_count = write_csv_dicts(
        manifest.output_csv,
        rows,
        fieldnames=output_fieldnames(manifest),
    )
    audit_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.run_id,
        "row_count": row_count,
        "prior_csv": str(manifest.prior_csv),
        "likelihood_csv": str(manifest.likelihood_csv),
        "output_csv": str(manifest.output_csv),
        "prior_sha256": sha256_file(manifest.prior_csv),
        "likelihood_sha256": sha256_file(manifest.likelihood_csv),
        "output_sha256": sha256_file(manifest.output_csv),
        "manifest": {
            key: str(value) if key.endswith("_csv") else value
            for key, value in asdict(manifest).items()
            if key != "raw"
        },
        **audit,
    }
    write_json(manifest.audit_json, audit_payload)
    return audit_payload
