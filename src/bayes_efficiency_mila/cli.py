"""Command-line interface for Bayes decomposition pilots."""

from __future__ import annotations

import argparse
import json
import sys

from .big_cleaned import prepare_pbm_ngram_bayes_manifest
from .combine import run_combine
from .manifest import BayesManifest
from .ngram_bayes import NgramBayesManifest, run_ngram_bayes


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest = BayesManifest.from_path(args.manifest)
    if args.check_inputs:
        manifest.validate_existing_inputs()
    print(json.dumps({"status": "ok", "run_id": manifest.run_id}, indent=2))
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    manifest = BayesManifest.from_path(args.manifest)
    audit = run_combine(manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "audit_json": str(manifest.audit_json),
                "row_count": audit["row_count"],
            },
            indent=2,
        )
    )
    return 0


def cmd_estimate_context_likelihood(args: argparse.Namespace) -> int:
    manifest = BayesManifest.from_path(args.manifest)
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "run_id": manifest.run_id,
                "message": (
                    "Context-likelihood estimation p(c | u) has not been "
                    "implemented yet. Add a tested reverse discourse or neural "
                    "likelihood estimator before using this Slurm path."
                ),
            },
            indent=2,
        )
    )
    return 2


def cmd_validate_ngram_manifest(args: argparse.Namespace) -> int:
    manifest = NgramBayesManifest.from_path(args.manifest)
    if args.check_inputs:
        manifest.validate_existing_inputs()
    print(json.dumps({"status": "ok", "run_id": manifest.run_id}, indent=2))
    return 0


def cmd_score_ngram_bayes(args: argparse.Namespace) -> int:
    manifest = NgramBayesManifest.from_path(args.manifest)
    audit = run_ngram_bayes(manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "audit_json": str(manifest.audit_json),
                "row_count": audit["row_count"],
            },
            indent=2,
        )
    )
    return 0


def _parse_dataset_filter(value: str | None) -> set[str] | None:
    if not value or value.upper() == "ALL":
        return None
    datasets = {item.strip() for item in value.split(",") if item.strip()}
    return datasets or None


def _parse_required_dataset_filter(value: str) -> set[str]:
    datasets = {item.strip() for item in value.split(",") if item.strip()}
    if not datasets:
        raise ValueError("At least one candidate dataset is required.")
    return datasets


def cmd_prepare_pbm_ngram_bayes_manifest(args: argparse.Namespace) -> int:
    source_models = tuple(item.strip() for item in args.source_models.split(",") if item.strip())
    audit = prepare_pbm_ngram_bayes_manifest(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        run_id=args.run_id,
        candidate_datasets=_parse_required_dataset_filter(args.candidate_datasets),
        train_datasets=_parse_dataset_filter(args.train_datasets),
        context_column=args.context_column,
        order=args.order,
        alpha=args.alpha,
        condition_tail_words=args.condition_tail_words,
        source_models=source_models,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bayes-efficiency-mila")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-inputs", action="store_true")
    validate.set_defaults(func=cmd_validate_manifest)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--manifest", required=True)
    combine.set_defaults(func=cmd_combine)

    likelihood = subparsers.add_parser("estimate-context-likelihood")
    likelihood.add_argument("--manifest", required=True)
    likelihood.set_defaults(func=cmd_estimate_context_likelihood)

    validate_ngram = subparsers.add_parser("validate-ngram-manifest")
    validate_ngram.add_argument("--manifest", required=True)
    validate_ngram.add_argument("--check-inputs", action="store_true")
    validate_ngram.set_defaults(func=cmd_validate_ngram_manifest)

    ngram_bayes = subparsers.add_parser("score-ngram-bayes")
    ngram_bayes.add_argument("--manifest", required=True)
    ngram_bayes.set_defaults(func=cmd_score_ngram_bayes)

    prep_pbm = subparsers.add_parser("prepare-pbm-ngram-bayes-manifest")
    prep_pbm.add_argument("--bundle-root", required=True)
    prep_pbm.add_argument("--output-root", required=True)
    prep_pbm.add_argument("--run-id", default="pbm_ngram_bayes_full79_train")
    prep_pbm.add_argument("--candidate-datasets", default="Brown,Manchester,Providence")
    prep_pbm.add_argument("--train-datasets", default="ALL")
    prep_pbm.add_argument("--context-column", default="context_k3")
    prep_pbm.add_argument("--order", type=int, default=3)
    prep_pbm.add_argument("--alpha", type=float, default=0.1)
    prep_pbm.add_argument("--condition-tail-words", type=int, default=16)
    prep_pbm.add_argument("--source-models", default="real,random,unigram,bigram,trigram")
    prep_pbm.set_defaults(func=cmd_prepare_pbm_ngram_bayes_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
