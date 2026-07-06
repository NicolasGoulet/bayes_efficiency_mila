"""Command-line interface for Bayes decomposition pilots."""

from __future__ import annotations

import argparse
import json
import sys

from .combine import run_combine
from .manifest import BayesManifest


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
