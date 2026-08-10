from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data.config import load_data_config
from .data.errors import DataConfigError, DataValidationError
from .data.manifest import build_manifest_rows, write_manifest_bundle
from .data.padding import (
    audit_padding_on_manifest,
    load_padding_config,
    materialize_padded_dataset,
    write_padding_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrrp-p0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-manifest",
        help="Inspect raw MATLAB files, build the P0 manifest, and run blocking checks.",
    )
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--raw-root", type=Path)
    build.add_argument("--output", required=True, type=Path)
    padding = subparsers.add_parser(
        "audit-padding",
        help="Audit deterministic complex-Gaussian power padding without writing padded data.",
    )
    padding.add_argument("--config", required=True, type=Path)
    padding.add_argument("--padding-config", required=True, type=Path)
    padding.add_argument("--raw-root", type=Path)
    padding.add_argument("--output", required=True, type=Path)
    materialize = subparsers.add_parser(
        "materialize-padding",
        help="Write the deterministic 601-bin padded profiles and traceable index.",
    )
    materialize.add_argument("--config", required=True, type=Path)
    materialize.add_argument("--padding-config", required=True, type=Path)
    materialize.add_argument("--raw-root", type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    smoke = subparsers.add_parser(
        "run-b0-smoke",
        help="Run the P0/B0 diagnostic end-to-end smoke pipeline.",
    )
    smoke.add_argument("--config", required=True, type=Path)
    smoke.add_argument("--bundle-root", type=Path)
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument("--device", default="auto")
    main_b0 = subparsers.add_parser(
        "run-b0-main",
        help="Run one preregistered P0/B0 main_v3 seed.",
    )
    main_b0.add_argument("--config", required=True, type=Path)
    main_b0.add_argument("--bundle-root", type=Path)
    main_b0.add_argument("--output", required=True, type=Path)
    main_b0.add_argument("--device", default="auto")
    aggregate_b0 = subparsers.add_parser(
        "aggregate-b0-main",
        help="Aggregate the five preregistered B0 model seeds with 95% confidence intervals.",
    )
    aggregate_b0.add_argument("--run-root", required=True, type=Path)
    aggregate_b0.add_argument("--output", required=True, type=Path)
    b1 = subparsers.add_parser("run-b1", help="Run B1 for all five frozen B0 checkpoints.")
    b1.add_argument("--config", required=True, type=Path)
    b1.add_argument("--bundle-root", required=True, type=Path)
    b1.add_argument("--b0-root", required=True, type=Path)
    b1.add_argument("--output-root", required=True, type=Path)
    b1.add_argument("--device", default="auto")
    set_model = subparsers.add_parser(
        "run-set-model", help="Train and evaluate B2 or B3 for all five seeds."
    )
    set_model.add_argument("--config", required=True, type=Path)
    set_model.add_argument("--bundle-root", required=True, type=Path)
    set_model.add_argument("--output-root", required=True, type=Path)
    set_model.add_argument("--device", default="auto")
    openmax = subparsers.add_parser(
        "run-openmax", help="Fit and evaluate B4 or B5 for all five frozen checkpoints."
    )
    openmax.add_argument("--config", required=True, type=Path)
    openmax.add_argument("--bundle-root", required=True, type=Path)
    openmax.add_argument("--source-root", required=True, type=Path)
    openmax.add_argument("--output-root", required=True, type=Path)
    openmax.add_argument("--device", default="auto")
    b6 = subparsers.add_parser(
        "run-b6-main", help="Run the deterministic B6 JDSR-OSR core adaptation for main V=3."
    )
    b6.add_argument("--config", required=True, type=Path)
    b6.add_argument("--bundle-root", required=True, type=Path)
    b6.add_argument("--output", required=True, type=Path)
    b6.add_argument("--device", default="auto")
    b6_v5 = subparsers.add_parser(
        "run-b6-paper-v5", help="Run the B6 V=5 paper-parameter-aligned auxiliary diagnostic."
    )
    b6_v5.add_argument("--config", required=True, type=Path)
    b6_v5.add_argument("--bundle-root", required=True, type=Path)
    b6_v5.add_argument("--output", required=True, type=Path)
    b6_v5.add_argument("--device", default="auto")
    aggregate_method = subparsers.add_parser(
        "aggregate-method", help="Aggregate five seed runs for B1-B5."
    )
    aggregate_method.add_argument("--run-root", required=True, type=Path)
    aggregate_method.add_argument("--output", required=True, type=Path)
    aggregate_method.add_argument("--baseline", required=True)
    aggregate_method.add_argument("--scope", required=True)
    audit_results = subparsers.add_parser(
        "audit-results", help="Recompute metrics from predictions and verify artifact hashes."
    )
    audit_results.add_argument("--root", required=True, action="append", type=Path)
    audit_results.add_argument("--output", required=True, type=Path)
    return parser


def _resolve_raw_root(raw_root: Path | None, env_name: str) -> Path:
    if raw_root is not None:
        return raw_root
    value = os.environ.get(env_name)
    if not value:
        raise DataConfigError(
            f"Provide --raw-root or set the configured environment variable {env_name}"
        )
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit-results":
            from .evaluation.result_audit import audit_result_tree

            result = audit_result_tree(args.root, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-b6-paper-v5":
            from .training.b6 import run_b6_paper_aligned_v5

            result = run_b6_paper_aligned_v5(
                args.config, args.bundle_root, args.output, device_request=args.device
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-b6-main":
            from .training.b6 import run_b6_main

            result = run_b6_main(
                args.config, args.bundle_root, args.output, device_request=args.device
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-openmax":
            from .training.openmax import run_openmax_all

            result = run_openmax_all(
                args.config, args.bundle_root, args.source_root, args.output_root,
                device_request=args.device,
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-set-model":
            from .training.set_models import run_set_model_all

            result = run_set_model_all(
                args.config, args.bundle_root, args.output_root,
                device_request=args.device,
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-b1":
            from .training.b1 import run_b1_all

            result = run_b1_all(
                args.config, args.bundle_root, args.b0_root, args.output_root,
                device_request=args.device,
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "aggregate-method":
            from .evaluation.method_aggregate import aggregate_method_seed_runs

            result = aggregate_method_seed_runs(
                args.run_root,
                args.output,
                expected_baseline=args.baseline,
                expected_scope=args.scope,
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "aggregate-b0-main":
            from .evaluation.aggregate import aggregate_b0_main_runs

            result = aggregate_b0_main_runs(args.run_root, args.output)
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"run-b0-smoke", "run-b0-main"}:
            from .training.b0_smoke import (
                load_b0_main_config,
                load_b0_smoke_config,
                run_b0_main,
                run_b0_smoke,
            )

            if args.command == "run-b0-smoke":
                b0_config = load_b0_smoke_config(args.config)
                runner = run_b0_smoke
            else:
                b0_config = load_b0_main_config(args.config)
                runner = run_b0_main
            bundle_root = args.bundle_root
            if bundle_root is None:
                env_name = str(b0_config["data"]["bundle_root_env"])
                value = os.environ.get(env_name)
                if not value:
                    raise DataConfigError(
                        f"Provide --bundle-root or set the configured variable {env_name}"
                    )
                bundle_root = Path(value)
            result = runner(
                args.config,
                bundle_root,
                args.output,
                device_request=args.device,
            )
            command_status = str(result.get("validation_status", "passed"))
            print(
                json.dumps(
                    {"status": command_status, **result},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if command_status == "passed" else 3
        config = load_data_config(args.config)
        raw_root = _resolve_raw_root(args.raw_root, config.source.root_env)
        build = build_manifest_rows(config, raw_root)
        if args.command == "build-manifest":
            result = write_manifest_bundle(build, config, args.output)
        elif args.command == "audit-padding":
            padding_config = load_padding_config(args.padding_config)
            report = audit_padding_on_manifest(build, config, padding_config, raw_root)
            result = write_padding_audit(report, padding_config, args.output)
        else:
            padding_config = load_padding_config(args.padding_config)
            result = materialize_padded_dataset(
                build,
                config,
                padding_config,
                raw_root,
                args.output,
            )
    except (DataConfigError, DataValidationError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    command_status = str(result.get("validation_status", "passed"))
    print(json.dumps({"status": command_status, **result}, ensure_ascii=False, indent=2))
    return 0 if command_status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
