#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Helpers for CI benchmark runs, normalization, and regression checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCommand:
    name: str
    argv: list[str]
    required: bool = True


def _python() -> str:
    return sys.executable


def _commands_for_suite(suite: str, raw_dir: Path) -> list[BenchmarkCommand]:
    python = _python()
    commands = [
        BenchmarkCommand(
            "cutlass_fused_moe",
            [
                python,
                "benchmark/benchmark_cutlass_fused_moe.py",
                "--save-path",
                str(raw_dir / "cutlass_fused_moe"),
            ],
        ),
        BenchmarkCommand(
            "flash_attn_decode",
            [
                python,
                "benchmark/benchmark_cutlass_flash_attn_decode.py",
                "--save-path",
                str(raw_dir / "flash_attn_decode"),
            ],
        ),
        BenchmarkCommand(
            "flash_attn_varlen",
            [
                python,
                "benchmark/benchmark_cutlass_flash_attn_varlen.py",
                "--save-path",
                str(raw_dir / "flash_attn_varlen"),
            ],
        ),
    ]

    if suite == "smoke":
        return commands

    if suite != "nightly":
        raise ValueError(f"Unsupported benchmark suite: {suite}")

    commands.extend([
        BenchmarkCommand(
            "grouped_topk",
            [
                python,
                "benchmark/benchmark_grouped_topk.py",
                "--save-path",
                str(raw_dir / "grouped_topk"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "topk_softmax",
            [
                python,
                "benchmark/benchmark_topk.py",
                "--scoring-func",
                "softmax",
                "--save-path",
                str(raw_dir / "topk"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "topk_sigmoid",
            [
                python,
                "benchmark/benchmark_topk.py",
                "--scoring-func",
                "sigmoid",
                "--save-path",
                str(raw_dir / "topk"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "topk_softplus_sqrt",
            [
                python,
                "benchmark/benchmark_topk_softplus_sqrt.py",
                "--save-path",
                str(raw_dir / "topk_softplus_sqrt"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "gemm_onednn",
            [
                python,
                "benchmark/benchmark_gemm_onednn.py",
                "--benchmarks",
                "bf16",
                "fp8_w8a16",
                "--save-path",
                str(raw_dir / "gemm_onednn"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "gdn_attn",
            [
                python,
                "benchmark/benchmark_gdn_attn.py",
                "--save-path",
                str(raw_dir / "gdn_attn"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "causal_conv1d",
            [
                python,
                "benchmark/benchmark_causal_conv1d.py",
                "--save-path",
                str(raw_dir / "causal_conv1d"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "gated_delta_rule",
            [
                python,
                "benchmark/benchmark_gated_delta_rule.py",
                "--save-path",
                str(raw_dir / "gated_delta_rule"),
            ],
            required=False,
        ),
        BenchmarkCommand(
            "lora",
            [
                python,
                "benchmark/benchmark_lora.py",
                "list_bench",
                "--dtype",
                "torch.float16",
                "--arg-pool-size",
                "32",
                "--batch-sizes",
                "1",
                "16",
                "64",
                "--hidden-sizes",
                "2048",
                "4096",
                "--lora-ranks",
                "16",
                "--num-loras",
                "1",
                "4",
                "--op-types",
                "bgmv_shrink",
                "bgmv_expand",
                "bgmv_expand_slice",
                "--seq-lengths",
                "1",
                "--sort-by-lora-id",
                "1",
                "-o",
                str(raw_dir / "lora"),
            ],
            required=False,
        ),
    ])
    return commands


def _prepare_command_outputs(command: BenchmarkCommand) -> None:
    output_flags = {"--save-path", "--output-directory", "-o"}
    for index, value in enumerate(command.argv[:-1]):
        if value in output_flags:
            Path(command.argv[index + 1]).mkdir(parents=True, exist_ok=True)


def run_suite(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    commands = _commands_for_suite(args.suite, raw_dir)
    manifest = {
        "suite": args.suite,
        "started_at": int(time.time()),
        "commands": [
            {
                "name": command.name,
                "argv": command.argv,
                "required": command.required,
            }
            for command in commands
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    env = os.environ.copy()
    env.setdefault("ZE_AFFINITY_MASK", "0")

    failures: list[dict[str, Any]] = []
    for command in commands:
        log_file = log_dir / f"{command.name}.log"
        print(f"::group::benchmark {command.name}", flush=True)
        print(" ".join(command.argv), flush=True)
        _prepare_command_outputs(command)
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        print("::endgroup::", flush=True)
        if return_code != 0:
            failures.append({
                "name": command.name,
                "required": command.required,
                "return_code": return_code,
                "log": str(log_file),
            })
            print(
                f"Benchmark {command.name} failed with {return_code}",
                file=sys.stderr,
            )
            if command.required:
                (output_dir / "run_failures.json").write_text(
                    json.dumps(failures, indent=2) + "\n")
                return return_code
            print(
                f"Continuing because {command.name} is low priority",
                file=sys.stderr,
            )

    (output_dir / "run_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n")

    return 0


def _try_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _looks_like_metric(name: str) -> bool:
    lowered = name.lower()
    patterns = (
        "latency",
        "time",
        "_us",
        "(us)",
        "us",
        "_ms",
        "(ms)",
        "tflops",
        "bandwidth",
        "gbs",
        "gb/s",
        "mbu",
        "vllm",
        "native",
        "compile",
        "flash",
        "gdn",
        "torch.mm",
    )
    return any(pattern in lowered for pattern in patterns)


def _unit_for_metric(name: str) -> str:
    lowered = name.lower()
    if "tflops" in lowered:
        return "tflops"
    if "bandwidth" in lowered or "gb/s" in lowered or "gbs" in lowered:
        return "gb/s"
    if "mbu" in lowered or "%" in lowered:
        return "percent"
    if "ms" in lowered and "gems" not in lowered:
        return "ms"
    return "us"


def _higher_is_better(metric: str, unit: str) -> bool:
    lowered = metric.lower()
    return unit in {"tflops", "gb/s", "percent"} or any(
        token in lowered for token in ("tflops", "bandwidth", "mbu"))


def _case_id(dimensions: dict[str, Any]) -> str:
    payload = json.dumps(dimensions, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["benchmark"]),
        str(record["metric"]),
        str(record["case_id"]),
        str(record.get("unit", "")),
    )


def _metadata() -> dict[str, Any]:
    keys = [
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_REF_NAME",
        "GITHUB_EVENT_NAME",
        "GITHUB_ACTOR",
        "RUNNER_NAME",
        "RUNNER_ARCH",
        "RUNNER_OS",
        "ZE_AFFINITY_MASK",
    ]
    return {key.lower(): os.environ.get(key, "") for key in keys}


def _records_from_csv(csv_file: Path, input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    benchmark = str(csv_file.relative_to(input_dir).with_suffix(""))
    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics = {
                key: _try_float(value)
                for key, value in row.items()
                if _looks_like_metric(key) and _try_float(value) is not None
            }
            if not metrics:
                continue
            dimensions = {
                key: value
                for key, value in row.items()
                if key not in metrics and value not in (None, "")
            }
            for metric, value in metrics.items():
                unit = _unit_for_metric(metric)
                records.append({
                    "benchmark": benchmark,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "higher_is_better": _higher_is_better(metric, unit),
                    "dimensions": dimensions,
                    "case_id": _case_id(dimensions),
                    "source": str(csv_file.relative_to(input_dir)),
                })
    return records


def _records_from_lora_pickle(
    pickle_file: Path,
    input_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with pickle_file.open("rb") as handle:
            measurements = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping {pickle_file}: {exc}", file=sys.stderr)
        return records

    benchmark = str(pickle_file.relative_to(input_dir).with_suffix(""))
    for measurement in measurements:
        dimensions = {
            "label": getattr(measurement, "label", ""),
            "sub_label": getattr(measurement, "sub_label", ""),
            "description": getattr(measurement, "description", ""),
        }
        value = float(measurement.median) * 1_000_000.0
        records.append({
            "benchmark": benchmark,
            "metric": "median_latency_us",
            "value": value,
            "unit": "us",
            "higher_is_better": False,
            "dimensions": dimensions,
            "case_id": _case_id(dimensions),
            "source": str(pickle_file.relative_to(input_dir)),
        })
    return records


def normalize(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for csv_file in sorted(input_dir.rglob("*.csv")):
        records.extend(_records_from_csv(csv_file, input_dir))
    for pickle_file in sorted(input_dir.rglob("*.pkl")):
        records.extend(_records_from_lora_pickle(pickle_file, input_dir))

    metadata = _metadata()
    for record in records:
        record.update(metadata)

    jsonl_file = output_dir / "results.jsonl"
    with jsonl_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    csv_file = output_dir / "results.csv"
    fieldnames = [
        "benchmark",
        "metric",
        "value",
        "unit",
        "higher_is_better",
        "case_id",
        "dimensions",
        "source",
        "github_sha",
        "github_ref_name",
        "github_run_id",
        "github_event_name",
        "runner_name",
        "ze_affinity_mask",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["dimensions"] = json.dumps(
                record.get("dimensions", {}), sort_keys=True)
            writer.writerow(row)

    (output_dir / "run_info.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Normalized {len(records)} benchmark metrics")
    return 0 if records else 2


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compare(args: argparse.Namespace) -> int:
    current = _load_jsonl(Path(args.current))
    baseline = _load_jsonl(Path(args.baseline))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_by_key = {_record_key(record): record for record in baseline}
    regressions = []
    improvements = []
    compared = 0

    for record in current:
        base = baseline_by_key.get(_record_key(record))
        if base is None:
            continue
        base_value = float(base["value"])
        current_value = float(record["value"])
        if base_value <= 0:
            continue
        compared += 1
        higher_is_better = bool(record.get("higher_is_better"))
        if higher_is_better:
            change = (base_value - current_value) / base_value
            improvement = (current_value - base_value) / base_value
        else:
            change = (current_value - base_value) / base_value
            improvement = (base_value - current_value) / base_value

        item = {
            "benchmark": record["benchmark"],
            "metric": record["metric"],
            "case_id": record["case_id"],
            "unit": record.get("unit", ""),
            "baseline": base_value,
            "current": current_value,
            "change": change,
            "dimensions": record.get("dimensions", {}),
        }
        if change > args.threshold:
            regressions.append(item)
        elif improvement > args.threshold:
            improvements.append(item)

    report = _format_report(compared, regressions, improvements,
                            args.threshold, bool(baseline))
    (output_dir / "regression_report.md").write_text(report)
    (output_dir / "regressions.json").write_text(
        json.dumps(regressions, indent=2, sort_keys=True) + "\n")
    print(report)

    if regressions and args.fail_on_regression:
        return 1
    return 0


def _format_report(
    compared: int,
    regressions: list[dict[str, Any]],
    improvements: list[dict[str, Any]],
    threshold: float,
    has_baseline: bool,
) -> str:
    lines = ["## Benchmark Regression Report", ""]
    if not has_baseline:
        lines.extend([
            "No baseline was found. This run will seed future comparisons.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        f"Compared metrics: **{compared}**",
        f"Regression threshold: **{threshold:.1%}**",
        f"Regressions: **{len(regressions)}**",
        f"Improvements: **{len(improvements)}**",
        "",
    ])

    if regressions:
        lines.extend([
            "### Regressions",
            "",
            "| Benchmark | Metric | Baseline | Current | Change | Case |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ])
        for item in regressions[:50]:
            lines.append(
                "| {benchmark} | {metric} | {baseline:.4g} | "
                "{current:.4g} | {change:.1%} | `{case_id}` |".format(
                    **item))
        lines.append("")
    else:
        lines.extend(["No benchmark regressions exceeded the threshold.", ""])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--suite", choices=["smoke", "nightly"],
                            default="nightly")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.set_defaults(func=run_suite)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--input-dir", required=True)
    normalize_parser.add_argument("--output-dir", required=True)
    normalize_parser.set_defaults(func=normalize)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--current", required=True)
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--threshold", type=float, default=0.10)
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
