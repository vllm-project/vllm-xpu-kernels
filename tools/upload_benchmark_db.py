#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upload nightly microbenchmark CSV results into Postgres.

Self-contained port of the CSV->params/metrics logic from
vllm_test_framework/core/db_updater.py::insert_kernel_data. Reads the raw wide
CSVs produced by the benchmark suite and inserts one row per shape into
``vllm_nightly_microbenchmark``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote_plus

logger = logging.getLogger("upload_benchmark_db")


def _strip_ylabel(col_name: str) -> str:
    """Remove trailing ' (ylabel)' appended by Triton >= 3.7.0 CSV export."""
    return re.sub(r"\s+\((?:[^()]*|\([^()]*\))*\)\s*$", "", col_name)


def _is_metric_col(col_name: str) -> bool:
    """Metric cols carry parenthesized units or a *flops keyword."""
    lower = col_name.lower()
    if re.search(r"\([^)]+\)", lower):
        return True
    return "tflops" in lower or "gflops" in lower


def _clean_metric(value: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _kernel_name(csv_file: Path, results_dir: Path) -> str:
    """Posix relpath under raw dir sans suffix (matches ``benchmark`` key)."""
    return csv_file.relative_to(results_dir).with_suffix("").as_posix()


def _rows_from_csv(csv_file: Path, results_dir: Path) -> list[dict]:
    kernel_name = _kernel_name(csv_file, results_dir)
    rows: list[dict] = []
    with csv_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            params: dict = {}
            metrics: dict = {}
            for key, value in row.items():
                if key is None:
                    continue
                clean_key = _strip_ylabel(key)
                if _is_metric_col(key):
                    metrics[clean_key] = _clean_metric(value)
                else:
                    params[clean_key] = value if value != "" else None
            params_hash = hashlib.md5(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()
            rows.append({
                "kernel_name": kernel_name,
                "params": params,
                "metrics": metrics,
                "params_hash": params_hash,
            })
    return rows


def collect_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for csv_file in sorted(results_dir.rglob("*.csv")):
        rows.extend(_rows_from_csv(csv_file, results_dir))
    return rows


def _db_url(args: argparse.Namespace) -> str:
    password = quote_plus(args.db_password)
    return (
        f"postgresql://{args.db_user}:{password}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )


def insert_rows(args: argparse.Namespace, rows: list[dict]) -> int:
    import psycopg  # pip install "psycopg[binary]"

    sql = (
        f"INSERT INTO {args.table} "
        "(kernel_name, docker_tag, params, metrics, node_label, "
        "params_hash, result_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    inserted = 0
    failed = 0
    with psycopg.connect(_db_url(args)) as conn, conn.cursor() as cur:
        for row in rows:
            try:
                cur.execute(sql, (
                    row["kernel_name"],
                    args.docker_tag,
                    json.dumps(row["params"]),
                    json.dumps(row["metrics"]),
                    args.node_label,
                    row["params_hash"],
                    args.result_id,
                ))
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "Row for kernel %s failed: %s", row["kernel_name"], exc
                )
        conn.commit()
    if failed:
        logger.warning("%d/%d rows failed to insert", failed, len(rows))
    return inserted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="benchmark-results/raw")
    parser.add_argument("--docker-tag", required=True)
    parser.add_argument(
        "--node-label", default=os.environ.get("DB_NODE_LABEL", "bmg")
    )
    parser.add_argument("--result-id", required=True)
    parser.add_argument(
        "--table",
        default=os.environ.get("DB_TABLE", "vllm_nightly_microbenchmark"),
    )
    parser.add_argument(
        "--db-host", default=os.environ.get("DB_HOST", "localhost")
    )
    parser.add_argument(
        "--db-port", default=os.environ.get("DB_PORT", "5432")
    )
    parser.add_argument(
        "--db-name", default=os.environ.get("DB_NAME", "vllm_benchmarks")
    )
    parser.add_argument(
        "--db-user", default=os.environ.get("DB_USER", "vllmadmin")
    )
    parser.add_argument(
        "--db-password", default=os.environ.get("DB_PASSWORD", "")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.warning(
            "Results dir %s not found; skipping DB upload.", results_dir
        )
        return 0

    rows = collect_rows(results_dir)
    if not rows:
        logger.warning(
            "No CSV rows found under %s; skipping DB upload.", results_dir
        )
        return 0

    logger.info(
        "Collected %d rows for docker_tag=%s result_id=%s",
        len(rows), args.docker_tag, args.result_id,
    )

    if args.dry_run:
        for row in rows[:5]:
            logger.info("DRY-RUN %s", json.dumps(row, sort_keys=True))
        logger.info("DRY-RUN total rows: %d", len(rows))
        return 0

    if not args.db_password:
        logger.warning("DB_PASSWORD is empty; skipping DB upload.")
        return 0

    try:
        inserted = insert_rows(args, rows)
        logger.info("Inserted %d rows into %s", inserted, args.table)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB upload failed (non-fatal): %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
