# SPDX-License-Identifier: Apache-2.0
"""Bridge PR ``/microbench`` comments to the internal Jenkins kernel
microbenchmark job and report the result back to the PR.

Standard library only. Inputs come from environment variables set by
``.github/workflows/pr-microbenchmark.yaml``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import ssl
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NoReturn

COMMAND = "/microbench"
MARKER = "<!-- vllm-xpu-microbench -->"
STATUS_CONTEXT = "intel-xpu/kernel-microbenchmark"
ALLOWED_OPS = (
    "flash_attn_decode",
    "flash_attn_varlen",
    "fused_moe",
    "gemm_onednn",
    "gdn_attn",
)
ALLOWED_ROLES = ("admin", "maintain", "write")
DEFAULT_ENV_FILE = "/etc/vllm-xpu-microbench/jenkins.env"
ARTIFACT_DIR = "artifact/kernel/logs/MICROBENCHMARK"
SAFE_CHARS = set(string.ascii_letters + string.digits + "_.-")
SAFE_METRIC_CHARS = SAFE_CHARS | set("()/% ")
HEX = set(string.hexdigits.lower())
USAGE = (
    "Usage: `/microbench [ops=<op>[,<op>...]]`\n\n"
    f"Available ops: {', '.join(f'`{op}`' for op in ALLOWED_OPS)}. "
    "Without `ops=` all kernels are benchmarked."
)
POLL_SECONDS = 60
MAX_CONSECUTIVE_ERRORS = 10


# ---------------------------------------------------------------- helpers
def fail(msg: str) -> NoReturn:
    print(f"::error::{msg}")
    sys.exit(1)


def env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if required and not value:
        fail(f"missing environment variable {name}")
    return value


def set_output(name: str, value: str) -> None:
    if "\n" in value or "\r" in value:
        fail(f"refusing multi-line output for {name}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def is_sha(value: str) -> bool:
    return len(value) == 40 and set(value) <= HEX


def safe_name(value: Any, allowed: set[str] = SAFE_CHARS) -> str:
    text = str(value) if value is not None else ""
    if text and len(text) <= 100 and set(text) <= allowed:
        return text
    return "?"


def http(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    context: ssl.SSLContext | None = None,
) -> tuple[int, Any, bytes]:
    req = urllib.request.Request(
        url, data=data, method=method, headers=headers or {}
    )
    with urllib.request.urlopen(req, timeout=60, context=context) as resp:
        return resp.status, resp.headers, resp.read()


# ----------------------------------------------------------------- github
def gh(method: str, path: str, body: Any = None) -> Any:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    headers = {
        "Authorization": f"Bearer {env('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    _, _, raw = http(f"{api}{path}", method, headers, data)
    return json.loads(raw) if raw else None


def react(repo: str, comment_id: int, content: str) -> None:
    try:
        gh(
            "POST",
            f"/repos/{repo}/issues/comments/{comment_id}/reactions",
            {"content": content},
        )
    except urllib.error.URLError as e:
        print(f"::warning::failed to add reaction: {e}")


def parse_ops(tokens: list[str]) -> tuple[str, str | None]:
    ops: list[str] = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if key != "ops" or not sep or not value:
            return "", f"Unknown argument `{safe_name(token)}`."
        for op in value.split(","):
            if op not in ALLOWED_OPS:
                return "", f"Unknown op `{safe_name(op)}`."
            if op not in ops:
                ops.append(op)
    return ",".join(ops), None


def cmd_authorize() -> None:
    repo = env("GITHUB_REPOSITORY")
    with open(env("GITHUB_EVENT_PATH")) as f:
        event = json.load(f)
    comment = event["comment"]
    issue = event["issue"]
    set_output("authorized", "false")

    lines = (comment.get("body") or "").strip().splitlines()
    tokens = lines[0].split() if lines else []
    if "pull_request" not in issue or not tokens or tokens[0] != COMMAND:
        return
    if comment["user"].get("type") != "User":
        return

    user = comment["user"]["login"]
    comment_id = int(comment["id"])
    pr_number = int(issue["number"])

    perm = gh(
        "GET",
        f"/repos/{repo}/collaborators/"
        f"{urllib.parse.quote(user, safe='')}/permission",
    )
    role = perm.get("role_name") or perm.get("permission")
    if role not in ALLOWED_ROLES:
        # No reply comment, so outsiders cannot make the bot post on demand.
        print(f"{user} has role {role!r}; not allowed to trigger")
        react(repo, comment_id, "-1")
        return

    def reject(reason: str) -> None:
        react(repo, comment_id, "confused")
        gh(
            "POST",
            f"/repos/{repo}/issues/{pr_number}/comments",
            {"body": f"{MARKER}\n@{user} {reason}\n\n{USAGE}"},
        )

    ops, err = parse_ops(tokens[1:])
    if err:
        reject(err)
        return

    pr = gh("GET", f"/repos/{repo}/pulls/{pr_number}")
    if pr.get("state") != "open":
        reject("Microbenchmarks can only be triggered on open PRs.")
        return
    sha = pr["head"]["sha"]
    if not is_sha(sha):
        fail("unexpected head sha format")

    react(repo, comment_id, "eyes")
    set_output("pr", str(pr_number))
    set_output("sha", sha)
    set_output("ops", ops)
    set_output("user", safe_name(user))
    set_output("authorized", "true")


# ---------------------------------------------------------------- jenkins
def load_env_file(required: bool) -> dict[str, str]:
    path = os.environ.get("MICROBENCH_JENKINS_ENV", DEFAULT_ENV_FILE)
    cfg: dict[str, str] = {}
    if not os.path.exists(path):
        if required:
            fail(f"Jenkins config {path} not found on this runner")
        return cfg
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip().strip("\"'")
    token = cfg.get("JENKINS_API_TOKEN", "")
    # Outside Actions the mask command is plain stdout and would leak it.
    if token and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::add-mask::{token}")
    return cfg


class Jenkins:

    def __init__(self) -> None:
        cfg = load_env_file(required=True)
        for key in ("JENKINS_URL", "JENKINS_USER", "JENKINS_API_TOKEN",
                    "JENKINS_JOB"):
            if not cfg.get(key):
                fail(f"{key} is not set in the Jenkins config")
        self.base = cfg["JENKINS_URL"].rstrip("/") + "/"
        self.job = cfg["JENKINS_JOB"]
        cred = f"{cfg['JENKINS_USER']}:{cfg['JENKINS_API_TOKEN']}"
        self.auth = "Basic " + base64.b64encode(cred.encode()).decode()
        cafile = cfg.get("JENKINS_CA_BUNDLE") or None
        self.ctx = ssl.create_default_context(cafile=cafile)

    def check_url(self, url: str) -> str:
        base = urllib.parse.urlsplit(self.base)
        target = urllib.parse.urlsplit(url)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc):
            fail("Jenkins returned a URL on an unexpected host")
        return url if url.endswith("/") else url + "/"

    def request(self, url: str, method: str = "GET",
                data: bytes | None = None) -> tuple[int, Any, bytes]:
        headers = {"Authorization": self.auth}
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return http(url, method, headers, data, self.ctx)

    def get_json(self, url: str) -> Any:
        return json.loads(self.request(url)[2])


def poll(fn: Any, timeout_minutes: int) -> Any:
    """Call fn() until it returns non-None, tolerating transient errors."""
    deadline = time.monotonic() + timeout_minutes * 60
    errors = 0
    while time.monotonic() < deadline:
        try:
            result = fn()
            errors = 0
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            errors += 1
            print(f"::warning::Jenkins poll failed ({errors}): {e}")
            if errors >= MAX_CONSECUTIVE_ERRORS:
                fail("Jenkins unreachable")
            result = None
        if result is not None:
            return result
        time.sleep(POLL_SECONDS)
    fail(f"timed out after {timeout_minutes} minutes")


def cmd_trigger() -> None:
    jenkins = Jenkins()
    sha = env("SHA")
    pr = env("PR")
    if not is_sha(sha) or not pr.isdigit():
        fail("invalid PR/SHA")
    params = {
        "PR_NUMBER": pr,
        "PR_HEAD_SHA": sha,
        "KERNEL_OPS": env("OPS", required=False),
        "GH_RUN_ID": env("GITHUB_RUN_ID"),
    }
    # Manual re-runs only: test the existing PR image instead of rebuilding.
    if env("REUSE_IMAGE", required=False) == "True":
        params["REUSE_IMAGE"] = "True"
    url = (f"{jenkins.base}job/{urllib.parse.quote(jenkins.job, safe='')}"
           "/buildWithParameters")
    data = urllib.parse.urlencode(params).encode()
    _, headers, _ = jenkins.request(url, "POST", data)
    location = headers.get("Location")
    if not location:
        fail("Jenkins did not return a queue location")
    queue_url = jenkins.check_url(location)
    print(f"Queued: {queue_url}")
    set_output("queue_url", queue_url)


def cmd_wait_start() -> None:
    jenkins = Jenkins()
    queue_url = jenkins.check_url(env("QUEUE_URL"))

    def started() -> str | None:
        item = jenkins.get_json(queue_url + "api/json")
        if item.get("cancelled"):
            fail("Jenkins queue item was cancelled")
        executable = item.get("executable") or {}
        if executable.get("url"):
            return jenkins.check_url(executable["url"])
        print(f"Waiting in Jenkins queue: {item.get('why')}")
        return None

    build_url = poll(started, int(env("QUEUE_TIMEOUT_MINUTES",
                                      required=False) or 240))
    print(f"Build started: {build_url}")
    set_output("build_url", build_url)


def cmd_wait() -> None:
    jenkins = Jenkins()
    build_url = jenkins.check_url(env("BUILD_URL"))

    def finished() -> str | None:
        info = jenkins.get_json(
            build_url + "api/json?tree=building,result")
        if info.get("building") or not info.get("result"):
            return None
        return str(info["result"])

    result = poll(finished, int(env("BUILD_TIMEOUT_MINUTES",
                                    required=False) or 420))
    print(f"Build finished: {result}")
    set_output("result", safe_name(result))

    try:
        _, _, raw = jenkins.request(
            f"{build_url}{ARTIFACT_DIR}/pr_summary.json")
        summary = json.loads(raw)
        with open(env("SUMMARY_FILE"), "w") as f:
            json.dump(summary, f)
    except (urllib.error.URLError, ValueError) as e:
        print(f"::warning::pr_summary.json not available: {e}")


def cmd_stop() -> None:
    jenkins = Jenkins()
    build_url = env("BUILD_URL", required=False)
    queue_url = env("QUEUE_URL", required=False)

    def started_build() -> str:
        item = jenkins.get_json(jenkins.check_url(queue_url) + "api/json")
        url = (item.get("executable") or {}).get("url")
        return jenkins.check_url(url) if url else ""

    try:
        if not build_url and queue_url:
            build_url = started_build()
            if not build_url:
                item_id = jenkins.check_url(queue_url).rstrip("/")
                item_id = item_id.split("/")[-1]
                if item_id.isdigit():
                    jenkins.request(
                        f"{jenkins.base}queue/cancelItem?id={item_id}",
                        "POST", b"")
                    print("Cancelled Jenkins queue item")
                # The item may have left the queue while we were cancelling.
                time.sleep(5)
                build_url = started_build()
        if build_url:
            jenkins.request(jenkins.check_url(build_url) + "stop", "POST",
                            b"")
            print("Stopped Jenkins build")
    except (urllib.error.URLError, ValueError) as e:
        print(f"::warning::failed to stop Jenkins job: {e}")


# ----------------------------------------------------------------- report
def fmt_float(value: Any, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def load_summary() -> dict[str, Any] | None:
    path = env("SUMMARY_FILE", required=False)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def final_state(result: str, job_status: str) -> str:
    # Correctness failures are reported in the summary but do not fail the run.
    if job_status == "cancelled" or result == "ABORTED":
        return "cancelled"
    if result == "SUCCESS":
        return "success"
    if result:
        return "failure"
    return "error"


STATE_TEXT = {
    "queued": "queued",
    "running": "running",
    "success": "passed",
    "failure": "failed",
    "cancelled": "cancelled",
    "error": "error",
}
COMMIT_STATE = {
    "queued": "pending",
    "running": "pending",
    "success": "success",
    "failure": "failure",
    "cancelled": "error",
    "error": "error",
}


def render(state: str, build_url: str, summary: dict[str, Any] | None,
           cfg: dict[str, str]) -> str:
    sha = env("SHA")
    ops = env("OPS", required=False)
    repo = env("GITHUB_REPOSITORY")
    run_url = (f"{env('GITHUB_SERVER_URL')}/{repo}/actions/runs/"
               f"{env('GITHUB_RUN_ID')}")
    lines = [
        MARKER,
        f"### XPU kernel microbenchmark: **{STATE_TEXT[state]}**",
        "",
        f"- Commit: `{sha[:7]}`",
        f"- Requested by: @{safe_name(env('REQUESTER'))}",
        "- Kernels: " + (", ".join(f"`{o}`" for o in ops.split(","))
                         if ops else "all"),
        f"- Workflow run: [link]({run_url})",
    ]
    if build_url:
        lines.append(f"- Jenkins build: [link]({build_url})")
    if state in ("queued", "running"):
        lines += ["", "Results will be posted here when the run finishes."]
        return "\n".join(lines)

    if not summary:
        lines += ["", "No benchmark summary was produced; "
                  "see the Jenkins build for details."]
        return "\n".join(lines)

    cur_tag = safe_name(summary.get("cur_docker_tag"))
    ref_tag = safe_name(summary.get("ref_docker_tag"))
    node = safe_name(summary.get("node_label"))
    grafana = cfg.get("GRAFANA_DASHBOARD_URL", "")

    def dashboard(kernel: str, metric: str) -> str:
        if not grafana or "?" in (cur_tag, kernel, metric):
            return "-"
        query = {
            "orgId": "1",
            "from": "now-30m",
            "to": "now",
            "timezone": "browser",
            "var-node_label": node if node != "?" else "",
            "var-kernel_name": kernel,
            "var-metric_key": metric,
            "var-cur_docker_tag": cur_tag,
            "var-ref_docker_tag": ref_tag if ref_tag != "?" else "",
            "var-shape_group": "$__all",
            "var-seq_len": "$__all",
        }
        qs = urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        return f"[Grafana]({grafana}?{qs})"

    threshold = fmt_float(
        100 * float(summary.get("regression_threshold") or 0.05), 0)
    lines += [
        "",
        f"Reference: `{ref_tag}` | Current: `{cur_tag}` | "
        f"Correctness failures: "
        f"**{fmt_int(summary.get('correctness_failures'))}**",
        "",
    ]
    kernels = summary.get("kernels") or []
    if not kernels:
        lines.append("No reference data found for comparison.")
        return "\n".join(lines)
    lines += [
        "| Kernel | Metric | Shapes compared | Geomean speedup "
        f"| Regressions (>{threshold}%) | Improvements (>{threshold}%) "
        "| Dashboard |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for k in kernels[:50]:
        if not isinstance(k, dict):
            continue
        name = safe_name(k.get("name"))
        entries = k.get("metrics")
        if not isinstance(entries, list) or not entries:
            entries = [{}]
        for m in entries[:12]:
            if not isinstance(m, dict):
                continue
            metric = safe_name(m.get("metric"), SAFE_METRIC_CHARS)
            speedup = fmt_float(m.get("geomean_speedup"), 3)
            lines.append(
                f"| {name} | {'`' + metric + '`' if metric != '?' else '-'} "
                f"| {fmt_int(m.get('compared'))} "
                f"| {speedup + 'x' if speedup != '-' else '-'} "
                f"| {fmt_int(m.get('regressions'))} "
                f"| {fmt_int(m.get('improvements'))} "
                f"| {dashboard(name, metric)} |")
    lines += [
        "",
        "Speedup > 1 means this PR is faster than the reference "
        "(lower latency, higher TFLOPS/bandwidth). Correctness failures are "
        "informational and do not fail this check.",
    ]
    return "\n".join(lines)


def cmd_report(state: str) -> None:
    repo = env("GITHUB_REPOSITORY")
    pr = env("PR")
    sha = env("SHA")
    if not pr.isdigit() or not is_sha(sha):
        fail("invalid PR/SHA")
    cfg = load_env_file(required=False)
    build_url = env("BUILD_URL", required=False)
    if build_url and cfg.get("JENKINS_URL"):
        base = urllib.parse.urlsplit(cfg["JENKINS_URL"])
        target = urllib.parse.urlsplit(build_url)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc):
            build_url = ""
    summary = None
    if state == "final":
        summary = load_summary()
        state = final_state(env("JENKINS_RESULT", required=False),
                            env("JOB_STATUS", required=False))

    body = render(state, build_url, summary, cfg)
    comment_id = env("COMMENT_ID", required=False)
    if comment_id.isdigit():
        gh("PATCH", f"/repos/{repo}/issues/comments/{comment_id}",
           {"body": body})
    else:
        created = gh("POST", f"/repos/{repo}/issues/{pr}/comments",
                     {"body": body})
        set_output("comment_id", str(created["id"]))

    run_url = (f"{env('GITHUB_SERVER_URL')}/{repo}/actions/runs/"
               f"{env('GITHUB_RUN_ID')}")
    gh("POST", f"/repos/{repo}/statuses/{sha}", {
        "state": COMMIT_STATE[state],
        "context": STATUS_CONTEXT,
        "description": f"Kernel microbenchmark {STATE_TEXT[state]}",
        "target_url": build_url or run_url,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "authorize", "trigger", "wait-start", "wait", "stop", "report"
    ])
    parser.add_argument("--state", default="final",
                        choices=["queued", "running", "final"])
    args = parser.parse_args()
    if args.command == "authorize":
        cmd_authorize()
    elif args.command == "trigger":
        cmd_trigger()
    elif args.command == "wait-start":
        cmd_wait_start()
    elif args.command == "wait":
        cmd_wait()
    elif args.command == "stop":
        cmd_stop()
    else:
        cmd_report(args.state)


if __name__ == "__main__":
    main()
