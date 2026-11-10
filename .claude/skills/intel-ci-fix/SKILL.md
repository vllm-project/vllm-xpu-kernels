---
name: intel-ci-fix
description: Fetches failed vLLM Intel CI build logs from Buildkite, summarizes failed jobs and uses job history to classify them as deterministic / flaky / hang / hardware / infra, pulls the matching image to reproduce on a local XPU machine, attempts to locate and fix the code, and produces a structured fix report.
---

# Intel CI Failure Diagnosis and Auto-Fix Guide (SOP)

When the user provides a Buildkite build link (e.g. `https://buildkite.com/vllm/intel-ci/builds/<id>`) and asks for analysis or a fix, follow the 5 phases below strictly. Constraints given by the user (e.g. "only the first four cards are available") take precedence over the defaults in this document.

## Phase 1: Fetch, Summarize & Triage

Do not read full raw logs into context; only look at the scanner summaries below, and run bounded `grep`s on the saved logs when you need details.

1. **Generate the scanner** (once per session). It has three modes:
   - Scan mode: for each build, fetch build metadata and the job list, filter jobs by a name regex, download and clean the logs (strip ANSI, `\r`, and `\x1b_bk;t=...\x07` timestamps), and print one JSON line per job (including every retry within the same build).
   - `table`: print the JSON lines from stdin as a compact table. By default it keeps only jobs that have a non-PASS attempt; `table all` keeps everything.
   - `local <LOG>`: summarize a local log (e.g. a repro log) with the same rules.

```bash
mkdir -p /tmp/intel-ci-fix && cat > /tmp/intel-ci-fix/bk_scan.py <<'EOF'
# usage: bk_scan.py ORG PIPELINE BUILD JOB_RE BACK FWD {FAIL|ALL} OUT_DIR
#        bk_scan.py local LOG_FILE   (summarize a local log, e.g. a repro run)
#        bk_scan.py table [all]      (compact view of JSON rows from stdin;
#                                     default keeps only jobs with a non-PASS attempt)
import http.cookiejar, json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

site = "https://buildkite.com"
NOISE = re.compile(r"\x1b_bk;t=\d+\x07|\x1b\[[0-9;?]*[A-Za-z]|\r")
FIELDS = {
    "image": r"Using image: (\S+)",
    "mask": r"ZE_AFFINITY_MASK is ([\d,]+)",
    "gpu_gib": r"(?:gpu memory used/total \(GiB\): \d+=[\d.]+/|Free memory on device \([\d.]+/)([\d.]+)",
    "final_commands": r"Final commands: (.*)",
    "bk_error": r"🚨 Error: (.*)",
}
INFRA = re.compile(
    r"i/o timeout|Error response from daemon[^\n]{0,80}|No such container|"
    r"Received cancellation signal|Could not resolve host|Connection reset|"
    r"device lost|UR_RESULT_ERROR_DEVICE_LOST|No space left on device|"
    r"on startup is less than desired GPU memory utilization|HTTP Error 5\d\d|"
    r"Cannot access gated repo|LocalEntryNotFoundError")
LOG_TS = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")


def summarize(text, row):
    for key, pattern in FIELDS.items():
        m = re.search(pattern, text)
        row[key] = m.group(1) if m else None
    row["shard_items"] = re.findall(r"Running (\d+) items in this shard", text)
    row["pytest_summary"] = re.findall(r"^=+ (.*?(?:passed|failed|error).*?) =+$", text, re.M)
    row["failed_tests"] = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+::\S+)", text, re.M)))
    errors = re.findall(r"^(?:E +|\(.*?\) ERROR .*?\] )(\w*(?:Error|Exception)\b.*)$", text, re.M)
    row["errors"] = list(dict.fromkeys(e[:300] for e in errors))[:5]
    row["infra"] = list(dict.fromkeys(m[:120] for m in INFRA.findall(text)))[:5]
    ts = [l for l in text.splitlines() if LOG_TS.search(l)]
    row["last_log_ts"] = LOG_TS.search(ts[-1]).group(1) if ts else None
    row["last_log_line"] = ts[-1][:200] if ts else None
    row["tail"] = [l[:200] for l in text.rstrip().splitlines()[-12:]]
    return row


def minutes(a, b):
    try:
        f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return round((f(b) - f(a)).total_seconds() / 60, 1)
    except Exception:
        return None


def status_of(j):
    if j.get("passed"):
        return "PASS"
    if j.get("soft_failed"):
        return "SOFT"
    return {"timed_out": "TIMEOUT", "canceled": "CANCELED",
            "expired": "EXPIRED"}.get(j.get("state"), "FAIL")


def scan(b):
    op = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    # Without Accept the log download endpoint returns 406.
    op.addheaders = [("User-Agent", "intel-ci-fix"), ("Accept", "*/*")]
    url = f"{site}/{org}/{pipe}/builds/{b}"
    try:
        op.open(url, timeout=30).read()
        info = json.load(op.open(f"{url}.json", timeout=30))
        jobs = json.load(op.open(f"{url}/data/jobs", timeout=30))["records"]
    except Exception:
        return []
    rows = []
    for j in jobs:
        name = j.get("name") or ""
        if j.get("type") != "script" or not j.get("finished_at") or not job_re.search(name):
            continue
        status = status_of(j)
        row = {"build": b, "branch": info.get("branch_name"),
               "commit": (info.get("commit_id") or "")[:12], "status": status,
               "state": j.get("state"), "exit_status": j.get("exit_status"),
               "attempt": (j.get("retries_count") or 0) + 1,
               "retried_in": (j.get("retried_in_job_uuid") or "")[:8] or None,
               "minutes": minutes(j.get("started_at"), j.get("finished_at")),
               "agent": j.get("agent_name"), "job": name, "id": j["id"]}
        rows.append(row)
        if scope != "ALL" and status == "PASS":
            continue
        dl = f"{site}/organizations/{org}/pipelines/{pipe}/builds/{b}/jobs/{j['id']}/download"
        try:
            text = NOISE.sub("", op.open(dl, timeout=120).read().decode("utf-8", "replace"))
        except Exception as e:
            row["log"] = f"download failed: {e}"
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]
        path = out / f"{b}-{slug}-{j['id'][:8]}.log"
        path.write_text(text)
        row["log"] = str(path)
        summarize(text, row)
    return rows


if sys.argv[1] == "table":
    rows = sorted(map(json.loads, sys.stdin), key=lambda r: (r["build"], r["job"], r["attempt"]))
    bad = {(r["build"], r["job"]) for r in rows if r["status"] != "PASS"}
    for r in rows:
        if sys.argv[2:] != ["all"] and (r["build"], r["job"]) not in bad:
            continue
        why = ""
        if r["status"] in ("TIMEOUT", "CANCELED"):
            why = "stuck after: " + (r.get("last_log_line") or "")[-100:]
        elif r["status"] != "PASS":
            why = (r.get("failed_tests") or r.get("errors") or r.get("infra")
                   or [r.get("bk_error") or ""])[0][:100]
        print(r["build"], (r["branch"] or "")[:24], r["commit"][:8], r["status"],
              f"try{r['attempt']}", f"{r['minutes']}min", r["agent"],
              f"mask={r.get('mask')}", f"gib={r.get('gpu_gib')}",
              f"last_ts={r.get('last_log_ts')}", r.get("pytest_summary") or "", why)
    sys.exit()

if sys.argv[1] == "local":
    raw = Path(sys.argv[2]).read_text(errors="replace")
    print(json.dumps(summarize(NOISE.sub("", raw), {"log": sys.argv[2]}), ensure_ascii=False))
    sys.exit()

org, pipe, build, job_re, back, fwd, scope, out = sys.argv[1:9]
build, back, fwd, job_re, out = int(build), int(back), int(fwd), re.compile(job_re), Path(out)
out.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(8) as ex:
    for rows in ex.map(scan, range(build - back, build + fwd + 1)):
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
EOF
```

2. **Fetch the failed jobs of the target build** (parse `<ORG>/<PIPELINE>/<BUILD>` from the URL, e.g. `vllm intel-ci 17266`):

   ```bash
   mkdir -p /tmp/ci-<BUILD>
   python3 /tmp/intel-ci-fix/bk_scan.py <ORG> <PIPELINE> <BUILD> '.' 0 0 FAIL /tmp/ci-<BUILD> > /tmp/ci-<BUILD>/jobs.jsonl
   python3 /tmp/intel-ci-fix/bk_scan.py table < /tmp/ci-<BUILD>/jobs.jsonl
   ```

   When you need the full fields, read only the matching line of `jobs.jsonl`. Key fields:
   - `status`: `PASS` / `SOFT` / `FAIL` / `TIMEOUT` (job aborted by its timeout) / `CANCELED` / `EXPIRED`. Logs of timed-out jobs also contain `Received cancellation signal`, so **use `status` to tell a timeout from a manual cancel**.
   - `attempt`, `retried_in`: the retry number within the same build and the job it was retried in. `FAIL/TIMEOUT try1` → `PASS try2` means it is intermittent, but **does not mean it can be ignored**.
   - `minutes`: job duration. A duration far above the normal passing duration usually means a hang.
   - `image`, `final_commands`, `failed_tests`, `errors` (including root causes from worker subprocesses, e.g. `(Worker pid=..) ERROR ... ValueError`).
   - `bk_error` (the Buildkite `🚨 Error:` line), `infra` (environment errors such as DNS, docker daemon, device lost, insufficient memory at startup, HF gated repo), `last_log_line` (the last timestamped log line, used to locate where a run hung), `tail` (the last 12 lines of the log).
   - `mask` (`ZE_AFFINITY_MASK`; number of commas + 1 is the card count), `gpu_gib` (total memory per card: 15.13 GiB ≈ B50 16 GB card, 22.71 GiB ≈ B60 24 GB card), `agent` (e.g. `B50-cicd01-6_7` encodes card type, machine, and card IDs), `shard_items` (number of test items in a parallel shard; the full list is in the log line `Running N items in this shard:`).

3. **Pull the image in the background right away**: as soon as you have `image`, start the pull in the background and do not wait for it (a pull usually takes several minutes). If the job is clearly an Infra failure where tests never started (see step 5), you can skip the pull for now. Only newly pulled images are recorded in `pulled_images.txt`; Phase 5 deletes only those images, so images that already existed locally are kept.

   ```bash
   if ! docker image inspect <IMAGE> >/dev/null 2>&1; then
     echo <IMAGE> >> /tmp/ci-<BUILD>/pulled_images.txt
     (nohup docker pull <IMAGE> > /tmp/ci-<BUILD>/pull.log 2>&1 &)
   fi
   ```

4. **Scan the job history**: for each failed job, use the stable part of its name as the regex (escape `+ ( )`), and scan about 30 builds back and 10 builds forward. `FAIL` downloads only non-passing logs; use `ALL` when you need passing logs for comparison.

   ```bash
   python3 /tmp/intel-ci-fix/bk_scan.py <ORG> <PIPELINE> <BUILD> 'Sample \+ Logits' 30 10 FAIL /tmp/ci-<BUILD>/history \
     > /tmp/ci-<BUILD>/history.jsonl
   python3 /tmp/intel-ci-fix/bk_scan.py table all < /tmp/ci-<BUILD>/history.jsonl
   ```

   - Explain **every other** non-PASS record in the history too (including `TIMEOUT`s hidden by automatic retries), not just the target build.
   - For the target failing tests, download passing logs with `ALL` and `grep` whether each test was `PASSED` or `SKIPPED` (a skipped "pass" is not evidence that the test passes).

5. **Classify** (give one conclusion with evidence for each failing test or failing job):
   - **Deterministic (regression)**: on `main`, fails consistently from some build onward across different agents and card types, and passed consistently before. Record the commits of the last passing and first failing builds, and run `git log --oneline <PASS_COMMIT>..<FAIL_COMMIT> -- <relevant paths>` to list suspect PRs.
   - **Flaky**: passes and fails on the same commit or adjacent builds; or the same test fails sporadically on different cards with different values or line numbers each time; or it passes on retry.
   - **Hang / Timeout**: `status=TIMEOUT`, or `minutes` far above the normal duration. Compare `last_log_line` across runs: if several runs (especially across card types) hang at the same point, it is a real hang. Treat it as Flaky, require a stress repro, and report it even if the automatic retry passed.
   - **Hardware/environment related**: the failure strongly correlates with `agent`, `gpu_gib` (card type), or `mask` (card count). For example, a multi-card test only really runs when ≥2 cards are assigned and is `SKIPPED` in all other runs. Other examples: insufficient memory at startup (`... on startup is less than desired GPU memory utilization`), device lost, or the card being used by another process.
   - **Infra**: tests never started (`pytest_summary` is empty, and `infra` or `tail` shows image pull, DNS, or docker daemon errors); or the agent was lost, the job was manually canceled (`CANCELED`), or there was an HF download or network error.
   - Builds on PR branches (`branch != main`) are for reference only; decide based on `main`. A failure that appears only on a single PR branch was usually introduced by that PR.
   - **Early exit**: if a failed job is Infra, and the same job later passed on `main` (ideally on the same agent), **skip Phase 2/3** for that job and go straight to the Phase 4 Infra template. All other categories still need to be reproduced.

6. **Summary table** (output to the user):

   | Failed Job | Image | Failed Test / Pytest Command | Status (try/duration) | Error Type | History (fails/total on main) | Category | Key Evidence (card type/count/agent/hang point) |
   |---|---|---|---|---|---|---|---|

   Example error types: Accuracy Mismatch, OOM / insufficient memory at startup, SYCL Kernel Error, Hang / Timeout, Import Error, Infra (DNS/image pull/download/network). The category determines the Phase 2 repro strategy:
   - Deterministic: reproduce directly;
   - Flaky / Hang: stress-test to reproduce;
   - Hardware related: match card type and count, or simulate CI's memory headroom;
   - Infra: early exit as described in step 5.

## Phase 2: Preflight & Reproduce

1. **Preflight**. Note any unmet item in the report; for items that affect the conclusion, confirm with the user via `ask_user` first:
   - **Available cards**: if the user specified available cards (e.g. "the first four cards"), choose only from those; otherwise list devices with `xpu-smi discovery` and confirm the target cards are idle with `xpu-smi stats -d <ID>`. Match the card count to CI's `mask`.
   - **Card type**: match the local card type (Device Name in `xpu-smi discovery`) to the CI `agent` prefix (B50/B60) and `gpu_gib`. If they differ, treat memory-related conclusions as reference only.
   - **Credentials and network**: `[ -n "$HF_TOKEN" ] || echo "HF_TOKEN missing"`. Without `HF_TOKEN`, gated models (e.g. `meta-llama/*`) fail with `Cannot access gated repo`; such failures do not count as a repro or a regression. If gated models are needed, ask the user for a token first; otherwise list those tests separately in the results and exclude them. Also confirm `http_proxy`/`https_proxy`/`no_proxy` are set.
   - **Image**: wait for the background pull from Phase 1 to finish:
     ```bash
     until docker image inspect <IMAGE> >/dev/null 2>&1; do
       pgrep -f "docker pull <IMAGE>" >/dev/null || { echo PULL_EXITED; tail -3 /tmp/ci-<BUILD>/pull.log; break; }
       sleep 15
     done
     ```
2. **Code version**: the image is built from the failing CI commit, so **reproduce in a separate worktree at the failing commit** (do not touch the current workspace). This keeps the code consistent with the dependencies and build artifacts in the image:
   ```bash
   git fetch origin <COMMIT> && git worktree add --detach /tmp/ci-<BUILD>/wt <COMMIT>
   ```
   Then use `git log --oneline <COMMIT>..origin/main -- <relevant paths>` to check whether `main` already has a related fix.
3. Narrow the failure down to a **single minimal failing op or test function** (e.g. narrow `pytest tests/kernels/` to `pytest tests/kernels/test_attention.py -k "test_xxx"`).
4. Run that minimal test on XPU (referred to below as the "repro command"). Do not add `-t`; otherwise the log contains `\r` and color codes that break `grep`:
   ```bash
   TEST_CMD='cd tests && pytest -v -s <MINIMAL_TEST>'
   docker run --rm --label intel-ci-fix=<BUILD> --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
     -e http_proxy -e https_proxy -e no_proxy -e HF_TOKEN -e PY_COLORS=0 \
     -e ZE_AFFINITY_MASK=<ZE_AFFINITY_MASK> \
     -v <HF_HOME>:/root/.cache/huggingface -v /tmp/ci-<BUILD>/wt:/workspace/vllm -w /workspace/vllm \
     --entrypoint timeout <IMAGE> --kill-after=30s <TIMEOUT> bash -c "$TEST_CMD" \
     > /tmp/ci-<BUILD>/repro.log 2>&1; echo EXIT=$?
   python3 /tmp/intel-ci-fix/bk_scan.py local /tmp/ci-<BUILD>/repro.log
   ```
   - Copy environment variables from the CI command (e.g. `VLLM_WORKER_MULTIPROC_METHOD=spawn`) into `TEST_CMD` verbatim.
   - `<TIMEOUT>`: use `30m` for a minimal test; for a full suite use 1.5× the `minutes` of passing runs in the history (at least `30m`).
   - **Flaky**: run in a loop (e.g. 10 times) and count the failures.
   - **Hang**: run in a loop with each timeout set to 2× the normal duration, and record the last log line when it hangs. If possible, capture the Python stack inside the container with `py-spy dump --pid <PID>` (run `pip install py-spy` first if the image lacks it).
5. **Check the repro result**:
   - Same error reproduced locally → go to Phase 3.
   - If the minimal test passes locally (cannot reproduce), run the full test suite of the original failed job with the same image, environment, and repro command: reuse `final_commands`; for parallel-shard jobs, use the test list from the log line `Running N items in this shard:` instead of a single test.
     - Test suite reproduces the same error → go to Phase 3.
     - Test suite fully passes → mark it `Cannot Reproduce` in the report, recording the commands and results of both the minimal test and the test suite.
   - If the suite did not finish, or only **other** errors appeared (e.g. gated failures caused by a missing `HF_TOKEN`) → record what actually happened; it counts as neither reproduced nor fully passing. Reconcile the local pass count with the CI pass count (e.g. local passed + environment failures = CI passed) to show the difference comes entirely from the environment.
   - Every container created in this phase must carry `--label intel-ci-fix=<BUILD>` (including stress loops and ad-hoc debug containers), so Phase 5 can find and remove containers left behind by timeouts or interruptions. The repro worktree is cleaned up in Phase 5.

## Phase 3: Diagnose & Fix
1. Locate the relevant source code in the host repository.
2. Before writing a fix, check for existing work to avoid duplicates:
   ```bash
   gh pr list --repo vllm-project/vllm --state open --search "<failing test name or keywords>"
   gh issue list --repo vllm-project/vllm --state open --search "<failing test name or keywords>"
   ```
   If an existing PR or issue already covers it, cite it in the report instead of writing another fix.
3. If the problem likely comes from PyTorch, the underlying driver, or oneCCL, try to provide a minimal PyTorch repro script that does not depend on vLLM, keeping only the ops or distributed communication needed to trigger the problem. Run it with the same image and XPU environment, and attach the script, the run command, relevant component versions, and actual results to the report. If it cannot be reproduced independently, state the limitation and do not attribute the problem to the lower layer directly.
4. If it is confirmed to be a vLLM problem, write a minimal fix patch, **within these safety boundaries**:
   - Base the fix on a separate branch or worktree from the latest `origin/main`. If the latest `main` cannot run in the image because of dependency or build-artifact mismatches, verify the patch in the failing-commit worktree and say so in the report.
   - After every change, rerun the Phase 2 repro command to confirm the change works and does not break other tests in the same file. For Flaky/Hang issues, verify with the same number of loop iterations.
   - Make at most **3** fix iterations; if the problem is still unsolved after 3, keep detailed diagnostic logs and roll back changes that did not work.

## Phase 4: Report
Every report includes the **build overview and Failed Jobs summary table** (including anomalies such as TIMEOUTs hidden by retries in the history) and **environment information** (image tag, local card type and cards used, and preflight differences from CI such as `HF_TOKEN`). Choose the rest of the content by category:

- **Infra (early exit)**: root-cause evidence (verbatim `bk_error`/`infra`), the later passing build and agent, and the recommended action (retry the job, or forward the agent's DNS/network/disk problem to the infra team). No repro or patch needed.
- **Fixed**: repro command and status, in-depth root-cause analysis, the `git diff` patch, and log evidence that the repro command passes after the fix (with loop counts for Flaky/Hang).
- **Flaky / Hang (not fixed)**: historical failure rate (fails/total on main), the hang point or failing values of each run, stress-test commands and local results, and related existing PRs/issues.
- **Cannot Reproduce**: commands and results of the minimal test and the full suite, differences from the CI environment, and suggested next steps.

## Phase 5: Cleanup
After the report is produced, remove the images pulled, containers created, and worktree used in this session. Keep the log directory `/tmp/ci-<BUILD>` for the report to reference. If the user asks to keep the image or environment for further debugging, skip the corresponding steps.
