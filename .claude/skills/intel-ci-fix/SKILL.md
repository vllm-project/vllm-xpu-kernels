---
name: intel-ci-fix
description: "Use when an Intel XPU CI (Buildkite vllm/intel-ci) build or job fails: fetches failed job logs, triages deterministic vs flaky failures, checks for duplicate PRs, reproduces in the matching <commit>-xpu CI docker image on a local Intel GPU, finds the root cause, and prepares a verified local fix branch plus a report for human review. Trigger phrases: intel ci failure, intel-ci build, XPU CI broken, BMG job failed, buildkite.com/vllm/intel-ci."
argument-hint: "A Buildkite intel-ci build/job URL (e.g. https://buildkite.com/vllm/intel-ci/builds/16493), or 'latest main' to scan recent main builds."
---

You diagnose and fix Intel XPU CI failures in vLLM (`vllm-project/vllm`, pipeline `vllm/intel-ci`).
Your output is a verified local fix branch in an isolated worktree, a ready-to-use PR description file, and a structured report.

## Hard Constraints

- NEVER push branches, open PRs, comment on PRs or issues, or retrigger CI automatically. Prepare the local commit and put the push / PR creation commands in **Next step for the user**.
- Only commit a verified fix (fails before, passes after). Otherwise report it under **Not fixed**.
- DO NOT delete or stop docker images or containers that you did not create. This machine is shared.
- DO NOT modify the user's main working tree. Work in a separate `git worktree` under `/tmp/ci-fix-<slug>`.
- If an open PR already fixes the failure, do not write a competing fix. Validate that PR instead and report review comments.
- Use only the GPUs the prompt or `ZE_AFFINITY_MASK` allows (for example `0,1`). Other cards belong to other users.
- Keep the session short. Stop and write the report, marking unresolved items under **Not fixed** with what you found and suggested next steps, if:
  - the failure does not reproduce after about 2 actual reproduction runs (triage alone is not an attempt),
  - the root cause is still unclear after about 3 fix attempts,
  - the fix would need a large refactor, a kernel or toolchain change, or a dependency upgrade,
  - or the work needs resources you don't have, such as a missing image, insufficient disk, or no healthy GPU.
  Do not keep trying blindly.

## Workflow

### 1. Fetch the failed jobs

Run [scripts/ci-fetch-log.sh](./scripts/ci-fetch-log.sh) directly from the repository root. By default it saves cleaned logs to `/tmp/ci-<build>-<job>.log`:

```bash
bash .agents/skills/intel-ci-fix/scripts/ci-fetch-log.sh --metadata "https://buildkite.com/vllm/intel-ci/builds/<build>"
```

From each failed job log, extract:
- `HEAD is now at <sha>`: the tested commit.
- `Using image: public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:<sha>-xpu`: the exact image.
- The `short test summary info` block and the first lines starting with `E` of each failure: this is the error signature.

### 2. Triage

- Check the same job across neighboring builds on both `main` and PR branches.
- A failure is **deterministic** if it fails with the same signature on 2 or more builds, or on different branches. Otherwise it is **flaky** (or **infra** if pytest never started due to Docker/network/device-lost).
- Flaky examples: pytest-timeout on TP=2 startup or shutdown, single-token stream divergence, threading Event timing. Record flaky/infra failures under **Not fixed** and do not fix them.
- For deterministic failures, find the trigger PR:
  1. For the same job on `main`, find the last passing build and the first failing build. Take their commits as `<good>` and `<bad>`.
  2. Run `git --no-pager log --oneline <good>..<bad> -- <test file> <paths in traceback frames>`. vLLM squash-merges, so `(#NNNNN)` in the commit title is the PR number.
  3. Confirm it during reproduction (step 4) when fast: revert the suspect commit and check that the test passes again. Report the PR as **confirmed** only if verified; otherwise **suspected**.
- Note whether the job's device (`B50` 16 GiB vs `B60` 24 GiB) or GPU count (`ZE_AFFINITY_MASK`) matters.

### 3. Check for duplicates (mandatory, before writing code)

```bash
gh pr list --repo vllm-project/vllm --state open --search "<test_name>"
gh pr list --repo vllm-project/vllm --state open --search "<file or kernel keywords>"
gh issue list --repo vllm-project/vllm --state open --search "<test_name>"
gh pr list --repo vllm-project/vllm --state merged --search "<test_name>" --limit 10
```

Also search `[XPU]` and `intel-gpu` titles from the last few days, and check whether the job already passes on the latest `main` builds. If a merged PR already fixed it, report that PR and stop.

### 4. Reproduce in the matching image

- **Image**: prefer `public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:<sha>-xpu`.
  - Check free disk space first (`df -h /`). An image is ~20 GB uncompressed.
  - If space is short, reuse a local `*-xpu` image only if its `triton` / `torch` / `vllm-xpu-kernels` versions match `requirements/xpu.txt` at `<sha>`. Version mismatches hide bugs.
- **Code**: `git worktree add -f /tmp/ci-fix-<slug> <sha>`.
- **Run**:

  ```bash
  docker run --rm --name ci-fix-<slug> --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --net=host -e ZE_AFFINITY_MASK=<gpu> \
    -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e no_proxy="localhost,127.0.0.1" \
    -e HF_TOKEN=$HF_TOKEN -v ${HF_HOME}:/root/.cache/huggingface \
    -v /tmp/ci-fix-<slug>:/workspace/vllm \
    --entrypoint bash $IMG -c 'cd /workspace/vllm/tests && python3 -m pytest -q <nodeid>'
  ```

- Reproduce the **same error signature** as in CI before moving on. Prefer writing a minimal standalone repro script when debugging kernels/ops.

### 5. Fix and verify

- Base the fix on the latest upstream `main`, not on `<sha>`:
  `git fetch <upstream_remote> main && git -C /tmp/ci-fix-<slug> switch -c ci-fix/<date>-<slug> <upstream_remote>/main`.
  Confirm the bug still exists on `main`.
- Make the smallest correct fix. Follow [AGENTS.md](../../../AGENTS.md), keep lines <= 88 chars, and gate XPU-specific workarounds on `current_platform.is_xpu()`.
- Verify in the same image:
  1. The failing test fails before the fix and passes after it.
  2. Run the nearest related test file **without and with** the fix, and diff the pass/fail lists.
  3. If the fix touches kernels, compare numerics against a reference implementation.
- Run `pre-commit run --files <changed files>` on the host if installed.
- Commit on `ci-fix/<date>-<slug>` with `git commit -s` and `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

### 6. Clean up and prepare PR description

- Remove your containers and only images you pulled during this session (if disk space is tight).
- Keep the fix worktree `/tmp/ci-fix-<slug>` and branch `ci-fix/<date>-<slug>` so the user can review them.

## Output Format

Reply in the user's language with these sections:
1. **Failure table**: job | test | signature | deterministic or flaky (with evidence: build numbers and branches).
2. **Root cause**: the trigger PR (link, author, merge date, confirmed or suspected), `<good>..<bad>` range, mechanism, and why it is XPU-specific.
3. **Reproduction**: image tag, versions (`torch` / `triton` / `vllm-xpu-kernels`), GPU, exact command, and observed error.
4. **Fix**: worktree path, branch name, base commit, diff summary, and review command `git -C /tmp/ci-fix-<slug> diff <remote>/main...HEAD` (or existing duplicate PR validation).
5. **Validation**: before/after results and baseline-vs-fix diff of related tests.
6. **PR description**: the body saved to `/tmp/ci-fix-<slug>.md`.
7. **Not fixed**: flaky/infra failures and anything not reproduced, with reasons.
8. **Next step for the user**: commands for the user to review the diff, push the branch to their fork, and open a PR if satisfied:

   ```bash
   git -C /tmp/ci-fix-<slug> diff <upstream_remote>/main...HEAD
   git -C /tmp/ci-fix-<slug> push <fork_remote> ci-fix/<date>-<slug>
   gh pr create --repo vllm-project/vllm --base main \
     --head <user>:ci-fix/<date>-<slug> \
     --title "[XPU][CI] <short summary>" --body-file /tmp/ci-fix-<slug>.md
   ```
