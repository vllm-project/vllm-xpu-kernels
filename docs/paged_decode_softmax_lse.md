# Paged Decode `softmax_lse` Support (XPU / Xe2)

**Repository:** `vllm-project/vllm-xpu-kernels`
**Date:** 2026-08-23
**Scope:** paged-decode FMHA, split-K reduction, Python planning, and tests

---

## 1. Summary

The XPU paged-decode attention path now returns `softmax_lse` (the per-query
log-sum-exp of the attention scores), matching the behaviour the chunked-prefill
kernel already offered. This is required by callers that need to merge attention
outputs computed over disjoint KV ranges — cascade / chunked attention, prefix
sharing, and pipeline-parallel KV splitting all rely on the LSE to rescale and
combine partial results.

The interesting part of the problem, and the part the request explicitly called
out, is that paged decode is not a single kernel. Depending on the split-K
decision it runs either one kernel or two:

```
num_kv_splits <= 1 :  XeFMHAFwdSplitKVKernel                       (1 kernel)
num_kv_splits  > 1 :  XeFMHAFwdSplitKVKernel  ->  ReduceSplitK     (2 kernels)
```

Only the *last* kernel in the chain holds the complete softmax statistics for a
query row, so LSE has to be produced from a different place in each case without
ever being written twice.

While validating the change, three genuine correctness bugs surfaced in the
existing decode path. Two of them corrupted the attention **output**, not just
the new LSE. All three are fixed here.

---

## 2. Background: the LSE math convention

The decode mainloop pre-multiplies `sm_scale` by `log2(e)`
(`chunk_prefill_mainloop.hpp`), so the running softmax statistics live in the
**log2 domain**:

```
rA_max = max_j (s_j * log2e)
rA_sum = Σ_j exp2(s_j * log2e − rA_max)
```

Converting to the natural-log LSE that callers expect:

```
lse = rA_max * ln2 + ln(rA_sum)          ln2 = 0.6931471805599453
```

Each split converts its statistics to one natural-log LSE and normalizes its
partial output:

```
LSE_i = max_i * ln2 + ln(sum_i)
O_i   = numerator_i / sum_i
```

The reducer follows the CUDA FlashAttention contract:

```
LSE_global = logsumexp_i(LSE_i)
weight_i   = exp(LSE_i - LSE_global)
O          = Σ_i weight_i * O_i
```

The attention sink, when present, is folded into `rA_sum` by the epilogue at
`idx_kv_split == 0`, so it flows into the LSE automatically. The reference
implementation must therefore include `exp(sink)` in its logsumexp.

**Buffer layout:** `(num_heads_q, total_seqlen_q)`, float32,
`lse_stride = total_seqlen_q`. This matches the CUDA / upstream FlashAttention
convention and the layout chunk_prefill already writes, so callers need no
transpose and the two paths can share one output tensor.

---

## 3. Design

### 3.1 Single writer, chosen at runtime

`ptrLSE` is computed in the FMHA kernel as:

```cpp
ElementLSE* ptrLSE =
    (p.softmax_lse != nullptr && num_kv_splits <= 1)
        ? p.softmax_lse + head_q_start * p.lse_stride + offset_lse
        : nullptr;
```

The condition `num_kv_splits <= 1` is exactly the negation of `need_reduce` in
`DecodeKernelLauncher::run`, so:

- when no reduce pass follows, the FMHA epilogue writes the LSE;
- when a reduce pass follows, the epilogue writes nothing and `ReduceSplitK`
  owns the write.

No races, no double-writes, no cross-kernel coordination needed.

### 3.2 CUDA-style normalized partial outputs

The FMHA epilogue always normalises each active split and writes one natural-log
LSE:

```cpp
row_lse = rA_max * ln2 + log(stats_sum);
softmax_lse_accum(q_row, split) = row_lse;
rA *= broadcast(1 / rA_sum);
```

`ReduceSplitK` computes `logsumexp` over those per-split LSE values and combines
the normalized outputs with `exp(LSE_i - LSE_global)`. This replaces the old
XPU-specific contract of unnormalized `Oaccum` plus separate `exp_sums` and
`max_logits` buffers.

The new contract has three useful properties:

- it matches CUDA FlashAttention's split-K representation;
- it needs one float32 statistics buffer instead of two;
- one effective split is naturally a pass-through (`weight = 1`), so the old
  single-split sentinel is no longer needed.

### 3.3 Runtime pointer instead of a template flag

chunk_prefill gates LSE on a `SoftmaxLSE_` template parameter, which is why its
LSE support is restricted to the `!Paged && !Local && !Sink` specialisation
(`fmha_xe2.cpp` rejects the rest). Decode already has a very large instantiation
matrix, and templating would double it.

Using a runtime null-pointer check keeps the instantiation count flat **and**
means decode LSE works for paged, sliding-window, and sink configurations with
no restrictions — strictly more capable than the prefill path.

---

## 4. Bugs found and fixed

> See [`paged_decode_bug_analysis.md`](./paged_decode_bug_analysis.md) for a
> diagrammed walkthrough of each subgroup's job and why these three bugs were
> invisible to the test suite.

### 4.1 Attention-sink row mis-mapping (corrupted output, pre-existing)

**Severity: high — silently wrong attention output whenever sinks were used.**

The epilogue reports softmax statistics per packed-GQA row `q_row`:

```cpp
int q_row = get<0>(blk_qv) * q_tile_rows + thr_id;
```

but the sink was folded into `rA_sum(0)` using a *different*, output-tile-derived
mapping:

```cpp
int base_row = get<0>(tOgO(_0{}, _0{}, _0{}));
int row_i    = base_row + (lane % size<0>(SGTileShapeO{}));
```

That second mapping is correct for normalising `O`, but `SGTileShapeO` is
`(2, 64)` while `TileShapeO` is `(8, 64)`, so the statistics path effectively
received `sink[row % 2]`.

This is nearly invisible with random sinks over a long KV cache (error ~1e-4),
which is why it survived. I isolated it by compiling a shape probe to recover the
CuTe layout constants, then designing a discriminating experiment: set
`sink[h] = 20 + (h % head_group_q)`, far above every score, so the recovered LSE
reads back the sink index directly.

```
H=8 KVH=2  expected rows [0,1,2,3,0,1,2,3]
           before fix    [0,1,0,1,0,1,0,1]     <-- rows 2,3 got sink of rows 0,1
           after fix     [0,1,2,3,0,1,2,3]
```

**Fix.** Snapshot the pre-sink denominator, and re-apply the sink at `q_row`
explicitly for the statistics path. The output normalisation path is left
untouched:

```cpp
ElementA stats_sum = rA_sum(0);
if constexpr (Sink) {
  if (row_valid && idx_kv_split == 0) {
    stats_sum += exp2(tSink(q_row) * kLog2e − rA_max(0));
  }
  ... // existing output-path sink fold, unchanged
}
```

The per-split LSE and direct `softmax_lse` output both use `stats_sum`.

**Impact beyond LSE.** Under the old unnormalized-partial contract,
`ReduceSplitK` used `exp_sums` as the final denominator, so the wrong sink
propagated into the result. Measured on a 2048-token sequence: head 2 produced
`0.1074` against a reference of `0.0537` — a 2× error.

### 4.2 `ReduceSplitK` discarded live splits in compact-grid mode

**Severity: high — dropped attention contributions.**

The reducer filtered "empty" splits with:

```cpp
if (i * num_blocks_per_split >= windowed_k_blocks) break;
```

where `num_blocks_per_split = ceil_div(windowed_k_blocks, seq_num_kv_splits)`.
This is self-consistent with the *legacy* on-device partition, which uses the
same formula. But the host planner `build_decode_split_plan` partitions evenly
into `base` / `base + 1` tiles — a strictly finer partition.

Counter-example: 33 tiles over 8 splits. `ceil(33/8) = 5`, so `7 * 5 = 35 >= 33`
and split 7 is discarded — even though the host plan guarantees it owns real
tiles.

**Fix.** Branch on the mode. The host plan already guarantees every emitted split
owns at least one tile, so the guard is simply skipped:

```cpp
const bool plan_driven = (p.splits_per_seq != nullptr);
...
if (!plan_driven && i * num_blocks_per_split >= windowed_k_blocks) break;
```

Applied at all three sites: the SLM-fill condition, the LSE accumulation loop,
and the output accumulation loop.

### 4.3 `build_decode_split_plan` ignored the sliding window

**Severity: high — wrong output for sliding-window + split-plan.**

For decode, all packed GQA heads sit at position `kv_len − 1`, so a left window
of `W` makes every tile before `k_block0` unreachable:

```
k_block0 = max(kv_len − 1 − W, 0) / kv_tile
```

The kernel applies this offset per work item
(`kv_split_offset = k_block0 + wl_tile_start`), but the host planner partitioned
the *full* `ceil_div(kv_len, kv_tile)` tile count. The result was a work list
that ran off the end of the window, producing 62.6% mismatched output elements
in the affected configurations.

This combination had **zero prior test coverage** — `host_kv_lens` appeared
nowhere in the test suite before this change.

**Fix.** Make the planner window-aware, mirroring the kernel's `k_block0`:

```python
def _windowed_tiles(kv: int) -> int:
    total = (kv + kv_tile - 1) // kv_tile
    if window_size_left is not None and window_size_left >= 0:
        total -= max(kv - 1 - window_size_left, 0) // kv_tile
    return max(1, total)
```

The call site normalises `-1` to `max_seqlen_k` exactly as `flash_api.cpp` does,
so host and device agree on what "unbounded left edge" means.

---

## 5. Files changed

| File | Purpose |
| --- | --- |
| `csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp` | Per-split LSE plumbing; CUDA-style LSE-weighted reducer; `plan_driven` fix |
| `csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp` | Normalize every partial output; write per-split/direct LSE; sink `stats_sum` fix |
| `csrc/xpu/attn/xe_2/paged_decode_xe2.cpp` | Tensor validation and one-buffer plumbing |
| `vllm_xpu_kernels/flash_attn_interface.py` | Window-aware `build_decode_split_plan` |
| `docs/group_split_kv_design.md` | Window-aware planning and `plan_driven` filtering |
| `csrc/xpu/attn/xe_2/paged_decode.hpp` | Wire `softmax_lse_accum` through both kernels |
| `csrc/flash_attn/flash_api.cpp` | Allocate one per-split LSE buffer and pass final LSE output |
| `csrc/xpu/attn/attn_interface.{h,cpp}` | Signature plumbing |
| `csrc/xpu/attn/xe_2/paged_decode_{utils.hpp,xe2.h}` | Signature plumbing |
| `tests/flash_attn/test_flash_attn_varlen_func.py` | Reference implementation and two regression tests |

No public op schema changed, so `custom-kernels-inventory.md` needs no update.

---

## 6. Testing

### New coverage

**`test_decode_with_paged_kv_softmax_lse`** — 312 cases. The grid is chosen so
each of the three statistic-producing paths is exercised:

| Configuration | Path under test |
| --- | --- |
| `num_splits_kv = 1` | FMHA epilogue writes LSE, no reduce |
| `num_splits_kv = 8`, long sequences | `ReduceSplitK` combines per-split statistics |
| `num_splits_kv = 8`, short sequences | one normalized partial output round-trips through the reducer |
| `use_split_plan = True` | compact-grid / `work_list` path (`plan_driven`) |

Crossed with head configs `(8,2)` and `(16,16)`, head sizes 64/128, block sizes
16/64, and `extra ∈ {no-sink, sink, sliding-window(127)}`. A new
`ref_paged_decode_softmax_lse` helper provides the paged / window / sink-aware
reference.

**`test_decode_paged_kv_sink_row_mapping`** — 16 cases. A dedicated regression
guard for §4.1. Random sinks cannot detect that bug, so this test uses dominant
sinks (`20 + row`) and asserts each head recovers *its own* sink index, across
`(8,1) / (8,2) / (8,4) / (16,2)` head groupings, both split modes, and both a
very short (5) and a long (2048) KV length.

### Results

```
tests/flash_attn/  ->  3903 passed, 1216 skipped, 0 failed   (64s)
```

Verified on Intel BMG (`bmg-g21-a0`), oneAPI 2026.1, pinned to a single GPU via
`ZE_AFFINITY_MASK=0`.

Build: clean, zero compile errors. `pre-commit` (ruff, yapf, clang-format,
codespell, isort, mypy, PyMarkdown, SPDX) all pass.

---

## 7. Known limitation

`return_softmax_lse=True` on a **mixed prefill + decode** batch still raises:

```
softmax_lse output is only supported when is_paged=false, is_local=false, is_sink=false
```

This restriction is pre-existing and lives in the chunk_prefill kernel, which
gates LSE on a template parameter and only instantiates the
`!Paged && !Local && !Sink` variant. The decode side of that path is fully wired
here, so the mixed-batch case will work as soon as the prefill restriction is
lifted — no further decode-side work required.

---

## 8. Build and verification commands

```bash
source /home/baodi/set_env_2026.1.sh
MAX_JOBS=48 xpu_build .venv/bin/python setup.py build_ext --inplace

ZE_AFFINITY_MASK=0 .venv/bin/python -m pytest tests/flash_attn/ -q
ZE_AFFINITY_MASK=0 .venv/bin/python -m pytest \
    tests/flash_attn/test_flash_attn_varlen_func.py -k "softmax_lse" -v
```

Notes:

- `xpu_build` strips `/opt/umds` from `LD_LIBRARY_PATH`; that IGC ICEs when
  AOT-compiling attention kernels. Do not use it when *running* tests.
- Keep `MAX_JOBS` around 48. The build assumes ~8 GB per compile process, and
  `MAX_JOBS=200` OOM-kills `icpx` on a 499 GB machine.
