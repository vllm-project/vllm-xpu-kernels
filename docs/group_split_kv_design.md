# Group Split-K for Mixed-Length Batch Decoding (Xe2)

## 1. Problem statement

The existing `paged_decode_xe2` kernel uses a static `(batch × heads_kv × num_kv_splits)`
work-group (WG) grid. For mixed-length decode batches this has two failure modes:

1. **Idle WGs.** A short sequence gets the same `num_kv_splits` slots as a long
   one. Splits whose tile range is empty exit immediately but still occupy a WG
   slot, so the launched grid is much larger than the work it carries.
2. **Unfair partition.** `num_blocks_per_split = ceil_div(blocks, splits)`
   concentrates work on the first `floor(blocks / blocks_per_split)` splits;
   the trailing split can be tiny or empty.
3. **Hidden GPU→CPU sync.** Any C++-side decision that wants to look at per-seq
   KV lengths must either pay for a `.item()` round-trip or duplicate the
   logic the scheduler already needs in Python.

The vLLM scheduler already knows `kv_lens` on the host. The PR pushes the
split planning to Python and makes the GPU read a precomputed `work_list`.

## 2. Architecture overview

```
                  vLLM scheduler                  (host)
                         │   kv_lens, num_splits_kv,
                         │   num_xe_cores, num_heads_kv
                         ▼
       flash_attn_interface.build_decode_split_plan      (host, Python)
                         │   splits_per_seq[B]    int32
                         │   work_list[total_wgs, 4]   int32
                         ▼  (.to(device, non_blocking=True))
              flash_attn_varlen_func → varlen_fwd        (PyTorch op)
                         │
                         ▼
              cutlass_paged_decode_impl                  (C++)
                         │  fills args.splits_per_seq,
                         │  args.work_list, args.total_wgs
                         ▼
            DecodeKernelLauncher::run                    (host launch)
                         │  patches scheduler.work_list,
                         │  scheduler.total_wgs,
                         │  scheduler.grid.z = total_wgs × heads_kv
                         ▼
                ┌────────┴────────┐
                ▼                 ▼
      XeFMHAFwdSplitKVKernel    ReduceSplitK
      (compact-grid branch)    (uses splits_per_seq
       reads work_list[wi]      directly, no heuristic
       → seq_idx, tile_start,    reapplied)
         tile_count, split_idx
```

Two execution modes live side-by-side in the kernels:

- **Legacy mode** (`work_list == nullptr`): identical to pre-PR behavior. The
  scheduler builds a `(B × H_kv × num_kv_splits)` grid and each kernel
  invocation reapplies the `is_single_split` heuristic on its own.
- **Compact mode** (`work_list != nullptr && total_wgs > 0`): grid is exactly
  `total_wgs × heads_kv`. Each WG reads one `DecodeWorkItem` from `work_list`
  and processes exactly its tile range. The kernel does no per-seq split math.

## 3. Algorithm — `build_decode_split_plan`

Inputs: `kv_lens[B]`, `kv_tile`, `num_kv_splits`, `num_xe_cores`, `num_heads_kv`.

```
tiles_per_seq[i]      = max(1, ceil_div(kv_lens[i], kv_tile))
total_tiles           = Σ tiles_per_seq

min_wgs               = max(1, num_xe_cores * 2 / num_heads_kv)
target_tiles_per_wg   = max(4, total_tiles / min_wgs)
min_blocks_for_split  = 32 if kv_tile ≤ 64 else 128

for each seq i:
    if tiles ≤ target_tiles_per_wg  or  tiles < min_blocks_for_split  or  num_kv_splits ≤ 1:
        n_splits = 1                              # single-split path
    else:
        n_splits = ceil_div(tiles, target_tiles_per_wg)
        n_splits = min(n_splits, num_kv_splits, tiles)   # cap

    base, rem = divmod(tiles, n_splits)
    for s in 0..n_splits-1:
        count = base + (1 if s < rem else 0)
        emit work_item(seq_idx = i,
                        kv_tile_start = running_offset,
                        kv_tile_count = count,
                        split_idx     = s)
        running_offset += count
```

### Correctness contract

The function returns `(splits_per_seq, work_list)` with these guarantees:

1. `Σ splits_per_seq == work_list.shape[0]`.
2. Every emitted `kv_tile_count ≥ 1`. (Cap `n_splits ≤ tiles` makes
   `base ≥ 1` regardless of `rem`.)
3. Per seq, the work items partition `[0, tiles)` exactly once
   (`Σ counts == tiles`, half-open, contiguous).
4. `splits_per_seq[i] ≤ num_kv_splits` so the static reduction buffer
   `[Oaccum, exp_sums, max_logits]` indexing on the GPU is in-bounds.

Properties 1 and 4 are exactly what `ReduceSplitK` needs to read only the
slots that were written. Properties 2 and 3 prevent the "phantom split"
problem that motivated the patch.

### Heuristics chosen

- `target_tiles_per_wg = max(4, total_tiles / min_wgs)` keeps the split-K
  overhead amortized (≥ 4 tiles per launched WG) while still oversubscribing
  the Xe-core count 2× when work is plentiful.
- `min_blocks_for_split` mirrors the GPU-side single-split threshold so
  short sequences keep the cheap path.
- Single-split is treated as `is_single_split == true` in the reducer, so
  the output goes directly to slot 0 with no cross-split reduction.

## 4. C++ plumbing

| File | Change |
|---|---|
| `flash_api.cpp` | Schema gains `Tensor? splits_per_seq, Tensor? work_list`. Forwarded to both prefill-opt and decode paths of `cutlass_paged_decode_interface`. |
| `attn_interface.{h,cpp}` | Adds the two optional tensors and forwards them to `cutlass_paged_decode_xe2`. |
| `paged_decode_xe2.{h,cpp}` | Wires the tensors into `paged_decode_args_t`: `args.splits_per_seq`, `args.work_list`, `args.total_wgs`. Both default to null/0 (legacy behavior). |
| `paged_decode.hpp` | `paged_decode_args_t` gains `splits_per_seq`, `work_list`, `total_wgs`. `DecodeKernelLauncher::run` patches the scheduler params *after* `to_underlying_arguments` so the compact-grid path is opt-in per launch. |
| `chunk_prefill_scheduler.hpp` | Adds `DecodeWorkItem` POD struct. `DecodeTileScheduler::Params` gains `work_list`, `total_wgs`. `to_underlying_arguments` shrinks `grid.z` to `total_wgs × heads_kv` when work-list is present. `get_block_coord()` returns a 7-tuple `(blk_q, blk_v, head, idx_b, idx_kv_split, wl_tile_start, wl_tile_count)`; the last two fields are `-1` in legacy mode so the FMHA kernel can branch. |
| `paged_decode_kernel.hpp` | `XeFMHAFwdSplitKVKernel` reads the 7-tuple, branches on `wl_tile_start >= 0`. Compact branch uses the precomputed range. Legacy branch keeps the original heuristic. `ReduceSplitK` reads `splits_per_seq[idx_b]` when provided and trusts it; otherwise mirrors the legacy heuristic. |

The legacy branch is intentionally preserved so:
- existing callers that do not provide `host_kv_lens` keep working,
- prefill (where `max_seqlen_q > 1`) never hits the new path.

## 5. Python interface

`flash_attn_varlen_func` accepts a new optional `host_kv_lens` tensor (or
list). When supplied, it is converted once to an `int32` device tensor and
used as `seqused_k`. The decode gate is:

```python
block_table is not None
and host_kv_lens is not None
and num_splits_kv is not None and num_splits_kv > 1
and max_seqlen_q == 1
```

Only inside this gate do we build the plan, and we only upload when
`work_list_cpu.numel() > 0`. Everywhere else the schema-default `None`
tensors are forwarded and the kernel takes the legacy branch.

## 6. Backward compatibility

- `varlen_fwd` schema gained two trailing `Tensor?` arguments; both default
  to `None`. Callers built against the old signature must be re-bound (this
  is a kernel package; pinning is expected).
- `paged_decode_args_t` defaults `splits_per_seq=nullptr`, `work_list=nullptr`,
  `total_wgs=0`. Any C++ caller that does not opt in gets the original grid.
- The FMHA kernel and the reducer both keep their original code paths under
  `wl_tile_start < 0` / `splits_per_seq == nullptr`.
- `compute_splits_per_seq` (legacy entry point) is kept as a thin shim over
  `build_decode_split_plan`.

## 7. Code review summary

### Correctness

- **Reducer / FMHA consistency.** Previously the reducer reapplied the
  `is_single_split` heuristic, which could diverge from the kernel when the
  host had already decided to split. Fix: when `splits_per_seq != nullptr`,
  the reducer trusts it; otherwise it preserves the legacy mirror logic.
- **Even partition guarantees `count ≥ 1`.** The cap `n_splits = min(n_splits,
  num_kv_splits, n_tiles)` plus `base, rem = divmod(n_tiles, n_splits)`
  ensures no empty work item is emitted. Pre-cap code could emit a trailing
  zero-count slot.
- **`is_single_split` scope fix.** In the FMHA kernel, the variable used to
  be declared inside the legacy `else` block but referenced by the epilogue
  call after the block. The work-list path would have left it undeclared.
  Now it is declared once at outer scope and set in both branches.
- **`cutlass_paged_decode_xe2` forwards `work_list`.** Previously the wrapper
  swallowed `work_list`, so the compact-grid path was effectively dead code.
  Now both `splits_per_seq` and `work_list` are forwarded.

### Robustness

- `_infer_num_xe_cores` quotes all `getattr` keys (`"gpu_slices"`,
  `"num_slices"`, `"slices"`, `"gpu_subslices_per_slice"`,
  `"num_subslices_per_slice"`, `"subslices_per_slice"`) and wraps the
  device-property query in `try/except` with a fallback to 20.
- The Python builder accepts a `list`, `tuple` or `torch.Tensor` for
  `kv_lens`, and converts internally without forcing device sync (it pulls
  the tensor to CPU only).

### Style / lint

- The Python file is `pycodestyle`-clean.
- `benchmark_cutlass_flash_attn_decode.py` was reformatted to respect
  line-length limits; configs are built with a single `_mk_cfg` helper to
  cut duplication. One pre-existing 84-char line remains and is not in
  scope of this PR.

### Things deliberately left out

- No GPU-side fallback computation: if Python decides splits, the kernel
  trusts them. The legacy branch is preserved verbatim for callers that do
  not opt in.
- The compact grid patches `scheduler.grid.z` after
  `to_underlying_arguments`; a cleaner solution would be to thread
  `work_list` through `to_underlying_arguments`, but that would require
  touching the upstream CUTLASS-SYCL adapter. The post-patch is local and
  reversible.

## 8. Performance

See `benchmark/benchmark_cutlass_flash_attn_decode.py` and the CSVs in
`benchmark/` for end-to-end and kernel-level numbers on mixed-length
batches. The wins concentrate on configs where a single long sequence
shared a batch with short sequences (`8xskewed`, `8xmixed`,
`16xrealistic_mixed`).
