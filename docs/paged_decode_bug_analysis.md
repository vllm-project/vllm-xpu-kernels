# Paged Decode: Three Latent Bugs, Illustrated

Companion to [`paged_decode_softmax_lse.md`](./paged_decode_softmax_lse.md).
That document describes *what* was added; this one explains *why the three
pre-existing bugs existed*, using diagrams of the work decomposition inside
the decode kernels.

All three were found while validating `softmax_lse`. Two of them corrupted the
attention **output**, not just the LSE, so they were real correctness bugs that
happened to be invisible to the test suite as it stood.

| # | Bug | Blast radius | Visible before? |
| --- | --- | --- | --- |
| 1 | Attention sink folded in with the wrong row index | Per-split/final LSE, and `O` under the old split contract | No — needs sink + multi-split + `head_group_q > 2` |
| 2 | `ReduceSplitK` discarded live splits under the host plan | `O` and `softmax_lse` silently lose one split's mass | No — needs the compact grid |
| 3 | `build_decode_split_plan` ignored the sliding window | 62% of output elements wrong | No — needs compact grid **and** a window |

---

## 1. The decode execution model

Paged decode runs **one or two** kernels:

```text
                     num_kv_splits <= 1                num_kv_splits > 1
                   ------------------------          ---------------------------
  kernel 1         XeFMHAFwdSplitKVKernel            XeFMHAFwdSplitKVKernel
                   writes O directly                 writes normalized Oaccum
                                                     + one LSE per split
                                                                |
  kernel 2                 (none)                     ReduceSplitK
                                                     combines splits -> O
```

Whoever runs *last* owns the `softmax_lse` write. Bugs 2 and 3 live in the
split-K path; bug 1 lives in the epilogue shared by both.

### 1.1 Inside one work-group of the FMHA kernel

For the `q8_h64_p64` instantiation (the shapes were extracted from the
compiler via a `XDumpProbe` template, and match the source formulas):

```text
TileShapeO   = (8, 64)   # WG output tile: 8 packed-GQA rows x 64 head dims
SGTileShapeA = (8, 64)   # each subgroup's private accumulator (FragA coshape)
FragS        = (8, 16)   # each subgroup's scores: 8 rows x its 16 keys
ReduceK      = 4         # subgroups splitting the K dimension
SGPerWG      = 4         # subgroups per work-group  -> 64 work-items
ReduceSGQ    = gcd(8, 4) = 4
ReduceSGV    = ReduceK / ReduceSGQ = 1
SGTileShapeO = shape_div((8,64), (4,1)) = (2, 64)   # ReduceFragA coshape
```

So: **4 subgroups x 16 lanes = 64 work-items**, and the 8-row output tile is
owned 2 rows per subgroup after the reduction.
[Appendix A](#appendix-a--where-4-x-16--64-and-2-rows-per-subgroup-come-from)
derives both numbers from the policy and walks one subgroup's dataflow.

The subgroups change job halfway through the epilogue. This role switch is the
root of bug 1.

#### Phase A — mainloop: subgroups split **K**

```text
   KV tile columns handled by this work-group
   |<---------------- 4 * (K/4) ---------------->|

   +--------+--------+--------+--------+
   | k_blk0 | k_blk1 | k_blk2 | k_blk3 |
   +--------+--------+--------+--------+
      SG0      SG1      SG2      SG3

   Each SG produces, for ALL 8 q-rows:
      tArA   : 8 x 64  partial (unnormalized) O
      tA_max : 8       per-row running max   (log2 domain)
      tA_sum : 8       per-row running sum   (log2 domain)
```

Every subgroup touches every q-row, but only a quarter of the keys.

#### Phase B — `reduce_A`: subgroups switch to splitting **Q**

```text
   SLM staging (a_data / a_max_data / a_sum_data)

            written by                     read by
   SG0 ->  [rows 0..7, k=0]  \
   SG1 ->  [rows 0..7, k=1]   >  barrier  ->  SG0 reads rows 0..1, all k
   SG2 ->  [rows 0..7, k=2]   /              SG1 reads rows 2..3, all k
   SG3 ->  [rows 0..7, k=3]  /               SG2 reads rows 4..5, all k
                                             SG3 reads rows 6..7, all k

   Per owner subgroup, for its 2 rows:
     rA_max = max_k  tA_max[k]
     scale_k = exp2(tA_max[k] - rA_max)
     rA_sum = sum_k  tA_sum[k] * scale_k
     rA     = sum_k  tArA[k]   * scale_k
```

After `reduce_A`, subgroup `i` is the **owner** of output rows `2i` and
`2i+1`. That is the mapping the global `O` store uses.

#### Phase C — epilogue writes

`reduce_A` returns three different objects. Keep them separate:

| Value | Meaning for one KV split | Logical subgroup shape | Mutated in Phase C? |
| --- | --- | --- | --- |
| `rA` | Unnormalized numerator, `sum_j exp(score_j - max) * V_j` | `(2,64)` | Yes — every split is normalized |
| `rA_max` | Maximum score used as the log2-domain origin | 2 rows | No |
| `rA_sum` | Denominator, `sum_j exp(score_j - max)` | 2 rows | Yes — sink is added, then it is inverted |

The three values have related row coordinates but different dataflows.

##### Phase C.1 — `rA`: output numerator only

`rA` has no statistics-store path. Its only destination is `O` or `Oaccum`:

```text
  rA from reduce_A
  logical shape per owner SG: (2 rows, 64 V dimensions)
                         |
             rA *= broadcast(1/rA_sum)
              normalize this split's O_i
                         |
               reorder(rA, tOrO)
            MMA layout -> copy layout
                         |
               copy(tOrO, tOgO)
                         |
          final O or Oaccum[split]
```

All 16 lanes in an owner subgroup participate. The subgroup owns two rows:

```text
  SG0 -> rows 0,1    SG1 -> rows 2,3
  SG2 -> rows 4,5    SG3 -> rows 6,7

  lane:       0 1 2 3 4 5 ... 14 15
  local row:  0 1 0 1 0 1 ...  0  1
```

Each lane owns eight of the subgroup's 128 `rA` elements, so the 16 lanes
together hold the complete `(2,64)` tile. `broadcast<0>` supplies the correct
row denominator to each element:

```cpp
for (int i = 0; i < rA.size(); ++i)
  rA(i) *= broadcast<0>(rA_sum, rA, i);
```

For example, in SG1, every `rA` element belonging to row 2 receives the
reciprocal carried by SG1.L0, while every row-3 element receives the reciprocal
carried by SG1.L1.

##### Phase C.2 — `rA_max`: read-only scale origin

`rA_max` is never modified in Phase C. It is read to express the sink relative
to the split maximum and to form this split's natural-log LSE:

```text
                           rA_max
                              |
           +------------------+------------------+
           |                                     |
  output-denominator sink             statistics-copy sink
  exp2(sink[row_i]*log2e-max)          exp2(sink[q_row]*log2e-max)
           |                                     |
   rA_sum for normalizing O_i           stats_sum for computing LSE_i
```

Its uses are:

```cpp
// Convert a natural-log sink score to the mainloop's log2 domain, then
// express it relative to this split's maximum.
exp2(tSink(row) * log2e - rA_max(0));

row_lse = rA_max(0) * ln2 + log(stats_sum);
```

The epilogue stores `row_lse` in `softmax_lse_accum` when a reducer follows,
or directly in the user-visible `softmax_lse` tensor otherwise.

##### Phase C.3 — `rA_sum`: the value that must fork

`rA_sum` is the only one of the three values with **two mutable meanings**:

```text
                         rA_sum(0)
               pre-sink denominator register
                              |
                   stats_sum = rA_sum(0)
                              |
              +---------------+---------------+
              |                               |
       output copy: rA_sum            statistics copy: stats_sum
              |                               |
  add sink using output row_i         add sink using statistics q_row
              |                               |
  invert and broadcast into rA        compute and write LSE_i
              |                               |
  normalized partial O_i              softmax_lse_accum[split]
              |                       ReduceSplitK if needed
              v
          O / Oaccum
```

The fork is explicit in the fixed code:

```cpp
ElementA stats_sum = rA_sum(0);

// Statistics copy: work-group-linear row identity.
if (row_valid && idx_kv_split == 0)
  stats_sum += exp2(tSink(q_row) * log2e - rA_max(0));

// Output copy: subgroup-copy row identity.
if (active && idx_kv_split == 0) {
  int row_i = base_row + lane % 2;
  rA_sum(0) += exp2(tSink(row_i) * log2e - rA_max(0));
}
```

###### The `rA_sum` output-copy row map

For output normalization, each subgroup uses its two logical row carriers:

```text
                 output rows          canonical rA_sum carriers
  SG0 (wi 0..15)     0,1             SG0.L0 -> row0, SG0.L1 -> row1
  SG1 (wi16..31)     2,3             SG1.L0 -> row2, SG1.L1 -> row3
  SG2 (wi32..47)     4,5             SG2.L0 -> row4, SG2.L1 -> row5
  SG3 (wi48..63)     6,7             SG3.L0 -> row6, SG3.L1 -> row7
```

The output row is derived from the block-copy partition:

```cpp
int base_row = get<0>(tOgO(_0{}, _0{}, _0{}));
int row_i = base_row + lane % 2;
```

That map is correct for adding the sink to the denominator that will normalize
the same row's `rA`.

###### The `stats_sum` statistics-copy row map

The physical SLM load is wider than the logical two-row `ReduceFragARow`.
`copy_block_s2r` uses a 16-lane `XE_1D_LDSM`; every lane receives one
consecutive float:

```text
  SG0 register after reducing the four K contributors

  SG0 lane:  L0  L1  L2  L3  L4  L5  L6  L7 | L8 ... L15
  value:     D0  D1  D2  D3  D4  D5  D6  D7 | padding
  q_row:      0   1   2   3   4   5   6   7

  Dr = complete pre-sink denominator for q-row r in this KV split
```

Thus SG0 lanes 0..7 provide an eight-row side channel:

```cpp
q_row = get<0>(blk_qv) * 8 + thr_id;
row_valid = thr_id < 8 && q_row < head_group_q;
```

They combine `stats_sum` with `rA_max` and write one natural-log
`softmax_lse_accum(q_row, split)`. SG1, SG2 and SG3 do not write statistics,
even though their lanes 0 and 1 carry the canonical denominators used by their
own `rA` output tiles.

##### Phase C.4 — where the three values rejoin

The exact action depends on whether a second kernel follows:

| Runtime case | `rA` | `rA_max` | `rA_sum` / `stats_sum` |
| --- | --- | --- | --- |
| No reducer (`num_kv_splits <= 1`) | Normalize and store final `O` | Used directly in LSE | `rA_sum` normalizes `rA`; `stats_sum` completes LSE |
| Reducer launched, one effective split | Normalize and store `Oaccum[0]` | Forms `LSE_0` | `rA_sum` normalizes `rA`; `stats_sum` completes `LSE_0`; weight is naturally 1 |
| True multi-split | Normalize and store `Oaccum[split]` | Forms `LSE_i` | `rA_sum` normalizes `rA`; `stats_sum` completes and stores `LSE_i` |

For true multi-split, `ReduceSplitK` combines two stored streams:

```text
  normalized rA_i               -> Oaccum_i
  rA_max_i*ln2 + ln(stats_sum_i)-> softmax_lse_accum_i

  LSE_global = logsumexp_i(LSE_i)
  weight_i   = exp(LSE_i - LSE_global)

  O = sum_i(weight_i * Oaccum_i)
```

The bug was specifically in the **`rA_sum` fork**. Before the fix,
`rA_sum(0)` was first mutated with `sink[row_i]` under the output-copy map and
then reused as though it contained `sink[q_row]` under the statistics map.
`rA` itself was not mis-addressed, and `rA_max` was not corrupted.

---

## 2. Bug 1 — attention sink folded in at the wrong row

### 2.1 What the code did

This is the pre-refactor two-buffer code in which split outputs were
unnormalized and the reducer consumed `exp_sums` plus `max_logits`:

```cpp
auto [rA, rA_max, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

if constexpr (Sink) {
  if (active && idx_kv_split == 0) {
    int base_row = get<0>(tOgO(_0{}, _0{}, _0{}));          // scheme (a)
    int lane     = get_sub_group().get_local_id()[0];
    int row_i    = base_row + (lane % size<0>(SGTileShapeO{}));  // + lane % 2
    if (row_i < head_group_q)
      rA_sum(0) += exp2(tSink(row_i) * log2e - rA_max(0));   // mutates rA_sum!
  }
}

if (row_valid) {                                             // scheme (b)
  exp_sums  (q_row, idx_kv_split) = rA_sum(0);               // <-- poisoned
  max_logits(q_row, idx_kv_split) = rA_max(0);
}
```

The sink is folded into `rA_sum(0)` using **scheme (a)**, because that is what
the `O` normalization needs. The statistics block then reads the very same
register under **scheme (b)**.

### 2.2 The observed corruption

A discriminating probe — random sinks are useless here, they hide index bugs —
sets `sink[h] = 20 + (h % head_group_q)` so the sink dominates every score and
`round(lse - 20)` reads back *the index the kernel actually used*:

```text
  head_group_q = 4,  8 query heads / 2 KV heads

  q_row :   0     1     2     3
  wanted : s[0]  s[1]  s[2]  s[3]
  got    : s[0]  s[1]  s[0]  s[1]        <-- sink[q_row % 2]
            OK    OK   WRONG WRONG

  head  :   0     1     2     3     4     5     6     7
  result:   OK    OK   FAIL  FAIL   OK    OK   FAIL  FAIL
```

The `% 2` is exactly `size<0>(SGTileShapeO{}) == 2` leaking out of scheme (a).

Independent confirmation that the *base* statistics were fine all along
(no sink, forced multi-split, `head_group_q = 4`, `kv_len = 4096`):

```text
  h=0 lse=8.75538 ref=8.75538   h=4 lse=8.84631 ref=8.84631
  h=1 lse=8.64682 ref=8.64682   h=5 lse=8.95448 ref=8.95448
  h=2 lse=8.85288 ref=8.85288   h=6 lse=8.76537 ref=8.76537
  h=3 lse=8.71882 ref=8.71882   h=7 lse=8.71132 ref=8.71132
```

All eight rows distinct and exact — so `rA_sum(0)`/`rA_max(0)` at work-item
`thr_id` really do belong to row `thr_id`. Only the sink addend was misplaced.

### 2.3 Why it also corrupted the output

Under the old two-buffer contract, **single-split** statistics were sentinels
and `O` was normalized in-epilogue, so a wrong `exp_sums` was harmless. In
**multi-split** mode `ReduceSplitK` divided by `sum_i exp_sums[i]`, so a
poisoned denominator scaled `O` directly:

```text
  head 2, multi-split, with sink:   got 0.1074   expected 0.0537   (2.0x)
```

### 2.4 The fix

Snapshot the denominator *before* the sink block and re-apply the sink under
scheme (b) for the statistics consumer only:

```cpp
ElementA stats_sum = rA_sum(0);                 // pre-sink snapshot

if constexpr (Sink) {
  if (row_valid && idx_kv_split == 0)           // scheme (b) - statistics
    stats_sum += exp2(tSink(q_row) * log2e - rA_max(0));
  if (active && idx_kv_split == 0) { ... }      // scheme (a) - O, untouched
}
```

Each consumer now gets a value addressed in its own scheme. The later
CUDA-style refactor retained this fork but replaced the two statistics buffers
with one per-split LSE and normalized every partial output.

---

## 3. Bug 2 — `ReduceSplitK` discarded live splits

### 3.1 Two different partitions of the same tiles

`ReduceSplitK` must know which of the `num_kv_splits` slots actually hold data.
It inferred that from a formula:

```cpp
int num_blocks_per_split = ceil_div(windowed_k_blocks, seq_num_kv_splits);
// "slot i is empty if it starts past the end"
if (i * num_blocks_per_split >= windowed_k_blocks) break;
```

That is self-consistent with the **legacy on-device** partition, which uses the
same `ceil_div`. It is *not* consistent with the **host plan**, which balances
tiles as `base` / `base + 1`.

```text
  33 tiles over 8 splits

  legacy  (ceil_div(33,8) = 5 tiles each):
    s0[0:5] s1[5:10] s2[10:15] s3[15:20] s4[20:25] s5[25:30] s6[30:33] s7[--]
                                                                       ^ genuinely empty
    guard: 7*5 = 35 >= 33  -> drop s7   CORRECT

  host plan (base=4, rem=1):
    s0[0:5] s1[5:9] s2[9:13] s3[13:17] s4[17:21] s5[21:25] s6[25:29] s7[29:33]
                                                                     ^^^^^^^^ REAL WORK
    guard: 7*5 = 35 >= 33  -> drop s7   WRONG
```

The host partition is strictly *finer*, so the coarse guard walks off the end
early and throws away a split that really did produce output — silently losing
its probability mass from both `O` and `softmax_lse`.

### 3.2 The fix

The host plan already guarantees every emitted split owns at least one tile, so
in compact-grid mode the guard is not needed at all:

```cpp
const bool plan_driven = (p.splits_per_seq != nullptr);
...
if (!plan_driven && i * num_blocks_per_split >= windowed_k_blocks) break;
```

Applied at all three sites: the SLM fill, the LSE accumulation loop, and the
output accumulation loop.

---

## 4. Bug 3 — the split plan ignored the sliding window

### 4.1 The offset mismatch

For decode, every packed GQA row sits at position `kv_len - 1`, so a left
window of `W` makes all tiles before `k_block0` unreachable:

```cpp
k_block0 = max(kv_len - 1 - W, 0) / kv_tile;
```

The kernel offsets every work item by that value:

```cpp
kv_split_offset = k_block0 + wl_tile_start;   // tile_start is window-relative
```

But the host planner partitioned the **full** tile count. The two disagreed:

```text
  kv_len = 4096, kv_tile = 64, window_size_left = 127

  total tiles  = ceil(4096/64)             = 64
  k_block0     = (4096-1-127)/64 = 3968/64 = 62
  live tiles   = 64 - 62                   =  2

  tile index:  0 ....................... 61 | 62  63 |
               |<-- outside the window -->|  |<-live->|

  planner (window-blind), 8 splits of 8 tiles:
      s0[0:8] s1[8:16] ... s7[56:64]

  kernel reads split i at  62 + tile_start:
      s0 -> tiles 62..69      (2 live + 6 past the end)
      s1 -> tiles 70..77      (entirely past the end)
      ...
      s7 -> tiles 118..125    (entirely past the end)
```

Result: 62.6% of output elements mismatched.

### 4.2 The fix

Teach the planner the same `k_block0`, and partition only the live tiles:

```python
def _windowed_tiles(kv: int) -> int:
    total = (kv + kv_tile - 1) // kv_tile
    if window_size_left is not None and window_size_left >= 0:
        k_block0 = max(kv - 1 - window_size_left, 0) // kv_tile
        total -= k_block0
    return max(1, total)
```

The call site normalizes `-1 -> max_seqlen_k` exactly the way `flash_api.cpp`
does, so host and device compute an identical `k_block0`. With
`window_size_left = -1` the helper reduces to the original formula, so the
non-window path is bit-identical to before.

---

## 5. Why the test suite never caught any of this

The compact-grid split plan is reached only behind a four-way gate in
`flash_attn_interface.py`:

```python
if (block_table is not None and host_kv_lens is not None
        and num_splits_kv is not None and num_splits_kv > 1
        and max_seqlen_q == 1):
```

`host_kv_lens` is the discriminator, and repo-wide it appeared in exactly two
places before this change: the library itself and `benchmark/src/`.

| Caller | Builds a plan? | Uses a window? | Uses a sink? |
| --- | --- | --- | --- |
| Existing decode tests | No — pass `seqused_k` | Yes, `(127, -1)` | Yes, random |
| `benchmark/` decode script | Yes | No — `window_size=(-1, -1)` | No |
| New `test_decode_with_paged_kv_softmax_lse` | Yes | Yes | Yes |

- **Bug 3** needs *plan* x *window*. No caller combined them.
- **Bug 2** needs the plan at all. Only the benchmark reached it, and a
  benchmark does not check numerics.
- **Bug 1** needs *sink* x *multi-split* x `head_group_q > 2` x a
  non-random sink. Random sinks over long KV perturb the result by ~1e-4,
  far under the 1e-2 tolerance.

The feature was introduced by commit `a59a3d4` *"Group splitkv for mixed batch
decoding (#337)"*, which touched 14 files — `benchmark/`, `csrc/`, `docs/`,
`vllm_xpu_kernels/` — and **zero test files**.

---

## 6. Regression coverage added

- `test_decode_with_paged_kv_softmax_lse` — 312 cases; parametrizes
  `use_split_plan` (legacy grid vs. host plan) crossed with sliding windows and
  sinks. This is the first coverage the compact-grid path has ever had, and it
  catches bugs 2 and 3.
- `test_decode_paged_kv_sink_row_mapping` — 16 cases; uses dominant sinks
  (`sink[h] = 20 + h`) so `lse - 20` reads back the row index the kernel used.
  This is the only construction that can detect bug 1; random sinks cannot.

Full suite after the fixes: **3903 passed, 1216 skipped, 0 failed**.

---

## 7. Techniques worth reusing

**Reading CuTe layout constants out of the compiler.** Declare, never define,
`template <int...> struct XDumpProbe;`, then instantiate it as a member of the
class you care about and read the values out of the error message:

```cpp
template <int...> struct XDumpProbe;
XDumpProbe<(int)cute::size<0>(TileShapeO{}),
           (int)cute::size<0>(SGTileShapeO{}),
           (int)ReduceK{}, (int)SGPerWG{}> probe_dummy_;
```

For **fragments** the plain `cute::size(Frag{})` form does not compile
(`C-style cast from 'void' to 'int' is not allowed`). Two forms that do work:

```cpp
// logical tile a fragment represents, per subgroup
(int)cute::size<0>(cute::atuple_coshape(FragA{}.tv_layout())),
(int)cute::size<1>(cute::atuple_coshape(FragA{}.tv_layout())),
// element count per work-item
(int)decltype(cute::size(FragARow{}))::value,
```

The error message also prints the fully-substituted enclosing type, which is a
free readout of the MMA atom, the tiled-MMA layouts and the tile shapes — no
extra probe entries needed for those.

Build one translation unit rather than the whole extension (~1 min vs ~30 min):

```bash
xpu_build ninja -C build/temp \
  csrc/xpu/attn/xe_2/CMakeFiles/attn_kernels_xe_2.dir/paged_decode_kernel_template_q8_h64_p64_ttt.cpp.o
```

The probe build always fails, so it never overwrites the object file and the
existing `.so` stays usable — but restore the header afterwards and re-run the
tests to confirm the tree is back to a known-good state.

**Discriminating probes beat random inputs for index bugs.** A random sink
makes a wrong row index look like noise. A sink that dominates every score
turns the LSE into a direct readout of the index, converting a 1e-4 numerical
smell into an exact integer mismatch.

**Ask which consumer owns the addressing.** Bug 1 exists because one register
served two consumers with two different row mappings. When a value is shared
between a data path (`O`) and a metadata path (statistics), check that both
address it the same way — or give each its own copy, which is what the fix does.

---

## Appendix A — where "4 x 16 = 64" and "2 rows per subgroup" come from

Everything below is derived from `decode_policy_qpacked_head<_8, _64, _64>`
(`csrc/xpu/attn/xe_2/fmha_utils.hpp`), i.e. `q_packed = 8`, `head_dim = 64`,
`kv_tile = 64` — the instantiation selected for `block_size = 64`,
`head_size = 64`, `head_group_q <= 8`.

### A.1 The policy

```cpp
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _64> {
  using ShapeQK          = Shape<q_packed, _64, _64>;   // (M=8q, N=64kv, K=64d)
  using ShapePV          = Shape<q_packed, _32, _64>;   // (M=8q, N=32d, K=64kv)
  using ShapeOut         = Shape<q_packed, head_dim>;   // (8, 64)
  using SubgroupLayoutQK = Layout<Shape<_1, _4, _1>>;   // (M, N, K) splits
};
```

`SubgroupLayoutQK = (1, 4, 1)` is the whole story for the subgroup count:
the work-group is **not** split over the query dimension (M) or the head
dimension (K), only over **KV (N), 4 ways**.

```text
SGPerWG = product(take<1,4>(shape(ThrLayoutVMNK)))
        = M_split * N_split * K_split
        = 1 * 4 * 1
        = 4 subgroups
```

On Xe2 a subgroup is `intel::sg_size = 16` lanes, so:

```text
work-group size = SGPerWG * sg_size = 4 * 16 = 64 work-items
```

That is the `4 x 16 = 64`. `thr_id = ThreadIdxX()` runs `0..63`, and
`sub_group_id = thr_id / 16`.

### A.2 The QK and PV subgroup tiles

Dividing each work-group tile by its subgroup layout:

```text
TileShapeQK = (8, 64, 64)   /  (1, 4, 1)  ->  SG QK tile = (8, 16, 64)
                                                    8 q-rows
                                                   16 keys      <- its slice
                                                   64 head dims

SubgroupLayoutPV = get_sg_layout_pv((1,4,1))
                 = (get<0>, _1, get<1>) = (1, 1, 4)
TileShapePV = (8, 32, 64)   /  (1, 1, 4)  ->  SG PV tile = (8, 32, 16)
                                                    8 q-rows
                                                   32 v-dims
                                                   16 keys      <- same slice
```

Note the transposition of roles: the KV dimension is the **N** (output) mode of
QK but the **K** (contraction) mode of PV. Splitting QK over N therefore
becomes splitting PV over K — which is precisely why a cross-subgroup reduction
is needed at all:

```text
ReduceK = size<3>(TiledMMAPV::ThrLayoutVMNK) = K_split of PV = 4
```

`VTiles = get<1>(ShapeOut) / get<1>(ShapePV) = 64 / 32 = 2`, so each subgroup
runs the PV MMA twice to cover the full 64-wide head dimension, giving

```text
SGTileShapeA = atuple_coshape(FragA{}.tv_layout()) = (8, 64)
```

Every subgroup holds a **complete 8 x 64 partial output** — all q-rows, all
head dims — for its own 16 keys.

### A.3 Why 2 rows per subgroup after the reduction

Four subgroups each hold a full `8 x 64` partial result. The reduction has to
be parallelized somehow, and the code parallelizes it over the **q** dimension:

```cpp
using ReduceSGQ     = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
                    // gcd(8, 4) = 4
using ReduceSGV     = /* v_avail_sg = ReduceK / ReduceSGQ = 4/4 = 1 */;
using ReduceSGLayout = Shape<ReduceSGQ, ReduceSGV>;      // (4, 1)
using SGTileShapeO  = shape_div(take<0,2>(SGTileShapeA{}),
                                shape(ReduceSGLayout{}));
                    // shape_div((8, 64), (4, 1)) = (2, 64)
```

`ReduceSGQ = gcd(8, 4) = 4` says: use 4 reduction destinations over the 8
q-rows. `SGTileShapeO = (2, 64)` is the consequence — **2 rows x 64 dims per
subgroup**. `ReduceSGV = 1` means the head dimension is not split further, so
each owner subgroup handles all 64 dims of its 2 rows.

The SLM addressing in `reduce_A` makes the ownership explicit:

```cpp
auto shape_A_row  = make_shape(get<0>(SGTileShapeO{}),   // 2   (row within blk)
                               shape(ReduceSGLayout{}),  // 4   (row block)
                               ReduceK{},                // 4   (contributor)
                               SGPerWG{} / ReduceK{});   // 1   (a_tile)
auto sA_row_stride = make_stride(_1{},                       // row  -> +1
                                 make_stride(_2{}, _0{}),    // block-> +2
                                 AlignedSGTileA_Q{},         // k    -> +16
                                 AlignedSGTileA_Q{} * ReduceK{});
...
copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
//                     ^^^^^ the row block a subgroup reads is its own k_blk
```

Row block `j` covers rows `{2j, 2j+1}` (stride 2, extent 2). Subgroup `k_blk`
reads row block `k_blk`. Hence:

```text
  SG0 -> rows 0,1     SG1 -> rows 2,3     SG2 -> rows 4,5     SG3 -> rows 6,7
```

`AlignedSGTileA_Q = ceil(8/16)*16 = 16` is padding so each contributor's row
vector starts on a 16-element boundary (a CuTe block-load constraint), which is
why the `k` stride is 16 and not 8.

### A.4 Shape reference

Values marked **[probed]** were read out of the compiler with the
`XDumpProbe` trick from [section 7](#7-techniques-worth-reusing); the probe for
`decode_policy_q8_h64_p64` returned `XDumpProbe<8, 16, 1, 2, 64, 1>`. The rest
follow from the policy or from the source layouts.

`coshape` is the logical tile a fragment represents **per subgroup**; the
per-work-item column is that divided by `sg_size = 16`.

| Tensor | Lives in | Logical shape (per subgroup) | Per work-item |
| --- | --- | --- | --- |
| `Q` tile | global -> reg | `8 x 64` bf16 | block 2D load |
| `K` tile | global -> reg | `64 x 16` bf16 (its 16 keys) | block 2D load |
| `V` tile | global -> reg | `16 x 64` bf16 (its 16 keys) | block 2D load |
| `FragS` (`tSrS`) | register | `(8, 16)` = q x keys **[probed]** | 8 floats |
| `FragA` (`tArA`) | register | `(8, 64)` = `SGTileShapeA` | 32 floats |
| `FragARow` (`tA_max`, `tA_sum`) | register | 8 rows | **1 float [probed]** |
| `sA` | SLM | `(q=2, v=64, rblk_dst=(4,1), rblk_src=4, a_tile=1)` | 2048 floats total |
| `sA_max`, `sA_sum` | SLM | `(q=2, rblk_dst=(4,1), rblk_src=4, a_tile=1)` | 64 floats total each |
| `ReduceFragA` (`rA`) | register | `(2, 64)` = `SGTileShapeO` **[probed]** | 8 floats |
| `ReduceFragARow` (`rA_max`, `rA_sum`) | register | 2 rows | **1 float [probed]** |
| `tOrO` / `tOgO` | register / global | `2 x 64` copy fragment | 8 floats |

Three entries deserve comment.

**`FragS` is `(8, 16)`, not `(8, 64)`.** The 16 is this subgroup's share of the
64-key tile — direct confirmation of the `(1, 4, 1)` split. The compiler also
confirms the atom and the tiles in the same message:

```text
MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>
TiledMMAQK : Layout<(1, 4, 1)>              tile (8, 64, 64)
TiledMMAPV : Layout<(1, 1, 4), (0, 0, 1)>   tile (8, 32, 64)
VTiles     : 2
```

**`ReduceFragA` is `(2, 64)`.** It is built as

```cpp
using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
    make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
```

`select<1,0>` writes the modes as `(v, q)` in the layout, but the coshape is
`(q=2, v=64)` — i.e. exactly `SGTileShapeO`. This is the 2-rows-per-subgroup
result of the reduction, in register form.

**Both row fragments hold exactly ONE float per work-item.** `FragARow` and
`ReduceFragARow` probe as size 1, so `tA_sum(0)` and `rA_sum(0)` are not
"row 0 of a vector" — they are *this work-item's* row. That is why the two
row-index expressions in the epilogue can both be applied to the same
register, and why disagreeing about which row a work-item stands for silently
mis-attributes a value instead of producing an out-of-range access.

### A.5 One subgroup's dataflow, end to end

Following **SG1** (`k_blk = 1`) through one KV tile of 64 keys. It is
responsible for keys `[16, 32)` during the mainloop and for output rows `{2,3}`
during the reduction.

```text
 ============================ PHASE A : MAINLOOP =========================
 role: SG1 owns KEYS 16..31, and ALL 8 q-rows.

   global loads (block 2D, subgroup-cooperative)
     Q  : 8 x 64  bf16      (shared by all 4 subgroups)
     K  : 64 x 16 bf16      <- only its 16 keys
     V  : 16 x 64 bf16      <- only its 16 keys

   S = Q . K^T                    DPAS XE_DPAS_TT<8, float, bf16>  (8x16x16)
       (8 x 64) . (64 x 16)  ->  FragS tSrS : coshape (8, 16)   4 K-steps
                                 = 128 floats = 8 per work-item

   softmax (log2 domain; scores pre-scaled by sm_scale * log2(e))
       row_max_new = max over its 16 columns          -> cross-lane reduce
       correction  = exp2(row_max_old - row_max_new)
       P           = exp2(S_sg - row_max_new)
       tA_sum     *= correction ;  tA_sum += rowsum(P)
       tArA       *= correction                       <- rescale old partial O

   A += P . V                     2 MMAs (VTiles = 2), each 8x32x16
       (8 x 16) . (16 x 64)  ->  FragA tArA : coshape (8, 64)
                                 = 512 floats = 32 per work-item

   registers held by SG1 after the KV loop:
       tArA   : FragA    coshape (8, 64)   32 floats/work-item
                         partial, unnormalized O   (its keys only)
       tA_max : FragARow 8 rows             1 float/work-item
       tA_sum : FragARow 8 rows             1 float/work-item

 ====================== PHASE B : reduce_A (SLM exchange) ================
 role switch: SG1 stops being "the keys 16..31 subgroup" and becomes
              "the rows 2..3 subgroup".

   write (every subgroup writes ALL 8 rows, tagged with its own k_blk)

     copy_block_r2s(tA_max, sA_max(_, _, k_blk=1, 0));   //  8 floats
     barrier_arrive(release)
     copy_block_r2s(tA_sum, sA_sum(_, _, k_blk=1, 0));   //  8 floats
     copy_block_r2s(tArA,   sA    (_, _, _, 1, 0));      // 512 floats

     destination slices:
       sA_max(_, _, k_blk, a_tile) : shape (2, (4,1)) = 8 elements,
                                     stride (1, (2,0)) -> offsets 0..7
       sA    (_, _, _, k_blk, a_t) : shape (2, 64, (4,1)) = 512 elements

     tA_max / tA_sum are FragARow = reduce<1>(FragA), i.e. the 64-wide V
     mode of SGTileShapeA = (8, 64) is collapsed -> 8 rows.

     Those 8 floats land in a 16-float slot: AlignedSGTileA_Q =
     ceil(8/16)*16 = 16 is the PADDED STRIDE between consecutive k_blk
     slots (a CuTe block-load alignment constraint), not the write size.
     The upper 8 floats of each slot are never touched.

     SLM footprint (SharedStorageReduceK):
       a_data     : size(SGTileShapeA) * SGPerWG = 8*64*4 = 2048 f = 8192 B
                    logical shape (q=2, v=64, rblk_dst=(4,1), rblk_src=4, 1)
       a_max_data : AlignedSGTileA_Q  * SGPerWG =   16*4  =   64 f =  256 B
                    logical shape (q=2, rblk_dst=(4,1), rblk_src=4, 1)
                    = 32 floats of data + 32 floats of padding
       a_sum_data : same                                            256 B
                                                        total  ~8.5 KB

   barrier_wait(acquire)      <- maxima are now visible
   barrier_arrive(release)

   logical target: rows 2..3 from ALL FOUR contributors
   physical load : 16 consecutive floats starting at row-block 1

     for kr in 0..3:  copy_block_s2r(sA_max(_, k_blk=1, kr, 0), rA_kmax[kr])

           SLM a_max_data                     SG1 lane registers
       +----+----+----+----+                 +----+----+----+----+---+
   kr=0| r0 r1 | r2 r3 | r4 r5 | r6 r7 | -> | r2 | r3 | r4 | r5 |...|
       +----+----+----+----+                 +----+----+----+----+---+
                    ^ load base                L0   L1   L2   L3
                                               ^^^^^^^^^
                                               canonical rows for SG1

   The same happens for kr=1,2,3. Lanes 0 and 1 are the logical
   ReduceFragARow values for SG1's rows 2 and 3. Lanes 2..5 physically
   receive rows 4..7; later lanes read padding or adjacent aligned storage.
   They are not selected by broadcast<0> for SG1's output tile.

   align the four scale factors, then reduce

     rA_max     = max(rA_kmax[0..3])                       # global row max
     scale[kr]  = exp2(rA_kmax[kr] - rA_max)               # re-align exponents

   barrier_wait(acquire)      <- A and A_sum are now visible

     rA_sum = sum_kr  sA_sum[rows 2..3, kr] * scale[kr]
     rA     = sum_kr  sA    [rows 2..3, kr] * scale[kr]

   SG1 now holds the COMPLETE softmax statistics and output for rows 2 and 3:
       rA     : ReduceFragA    coshape (2, 64) = SGTileShapeO   8 floats/wi
       rA_max : ReduceFragARow 2 rows                           1 float/wi
       rA_sum : ReduceFragARow 2 rows                           1 float/wi

 ========================= PHASE C : EPILOGUE WRITES =====================

   See "Phase C - epilogue writes" above for the complete lane table and
   the fork between the output and statistics consumers.

   (a) attention sink, folded into the denominator          [O consumer]
         row_i = base_row + (lane % size<0>(SGTileShapeO))  # SG1 -> rows 2,3
         rA_sum(0) += exp2(sink[row_i]*log2(e) - rA_max(0))

   (b) natural-log LSE for split-K                  [work-items 0..7 only]
         q_row = get<0>(blk_qv) * 8 + thr_id
         row_lse = rA_max(0) * ln2 + log(stats_sum)
         softmax_lse_accum(q_row, split) = row_lse

   (c) output store
         rA *= 1 / rA_sum                            # normalize every split
         reorder(rA, tOrO)                           # (2,64) accumulator
                                                     #   -> (2,64) copy frag
         copy(copy_o, tOrO, tOgO)                    # 2 x 64 block 2D store
```

The `reorder(rA, tOrO)` in step (c) is the tell: the reduction result `rA`
lives in the **MMA accumulator** layout, while `tOrO`/`tOgO` live in the
**block-2D copy** layout, and a shuffle is required between them.

Bug 1 sat exactly on that seam. Step (a) derives its row index from `tOgO`, the
copy-layout coordinate — correct for the value it is about to feed into
step (c), and confirmed correct by a per-row dominant-sink probe:

```text
  sink@row0 -> |O| = [0.000, 0.264, 0.243, 0.328]   zeroed row 0   OK
  sink@row1 -> |O| = [0.188, 0.000, 0.243, 0.328]   zeroed row 1   OK
  sink@row2 -> |O| = [0.188, 0.264, 0.000, 0.328]   zeroed row 2   OK
  sink@row3 -> |O| = [0.188, 0.264, 0.243, 0.000]   zeroed row 3   OK
```

But step (b) then re-read the very same mutated register under the
work-group-linear `q_row` scheme, where the correct index is `thr_id`. The two
expressions agree only for the first `size<0>(SGTileShapeO{}) = 2` rows, which
is why the failure signature was exactly `sink[q_row % 2]`.

The fix does not try to reconcile the layouts — it gives each consumer its own
value (`stats_sum` for (b), `rA_sum` for (a)), so neither has to know about the
other's addressing.
