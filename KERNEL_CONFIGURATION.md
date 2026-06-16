# vLLM-XPU Kernel Configuration Guide

vLLM-XPU selectively compiles attention kernel variants at build time. This
reduces build time and binary size, but means you may need to recompile if your
model requires a kernel combination that was not included in the build.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Kernel Types](#kernel-types)
3. [Configuration Presets](#configuration-presets)
4. [Config File Format](#config-file-format)
5. [How to Determine Your Model's Config](#how-to-determine-your-models-config)
6. [Bool Combinations](#bool-combinations)
7. [Custom Configuration](#custom-configuration)
8. [Build & Install](#build--install)
9. [Troubleshooting](#troubleshooting)
10. [Performance Notes](#performance-notes)

---

## Quick Start

### If you see a kernel missing error:

```
❌ Chunk prefill kernel tuple not compiled for this configuration.
```

**Option A: Use the full preset — builds all kernel variants (~60 min)**

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf \
  pip install .
```

**Option B: Use the default preset — Llama / Qwen / DeepSeek only (~2 min)**

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf \
  pip install .
```

**Option C: Custom config — add only the missing combination**

The error message tells you exactly which line to add. For example:

```
Add this line to your chunk_prefill config file:
  128,true,true,false,false,false
Then rebuild:
  VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf pip install .
```

---

## Kernel Types

vLLM-XPU has two main kernel categories:

### 1. Chunk Prefill (Prompt Processing)

- Used when processing prompt tokens
- Configured via: `VLLM_CHUNK_PREFILL_CONFIG`
- Parameters: `headsize`, `paged`, `causal`, `local`, `sink`, `lse`

### 2. Paged Decode (Token Generation)

- Used when generating tokens one by one
- Configured via: `VLLM_PAGED_DECODE_CONFIG`
- Parameters: `qgroup`, `headsize`, `pagesize`, `causal`, `local`, `sink`

---

## Configuration Presets

Config files are located in `csrc/xpu/attn/kernel_configs/`.

### Chunk Prefill

| File | Kernels | Use Case |
|------|---------|----------|
| `chunk_prefill_full.conf` | 216 | All combinations — supports every model |
| `chunk_prefill_default.conf` | ~13 | Llama, Qwen, DeepSeek MLA, Falcon (default build) |

### Paged Decode

| File | Kernels | Use Case |
|------|---------|----------|
| `paged_decode_full.conf` | 384 | All combinations — supports every model |
| `paged_decode_default.conf` | ~17 | Llama, Qwen, DeepSeek MLA, Falcon (default build) |

### Recommended Config per Model Family

| Model Family | head_size | Chunk Prefill | Paged Decode |
|--------------|-----------|---------------|--------------|
| Llama-2/3, Qwen, Mistral | 128 | `default` | `default` |
| DeepSeek-V2/V3/R1 (MLA) | 192 | `default` | `default` |
| Gemma-2 | 256 | `full` | `full` |
| Mixed / other models | multiple | `full` | `full` |

---

## Config File Format

### Chunk Prefill

```
# Lines starting with # are comments. Empty lines are ignored.
# Use 'all' to build everything (same as chunk_prefill_full.conf).

# Format: headsize,paged,causal,local,sink,lse
128,true,true,false,false,false
128,false,true,false,false,false
128,false,true,false,false,true   # lse=true: only valid when paged=false,local=false,sink=false
192,true,true,false,false,false
```

**Parameters:**
- `headsize` — head dimension: `64`, `96`, `128`, `192`, `256`, or `512`
- `paged` — whether paged KV cache is used
- `causal` — whether causal masking is applied
- `local` — whether sliding window attention is used
- `sink` — whether StreamingLLM attention sinks are used
- `lse` — whether log-sum-exp is output (requires `paged=false`, `local=false`, `sink=false`)

If boolean flags are omitted, all 18 valid combinations are generated for that headsize.

### Paged Decode

```
# Lines starting with # are comments. Empty lines are ignored.
# Use 'all' to build everything (same as paged_decode_full.conf).

# Format: qgroup,headsize,pagesize[,causal,local,sink]
# If causal/local/sink are omitted, all 8 bool combinations are generated.
8,128,16,true,false,false
8,128,32,true,false,false
8,128,64,true,false,false

# Omit bool flags to generate all 8 combinations for this shape:
8,192,16
```

**Parameters:**
- `qgroup` — GQA group bucket (packed-Q tile size): `8` (ratio ≤ 8) or `16`
  (ratio > 8). The decode kernel tiles the GQA head-group across the grid, so
  `qgroup` is a tile size, **not** a hard cap on `num_heads_q / num_heads_kv`.
  Large MQA ratios (e.g. Falcon-7B's 71 query heads / 1 KV head) use the `16`
  bucket and are split into `ceil(ratio / 16)` work-groups.
- `headsize` — head dimension: `64`, `96`, `128`, `192`, `256`, or `512`
- `pagesize` — KV cache block size: `16`, `32`, `64`, or `128`
- `causal` — whether causal masking is used (almost always `true` for decode)
- `local` — whether sliding window attention is used
- `sink` — whether StreamingLLM attention sinks are used

---

## How to Determine Your Model's Config

For a given model you need:
1. **head_size**: `hidden_size / num_attention_heads`
2. **GQA ratio** (decode only): `num_attention_heads / num_key_value_heads` → qgroup `8` (ratio ≤ 8) or `16` (ratio > 8, including large MQA ratios)
3. **page_size** (decode only): your vLLM deployment's `--block-size` (default: 16)

Common model parameters:

| Model | head_size | GQA ratio | qgroup |
|-------|-----------|-----------|--------|
| Llama-3-8B | 128 | 1 (MHA) | 8 |
| Llama-3-70B | 128 | 8 | 8 |
| Qwen2-72B | 128 | 8 | 8 |
| Qwen3-30B-A3B | 128 | 4 | 8 |
| DeepSeek-V3 (MLA) | 128 + 192 | varies | 8 |
| Gemma-2-27B | 256 | 2 | 8 |
| Mistral-7B | 128 | 8 | 8 |
| Falcon-7B (MQA) | 64 | 71 | 16 |

---

## Bool Combinations

### Chunk Prefill: 5 Bool Parameters (`paged`, `causal`, `local`, `sink`, `lse`)

**LSE constraint:** `lse=true` is only valid when `paged=false`, `local=false`, `sink=false`.
It is used for distributed attention merging (chunked prefill states).

| paged | causal | local | sink | lse | Valid? | Use Case |
|-------|--------|-------|------|-----|--------|----------|
| true  | true   | false | false | false | ✅ | Paged KV cache |
| false | true   | false | false | false | ✅ | Initial prompt, no paging |
| false | true   | false | false | true  | ✅ | Chunked prefill state merge |
| false | true   | true  | false | false | ✅ | Sliding window attention |
| false | true   | true  | true  | false | ✅ | Sliding window + sink tokens |
| false | true   | false | true  | false | ✅ | Sink token optimization |
| any   | any    | any   | any   | true  | ❌ | LSE requires paged=false, local=false, sink=false |

### Paged Decode: 3 Bool Parameters (`causal`, `local`, `sink`)

| causal | local | sink | Use Case |
|--------|-------|------|----------|
| true   | false | false | Standard causal (most common) |
| true   | true  | false | Sliding window (Mistral, Qwen long-context) |
| true   | true  | true  | Sliding window + sink tokens |

---

## Custom Configuration

### Step 1: Identify Your Requirements

```python
# Pseudo-code
head_size = model.hidden_size // model.num_heads
causal    = True                          # most language models
local     = model.has_sliding_window      # e.g. Mistral, Qwen
sink      = model.has_sink_tokens
paged     = True                          # standard vLLM KV cache
lse       = model.uses_distributed_attn  # multi-GPU chunked prefill merge

qgroup    = 8 if (num_q_heads / num_kv_heads) <= 8 else 16
page_size = 64  # default vLLM-xpu --block-size
```

### Step 2: Create Config Files

`csrc/xpu/attn/kernel_configs/chunk_prefill_custom.conf`:

```conf
# Llama / Qwen (head_size=128)
128,true,true,false,false,false
128,false,true,false,false,false
128,false,true,false,false,true

# Mistral with sliding window (head_size=128)
128,false,true,true,false,false

# DeepSeek MLA (head_size=192)
192,true,true,false,false,false
192,false,true,false,false,false
192,false,true,false,false,true
```

`csrc/xpu/attn/kernel_configs/paged_decode_custom.conf`:

```conf
# Format: qgroup,headsize,pagesize,causal,local,sink

# Llama / Qwen — standard causal, page size 16 and 64
8,128,16,true,false,false
8,128,64,true,false,false

# DeepSeek MLA
8,192,16,true,false,false
8,192,64,true,false,false

# With sliding window (uncomment if needed)
# 8,128,16,true,true,false
```

### Step 3: Rebuild

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_custom.conf \
  pip install .
```

---

## Build & Install

### Via environment variable (pip)

```bash
# Default build (full config — all kernel variants)
pip install .

# Optimized build (Llama/Qwen/DeepSeek only, ~97% fewer kernels)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf \
  pip install .

# Full config (explicit)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf \
  pip install .

# Custom config
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_custom.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_custom.conf \
  pip install .
```

Shorthand names (without `.conf`) are resolved automatically:

```bash
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default \
VLLM_PAGED_DECODE_CONFIG=paged_decode_full \
  pip install .
```

### Via CMake directly

```bash
cmake -DVLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default \
      -DVLLM_PAGED_DECODE_CONFIG=paged_decode_full \
      ...

# Or with a full path:
cmake -DVLLM_CHUNK_PREFILL_CONFIG=/path/to/custom_prefill.conf \
      -DVLLM_PAGED_DECODE_CONFIG=/path/to/custom_decode.conf \
      ...
```

---

## Troubleshooting

### Error: "Chunk prefill kernel not compiled for this configuration"

**Cause:** The `head_size` for your model is not in the config.

```bash
# Option 1: Full config (all head sizes)
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf pip install .

# Option 2: Add the head_size to your config and rebuild
# Edit csrc/xpu/attn/kernel_configs/chunk_prefill_default.conf, add:
#   <your_head_size>,true,true,false,false,false
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf pip install .
```

### Error: "Chunk prefill kernel tuple not compiled for this configuration"

**Cause:** The bool combination (`paged`/`causal`/`local`/`sink`/`lse`) is not compiled for this head_size.
The error message prints the exact config line to add.

```bash
# Option 1: Full config
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf pip install .

# Option 2: Add the specific line printed in the error message
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf pip install .
```

### Error: "Paged decode kernel not compiled for this configuration"

**Cause:** The `(qgroup, headsize, pagesize)` combination is not in the config.

```bash
# Option 1: Full config
VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf pip install .

# Option 2: Add the line printed in the error message
VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf pip install .
```

### How to check which configs were compiled

CMake prints a summary during build:

```
-- Generated chunk_prefill kernel sources: 9 files
   (config: .../chunk_prefill_default.conf)
-- Generated paged_decode kernel sources: 384 files
   (config: .../paged_decode_full.conf)
```

Inspect the generated policy-availability headers directly:

```bash
cat build/temp_template/csrc/xpu/attn/xe_2/chunk_prefill_enabled_policies_gen.hpp
cat build/temp_template/csrc/xpu/attn/xe_2/paged_decode_enabled_policies_gen.hpp
```

---

## Performance Notes

### Build Time vs Runtime Flexibility

| Config | Build Time | Chunk Prefill Kernels | Paged Decode Kernels | Flexibility |
|--------|------------|----------------------|----------------------|-------------|
| `default` | ~2 min | ~13 | ~17 | Llama, Qwen, DeepSeek MLA, Falcon |
| `full` | ~60 min | 216 | 384 | All models |

### Binary Size Impact

- Each unique (head_size, bool-combination) → ~500 KB–2 MB compiled kernel
- `default`: ~5 MB | `full`: ~100 MB+

### Recommendation

| Scenario | Config |
|----------|--------|
| Development / first-time setup | `full` |
| CI/CD (broad compatibility) | `full` |
| Production (Llama / Qwen / DeepSeek) | `default` |
| Production (other or unknown models) | `full` or custom |

---

## Implementation References

- Config file parsing: `csrc/xpu/attn/xe_2/chunk_prefill_configure.cmake`, `paged_decode_configure.cmake`
- Runtime policy checks: `csrc/xpu/attn/xe_2/chunk_prefill_utils.hpp`, `paged_decode_utils.hpp`
- Config files: `csrc/xpu/attn/kernel_configs/`
