# Kernel Configuration

This directory contains kernel configuration files that control which attention
kernel variants are compiled. Configs are separated into two categories:

- **Paged Decode** (`paged_decode_*.conf`) — decode-phase attention kernels
- **Chunk Prefill** (`chunk_prefill_*.conf`) — prefill-phase attention kernels

Using model-specific configs instead of the full configs can **reduce compile time
by 80-95%** and produce significantly smaller binaries.

## Usage

### Via CMake directly:

```bash
# Shorthand (resolved to kernel_configs/ automatically)
cmake -DVLLM_PAGED_DECODE_CONFIG=paged_decode_llama \
      -DVLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_llama ...

# Full path:
cmake -DVLLM_PAGED_DECODE_CONFIG=/path/to/custom_decode.conf \
      -DVLLM_CHUNK_PREFILL_CONFIG=/path/to/custom_prefill.conf ...
```

### Via environment variable (with pip/setup.py):

```bash
VLLM_PAGED_DECODE_CONFIG=paged_decode_llama \
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_llama \
  pip install -e .
```

### Via absolute path:

```bash
VLLM_PAGED_DECODE_CONFIG=/path/to/custom_decode.conf \
VLLM_CHUNK_PREFILL_CONFIG=/path/to/custom_prefill.conf \
  pip install -e .
```

---

## Paged Decode Configs

### Available Presets

| Config | Kernels | Reduction | Models Covered |
|--------|---------|-----------|----------------|
| `paged_decode_full.conf` | 384 | 0% (baseline) | All possible combinations |
| `paged_decode_common.conf` | ~20 | ~95% | Llama, Qwen, Mistral, DeepSeek, Gemma |
| `paged_decode_llama.conf` | ~3 | ~99% | Llama-2, Llama-3, CodeLlama |
| `paged_decode_qwen.conf` | ~3 | ~99% | Qwen2, Qwen2.5, Qwen3 |
| `paged_decode_deepseek.conf` | ~6 | ~98% | DeepSeek-V2, V3, R1 |

### Config File Format

```
# Lines starting with # are comments
# Empty lines are ignored

# Use 'all' to build everything (equivalent to paged_decode_full.conf):
# all

# Each line specifies: qgroup,headsize,pagesize
# All 8 boolean combinations (causal/local/sink) are generated automatically
8,128,64
8,128,16
16,128,64

# Optionally specify exact boolean flags: qgroup,headsize,pagesize,causal,local,sink
8,128,64,true,false,false
```

### Parameters:
- **qgroup**: GQA group size bucket — `8` (ratio ≤ 8) or `16` (ratio 9-16)
- **headsize**: Head dimension — `64`, `96`, `128`, `192`, `256`, or `512`
- **pagesize**: KV cache page/block size — `16`, `32`, `64`, or `128`
- **causal**: Whether causal masking is used (almost always `true` for decode)
- **local**: Whether sliding window attention is used
- **sink**: Whether StreamingLLM attention sinks are used

---

## Chunk Prefill Configs

### Available Presets

| Config | Kernels | Reduction | Models Covered |
|--------|---------|-----------|----------------|
| `chunk_prefill_full.conf` | 216 | 0% (baseline) | All possible combinations |
| `chunk_prefill_common.conf` | ~12 | ~94% | Llama, Qwen, Mistral, DeepSeek, Gemma |
| `chunk_prefill_llama.conf` | ~3 | ~99% | Llama-2, Llama-3, CodeLlama |
| `chunk_prefill_qwen.conf` | ~3 | ~99% | Qwen2, Qwen2.5, Qwen3 |
| `chunk_prefill_deepseek.conf` | ~6 | ~97% | DeepSeek-V2, V3, R1 |

### Config File Format

```
# Lines starting with # are comments
# Empty lines are ignored

# Use 'all' to build everything (equivalent to chunk_prefill_full.conf):
# all

# Each line specifies: headsize,paged,causal,local,sink,lse
128,true,true,false,false,false
128,false,true,false,false,true
192,true,true,false,false,false
```

### Parameters:
- **headsize**: Head dimension — `64`, `96`, `128`, `192`, `256`, or `512`
- **paged**: Whether paged KV cache is used
- **causal**: Whether causal masking is applied
- **local**: Whether sliding window attention is used
- **sink**: Whether StreamingLLM attention sinks are used
- **lse**: Whether log-sum-exp is output (only valid when paged=false, local=false, sink=false)

---

## How to Determine Your Model's Config

For a given model, you need:
1. **head_size**: Check the model config (`hidden_size / num_attention_heads`)
2. **GQA ratio** (decode only): `num_attention_heads / num_key_value_heads` → maps to qgroup 8 or 16
3. **page_size** (decode only): Your vLLM deployment's `--block-size` (default: 16)

Common model parameters:
| Model | head_size | GQA ratio | qgroup |
|-------|-----------|-----------|--------|
| Llama-3-8B | 128 | 1 (MHA) | 8 |
| Llama-3-70B | 128 | 8 | 8 |
| Qwen2-72B | 128 | 8 | 8 |
| Qwen3-30B-A3B | 128 | 4 | 8 |
| DeepSeek-V3 | 128+192 | varies | 8 |
| Gemma-2-27B | 256 | 2 | 8 |
| Mistral-7B | 128 | 8 | 8 |

## Creating a Custom Config

### Paged Decode

```bash
# my_decode.conf - head_size=128, GQA=8, page_size 16 and 64
8,128,16
8,128,64
```

### Chunk Prefill

```bash
# my_prefill.conf - head_size=128, paged + non-paged causal
128,true,true,false,false,false
128,false,true,false,false,false
128,false,true,false,false,true
```

Then build with:

```bash
VLLM_PAGED_DECODE_CONFIG=/path/to/my_decode.conf \
VLLM_CHUNK_PREFILL_CONFIG=/path/to/my_prefill.conf \
  pip install -e .
```

## Runtime Behavior

If your deployment encounters a kernel configuration that wasn't compiled, you'll get
a clear error message:

```
RuntimeError: Paged decode kernel not compiled for this configuration.
Rebuild with a kernel config that includes the required policy,
or use VLLM_PAGED_DECODE_CONFIG=.../kernel_configs/paged_decode_full.conf
```

```
RuntimeError: Chunk prefill kernel not compiled for this configuration.
Rebuild with a kernel config that includes the required policy,
or use VLLM_CHUNK_PREFILL_CONFIG=.../kernel_configs/chunk_prefill_full.conf
```
