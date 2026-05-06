# Paged Decode Kernel Configuration

This directory contains kernel configuration files that control which paged decode
attention kernel variants are compiled. Using a model-specific config instead of the
full config can **reduce compile time by 80-95%** and produce significantly smaller
binaries.

## Usage

### Via CMake directly:

```bash
cmake -DVLLM_PAGED_DECODE_CONFIG=llama ...
# or with full path:
cmake -DVLLM_PAGED_DECODE_CONFIG=/path/to/my_custom.conf ...
```

### Via environment variable (with pip/setup.py):

```bash
VLLM_PAGED_DECODE_CONFIG=llama pip install -e .
VLLM_PAGED_DECODE_CONFIG=common pip install -e .
```

### Via absolute path:

```bash
VLLM_PAGED_DECODE_CONFIG=/path/to/custom.conf pip install -e .
```

## Available Presets

| Config | Kernels | Reduction | Models Covered |
|--------|---------|-----------|----------------|
| `full.conf` | 384 | 0% (baseline) | All possible combinations |
| `common.conf` | ~72 | ~81% | Llama, Qwen, Mistral, DeepSeek, Gemma |
| `llama.conf` | ~24 | ~94% | Llama-2, Llama-3, CodeLlama |
| `qwen.conf` | ~24 | ~94% | Qwen2, Qwen2.5, Qwen3 |
| `deepseek.conf` | ~48 | ~88% | DeepSeek-V2, V3, R1 |

## Config File Format

```
# Lines starting with # are comments
# Empty lines are ignored

# Use 'all' to build everything (equivalent to full.conf):
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

## How to Determine Your Model's Config

For a given model, you need:
1. **head_size**: Check the model config (`hidden_size / num_attention_heads`)
2. **GQA ratio**: `num_attention_heads / num_key_value_heads` → maps to qgroup 8 or 16
3. **page_size**: Your vLLM deployment's `--block-size` (default: 16)

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

Simply create a `.conf` file with one line per (qgroup, headsize, pagesize) tuple:

```bash
# my_model.conf - for a model with head_size=128, GQA=8, using page_size 16 and 64
8,128,16
8,128,64
```

Then build with:

```bash
VLLM_PAGED_DECODE_CONFIG=/path/to/my_model.conf pip install -e .
```

## Runtime Behavior

If your deployment encounters a kernel configuration that wasn't compiled, you'll get
a clear error message:

```
RuntimeError: Paged decode kernel not compiled for this configuration.
Rebuild with a kernel config that includes the required policy,
or use VLLM_PAGED_DECODE_CONFIG=.../kernel_configs/full.conf
```
