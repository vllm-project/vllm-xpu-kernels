# Custom kernel inventory

This checkout did not contain an inventory before the Qwen4-Exp work. The
entries below cover the operators added by that effort.

| Extension | Operator | Purpose |
| --- | --- | --- |
| `_xpu_C` | `qwen4_exp_ple_ngram_ids` | Compute request-aware PLE n-gram embedding IDs. |
| `_xpu_C` | `qwen4_exp_ple_conv` | Run decode, prefill, or speculative dilated short convolution and update persistent state. |
| `_xpu_C` | `qwen4_exp_qsa_visible_blocks` | Derive visible compressed-block counts. |
| `_xpu_C` | `qwen4_exp_qsa_expand_block_indices` | Expand selected compressed blocks to causal token indices. |
| `_xpu_C` | `qwen4_exp_qsa_store_cache_rows` | Store fixed-width data or position rows into paged caches. |
| `_xpu_C` | `qwen4_exp_qsa_compress_groups` | Pool completed raw-key groups and recover their first RoPE positions. |
