// N-gram embedding index kernel for LongCat-Flash (n-gram embedding variant).
//
// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/ngram_embedding_kernels.cu
// which is in turn adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/ngram_embedding.cuh
//
// For each position, computes the hashed n-gram embedding ids that index the
// concatenated embedder table. Integer tensors are int32 except ``row_indices``
// (int64); the token table is ``[max_running_reqs, max_context_len]`` int32,
// where a negative entry marks an ignored token (e.g. an EOS boundary).

#include <ATen/DeviceGuard.h>
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include <cstdint>

#include "utils.h"

namespace vllm::ngram_embedding {

constexpr int kBlockThreads = 256;

class ComputeNGramIdsKernel {
 public:
  ComputeNGramIdsKernel(
      int batch_size,
      int ne_n,
      int ne_k,
      const int32_t* __restrict__ ne_weights,
      const int32_t* __restrict__ ne_mods,
      const int32_t* __restrict__ exclusive_ne_embedder_size_sums,
      const int32_t* __restrict__ exclusive_req_len_sums,
      const int32_t* __restrict__ ne_token_table,
      int max_context_len,
      const int64_t* __restrict__ row_indices,
      const int32_t* __restrict__ column_starts,
      int32_t* __restrict__ n_gram_ids)
      : batch_size_(batch_size),
        ne_n_(ne_n),
        ne_k_(ne_k),
        ne_weights_(ne_weights),
        ne_mods_(ne_mods),
        exclusive_ne_embedder_size_sums_(exclusive_ne_embedder_size_sums),
        exclusive_req_len_sums_(exclusive_req_len_sums),
        ne_token_table_(ne_token_table),
        max_context_len_(max_context_len),
        row_indices_(row_indices),
        column_starts_(column_starts),
        n_gram_ids_(n_gram_ids) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int group_id = item.get_group(0);
    const int req_id = group_id % batch_size_;
    const int config_id = group_id / batch_size_;
    // n and k are offset from their physical meaning: n = real_n - 2,
    // k = real_k - 1 (they index into ne_weights / ne_mods).
    const int k = config_id % ne_k_;
    const int n = config_id / ne_k_;
    const int ne_weight_base_idx = n * ne_k_ * ne_n_ + k * ne_n_;
    const uint64_t ne_mod = static_cast<uint64_t>(ne_mods_[n * ne_k_ + k]);
    const int32_t size_offset = exclusive_ne_embedder_size_sums_[n * ne_k_ + k];

    const int req_start = exclusive_req_len_sums_[req_id];
    const int req_end = exclusive_req_len_sums_[req_id + 1];
    const int64_t req_token_table_index =
        row_indices_[req_id] * static_cast<int64_t>(max_context_len_);
    const int64_t req_column_base =
        req_token_table_index + column_starts_[req_id];

    for (int i = req_start + static_cast<int>(item.get_local_id(0));
         i < req_end;
         i += static_cast<int>(item.get_local_range(0))) {
      uint64_t n_gram_id = 0;
      const int64_t current_token_table_index =
          req_column_base + (i - req_start);
      for (int j = 0; j < n + 2; j++) {
        const int64_t idx = current_token_table_index - j;
        if (idx < req_token_table_index) {
          break;  // outside this request's range
        }
        const int32_t token = ne_token_table_[idx];
        if (token < 0) {
          break;  // ignored token
        }
        const uint64_t term =
            static_cast<uint64_t>(token) *
            static_cast<uint64_t>(ne_weights_[ne_weight_base_idx + j]);
        n_gram_id += term % ne_mod;
      }
      n_gram_id %= ne_mod;
      n_gram_id += size_offset;
      n_gram_ids_[i * (ne_n_ - 1) * ne_k_ + n * ne_k_ + k] =
          static_cast<int32_t>(n_gram_id);
    }
  }

 private:
  const int batch_size_;
  const int ne_n_;
  const int ne_k_;
  const int32_t* __restrict__ ne_weights_;  // [ne_n-1, ne_k, ne_n]
  const int32_t* __restrict__ ne_mods_;     // [ne_n-1, ne_k]
  const int32_t* __restrict__ exclusive_ne_embedder_size_sums_;  // [(ne_n-1)*
                                                                 // ne_k + 1]
  const int32_t* __restrict__ exclusive_req_len_sums_;  // [batch_size + 1]
  const int32_t* __restrict__ ne_token_table_;          // [max_running_reqs,
                                                        // max_context_len]
  const int max_context_len_;
  const int64_t* __restrict__ row_indices_;    // [batch_size]
  const int32_t* __restrict__ column_starts_;  // [batch_size]
  int32_t* __restrict__ n_gram_ids_;           // [token_num, (ne_n-1)*ne_k]
};

}  // namespace vllm::ngram_embedding

void ngram_compute_n_gram_ids(
    int64_t ne_n,
    int64_t ne_k,
    torch::Tensor& ne_weights,
    torch::Tensor& ne_mods,
    torch::Tensor& exclusive_ne_embedder_size_sums,
    torch::Tensor& exclusive_req_len_sums,
    torch::Tensor& ne_token_table,
    torch::Tensor& row_indices,
    torch::Tensor& column_starts,
    torch::Tensor& n_gram_ids) {
  for (const torch::Tensor* t :
       {&ne_weights,
        &ne_mods,
        &exclusive_ne_embedder_size_sums,
        &exclusive_req_len_sums,
        &ne_token_table,
        &column_starts,
        &n_gram_ids}) {
    CHECK_DEVICE((*t));
    CHECK_CONTIGUOUS((*t));
    TORCH_CHECK(
        t->scalar_type() == at::kInt,
        "ngram_compute_n_gram_ids expects int32 tensors, got ",
        t->scalar_type());
  }
  CHECK_DEVICE(row_indices);
  CHECK_CONTIGUOUS(row_indices);
  TORCH_CHECK(
      row_indices.scalar_type() == at::kLong,
      "row_indices must be int64, got ",
      row_indices.scalar_type());
  TORCH_CHECK(ne_token_table.dim() == 2, "ne_token_table must be 2D");

  const int batch_size = static_cast<int>(exclusive_req_len_sums.size(0) - 1);
  const int max_context_len = static_cast<int>(ne_token_table.size(1));
  const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
  const int grid_size = num_configs * batch_size;
  if (grid_size <= 0) return;

  const at::DeviceGuard device_guard(ne_weights.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const int block = vllm::ngram_embedding::kBlockThreads;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(grid_size) * block, static_cast<size_t>(block)),
        vllm::ngram_embedding::ComputeNGramIdsKernel(
            batch_size,
            static_cast<int>(ne_n),
            static_cast<int>(ne_k),
            ne_weights.data_ptr<int32_t>(),
            ne_mods.data_ptr<int32_t>(),
            exclusive_ne_embedder_size_sums.data_ptr<int32_t>(),
            exclusive_req_len_sums.data_ptr<int32_t>(),
            ne_token_table.data_ptr<int32_t>(),
            max_context_len,
            row_indices.data_ptr<int64_t>(),
            column_starts.data_ptr<int32_t>(),
            n_gram_ids.data_ptr<int32_t>()));
  });
}
