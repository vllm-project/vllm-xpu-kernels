#pragma once
#include <sycl/sycl.hpp>

#include <limits>

namespace CompactSamplingMaskImpl {
struct compact_sampling_mask_kernel {
 public:
  static constexpr int sub_group_size = 16;
  static constexpr int group_size = 512;
  static constexpr int VEC_SIZE = 4;

  compact_sampling_mask_kernel(
      const float* logits,
      const int* num_sampled_tokens,
      int* token_ids,
      uint8_t* packed_mask,
      int* counts,
      const int num_reqs,
      const int max_num_kept,
      const int vocab_size)
      : logits(logits),
        num_sampled_tokens(num_sampled_tokens),
        token_ids(token_ids),
        packed_mask(packed_mask),
        counts(counts),
        num_reqs(num_reqs),
        max_num_kept(max_num_kept),
        vocab_size(vocab_size) {}

  static inline sycl::nd_range<1>
  get_nd_range(const int num_reqs, const int vocab_size) {
    int local_size = group_size;
    if (vocab_size < group_size) {
      local_size =
          (vocab_size + sub_group_size - 1) / sub_group_size * sub_group_size;
    }
    sycl::range<1> local(local_size);
    sycl::range<1> global(num_reqs);
    return sycl::nd_range<1>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    const int batch_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int local_range = item.get_local_range(0);

    const int aligned_vocab_size = (vocab_size + 7) / 8;

    const float* logits_ptr = logits + batch_id * vocab_size;
    int* token_ids_ptr = token_ids + batch_id * max_num_kept;
    uint8_t* packed_mask_ptr = packed_mask + batch_id * aligned_vocab_size;

    int num_sampled_tokens_ = num_sampled_tokens[batch_id];
    bool is_active = num_sampled_tokens_ > 0;

    if (!is_active) {
      counts[batch_id] = 0;

      using VecT = sycl::vec<int, VEC_SIZE>;
      constexpr int bytes_per_vec = static_cast<int>(sizeof(int) * VEC_SIZE);
      VecT* packed_mask_vec_ptr = reinterpret_cast<VecT*>(packed_mask_ptr);
      const int aligned_vocab_vecs = aligned_vocab_size / bytes_per_vec;

      VecT zero_vec(0);
      for (int i = local_id; i < aligned_vocab_vecs; i += local_range) {
        packed_mask_vec_ptr[i] = zero_vec;
      }

      // Tail bytes that don't fill a full 4-int (16-byte) pack.
      for (int i = aligned_vocab_vecs * bytes_per_vec + local_id;
           i < aligned_vocab_size;
           i += local_range) {
        packed_mask_ptr[i] = 0;
      }

      return;
    }

    // Same-sized "wave" (local_range * VEC_SIZE contiguous tokens) per
    // iteration for every work-item, so the group-wide scan/reduce calls
    // below stay convergent (every work-item calls them the same number
    // of times) and the compact token ids come out in increasing order,
    // matching the Triton kernel's per-block cumsum semantics.
    using LogitsVecT = sycl::vec<float, VEC_SIZE>;
    using TokenIdVecT = sycl::vec<int, VEC_SIZE>;
    const int wave_size = local_range * VEC_SIZE;
    const int num_waves = (vocab_size + wave_size - 1) / wave_size;
    auto sub_group = item.get_sub_group();

    int running_count = 0;

    for (int w = 0; w < num_waves; ++w) {
      const int base = w * wave_size + local_id * VEC_SIZE;
      const bool full_load = base + VEC_SIZE <= vocab_size;

      // Load: one vec read for a full in-bounds chunk, scalar fallback
      // (padded with -inf, like Triton's `other=-inf`) for the tail.
      float local_logits[VEC_SIZE];
      if (full_load) {
        *reinterpret_cast<LogitsVecT*>(local_logits) =
            *reinterpret_cast<const LogitsVecT*>(logits_ptr + base);
      } else {
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          const int idx = base + j;
          local_logits[j] = idx < vocab_size
                                ? logits_ptr[idx]
                                : -std::numeric_limits<float>::infinity();
        }
      }

      // Process element-by-element on the local copy.
      bool keep[VEC_SIZE];
      int local_keep = 0;
      int local_token_ids[VEC_SIZE];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        keep[j] = sycl::isfinite(local_logits[j]);
        if (keep[j]) {
          local_token_ids[local_keep++] = base + j;
        }
      }

      const int exclusive_prefix = sycl::exclusive_scan_over_group(
          item.get_group(), local_keep, sycl::plus<int>());
      const int wave_total = sycl::reduce_over_group(
          item.get_group(), local_keep, sycl::plus<int>());

      // Store: results are buffered in `local_token_ids` above; flush
      // with a single vec store when this chunk is fully kept and fits,
      // otherwise fall back to per-element stores.
      const int write_pos = running_count + exclusive_prefix;
      if (local_keep == VEC_SIZE && write_pos + VEC_SIZE <= max_num_kept) {
        *reinterpret_cast<TokenIdVecT*>(token_ids_ptr + write_pos) =
            *reinterpret_cast<TokenIdVecT*>(local_token_ids);
      } else {
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          if (j < local_keep && write_pos + j < max_num_kept) {
            token_ids_ptr[write_pos + j] = local_token_ids[j];
          }
        }
      }

      // packed_mask: pack `keep` into bits, 8 tokens per byte (little
      // bit order, matching np.packbits(bitorder="little")). VEC_SIZE
      // (4) keep flags only fill half a byte, so this work-item's
      // nibble is combined with the adjacent chunk-index's nibble
      // (even/odd pair, i.e. neighboring lanes) via a sub-group
      // shuffle to form the full byte before the (single) store.
      static_assert(
          VEC_SIZE == 4,
          "packed_mask packing assumes 4 keep bits (a nibble) per "
          "work-item, two of which combine into one byte");
      uint8_t nibble = 0;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        if (keep[j]) {
          nibble |= static_cast<uint8_t>(1u << j);
        }
      }
      const int chunk_index = w * local_range + local_id;
      const int byte_index = chunk_index / 2;
      const int nibble_shift = (chunk_index % 2) * VEC_SIZE;
      const int shifted_nibble = static_cast<int>(nibble) << nibble_shift;

      const int partner_nibble =
          sycl::permute_group_by_xor(sub_group, shifted_nibble, 1);
      const uint8_t byte_val =
          static_cast<uint8_t>(shifted_nibble | partner_nibble);

      if ((chunk_index % 2) == 0 && byte_index < aligned_vocab_size) {
        packed_mask_ptr[byte_index] = byte_val;
      }

      running_count += wave_total;
    }

    if (0 == local_id) {
      counts[batch_id] = running_count;
    }
  }

 private:
  const float* logits;
  const int* num_sampled_tokens;
  int* token_ids;
  uint8_t* packed_mask;
  int* counts;
  const int num_reqs;
  const int max_num_kept;
  const int vocab_size;
};

void compact_sampling_mask_kernel_launcher(
    sycl::queue& queue,
    const float* logits,
    const int* num_sampled_tokens,
    int* token_ids,
    uint8_t* packed_mask,
    int* counts,
    const int num_reqs,
    const int max_num_kept,
    const int vocab_size) {
  using KERNEL = compact_sampling_mask_kernel;
  auto range = KERNEL::get_nd_range(num_reqs, vocab_size);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL task(
        logits,
        num_sampled_tokens,
        token_ids,
        packed_mask,
        counts,
        num_reqs,
        max_num_kept,
        vocab_size);
    cgh.parallel_for(range, task);
  });
}
}  // namespace CompactSamplingMaskImpl
