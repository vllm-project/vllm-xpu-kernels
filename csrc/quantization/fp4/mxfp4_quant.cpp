// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <type_traits>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp4/mxfp4_quant.h"

// per_token_group_quant_mxfp4
//
// Quantise input [M, N] to MXFP4 block format (MX specification).
//
// Arguments:
//   input    [M, N]       – float / half / bfloat16 input tensor
//   output_q [M, N/2]     – packed FP4 output (uint8, two nibbles per byte)
//   output_s [M, N/g]     – UE8M0-rounded power-of-two scale (float32)
//                           May be column-major (see below).
//   group_size            – block size; must be 32 for the MX format
//   eps                   – absolute minimum to avoid log2(0); default 1e-10
void per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_q.is_contiguous(), "output_q must be contiguous");
  TORCH_CHECK(
      group_size == 32, "MXFP4 requires group_size == 32, got ", group_size);
  TORCH_CHECK(
      input.numel() % group_size == 0,
      "input numel must be divisible by group_size");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be 2-D");
  const auto moe_layout = vllm::make_moe_scale_layout(
      output_s,
      expert_scale_desc,
      static_cast<int>(input.size(-1) / group_size),
      input.size(-1) > 0 ? input.numel() / input.size(-1) : 0);
  TORCH_CHECK(
      output_q.scalar_type() == at::ScalarType::Byte, "output_q must be uint8");

  const int num_groups = static_cast<int>(input.numel() / group_size);
  if (num_groups == 0) {
    // No work to do for empty input; ensure outputs are empty as well.
    TORCH_CHECK(
        output_q.numel() == 0, "output_q must be empty when input is empty");
    TORCH_CHECK(
        moe_layout.enabled() || output_s.numel() == 0,
        "output_s must be empty when input is empty");
    return;
  }

  const at::DeviceGuard device_guard(input.device());

  // Choose how many sub-groups to pack into one work-group.
  constexpr int THREADS_PER_GROUP = 32;
  int groups_per_block = 1;
  if (num_groups % 16 == 0)
    groups_per_block = 16;
  else if (num_groups % 8 == 0)
    groups_per_block = 8;
  else if (num_groups % 4 == 0)
    groups_per_block = 4;
  else if (num_groups % 2 == 0)
    groups_per_block = 2;

  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  // Detect column-major scale layout.
  const bool is_column_major =
      !moe_layout.enabled() && output_s.stride(0) < output_s.stride(1);
  float* const scale_ptr =
      moe_layout.enabled() ? nullptr : output_s.data_ptr<float>();
  // scale_num_cols = number of groups per token row (= N / group_size).
  const int scale_num_cols = output_s.size(1);
  const int scale_stride = static_cast<int>(output_s.stride(1));

  sycl::range<1> grid(num_blocks);
  sycl::range<1> block(num_threads);
  auto& queue = vllm::xpu::vllmGetQueue();

  const int hidden = static_cast<int>(input.size(-1));
  const int64_t num_rows = input.numel() / hidden;
  const int64_t scale_stride_token =
      is_column_major ? 1LL : static_cast<int64_t>(scale_num_cols);
  const int64_t scale_stride_group =
      is_column_major ? static_cast<int64_t>(scale_stride) : 1LL;

  bool use_vec = (hidden % group_size == 0);

  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_mxfp4_kernel", [&] {
        constexpr int VEC = 16 / sizeof(scalar_t);
        if (use_vec && (hidden % VEC == 0) && (group_size % VEC == 0)) {
          const int num_chunks = hidden / VEC;
          const int chunks_per_group = static_cast<int>(group_size) / VEC;
          const int64_t saturate_wi = static_cast<int64_t>(eu_count) * 128;
          const int64_t base_wi = static_cast<int64_t>(num_rows) * num_chunks;
          int cpt = 1;
          for (int c : {4, 2}) {
            if (chunks_per_group % c == 0 && base_wi / c >= saturate_wi) {
              cpt = c;
              break;
            }
          }
          const int threads_per_row = num_chunks / cpt;
          const int rows_per_wg =
              threads_per_row >= 32 ? 1 : 32 / threads_per_row;
          int64_t wg_size = std::min<int64_t>(
              ((static_cast<int64_t>(threads_per_row) * rows_per_wg + 31) /
               32) *
                  32,
              1024);
          if (wg_size < 32) wg_size = 32;
          const int64_t num_wgs = (num_rows + rows_per_wg - 1) / rows_per_wg;
          auto launch = [&](auto cpt_tag) {
            constexpr int CPT = decltype(cpt_tag)::value;
            if constexpr (CPT * VEC == 32) {
              if (!moe_layout.enabled() && !is_column_major &&
                  output_s.is_contiguous() &&
                  scale_num_cols == hidden / group_size) {
                constexpr int FLAT_WG_SIZE = 128;
                const int64_t flat_groups =
                    (static_cast<int64_t>(num_groups) + FLAT_WG_SIZE - 1) /
                    FLAT_WG_SIZE;
                queue.submit([&](sycl::handler& cgh) {
                  auto kernel = vllm::mxfp4::
                      per_token_group_quant_mxfp4_vec_kernel<
                          scalar_t, CPT, true>(
                          output_q.data_ptr<uint8_t>(),
                          scale_ptr,
                          input.data_ptr<scalar_t>(),
                          32,
                          32,
                          static_cast<float>(eps),
                          1,
                          1,
                          num_groups,
                          1,
                          moe_layout);
                  cgh.parallel_for(
                      sycl::nd_range<1>(
                          flat_groups * FLAT_WG_SIZE, FLAT_WG_SIZE),
                      kernel);
                });
                return;
              }

              // MoE scale surfaces are group-major, so a one-row work-group
              // scatters its e8m0 bytes `ld` apart; stage a ROWS x K tile in
              // SLM instead. The staging barrier is amortised once the row
              // count is high enough (measured crossover near 4096 rows), and
              // always pays when rows are too narrow to fill a sub-group.
              constexpr int64_t kMinStagingRows = 4096;
              const bool narrow_rows = threads_per_row * rows_per_wg < 32;
              const bool scatter_amortised = num_rows >= kMinStagingRows;
              if (moe_layout.enabled() && (narrow_rows || scatter_amortised)) {
                constexpr int kMaxWorkGroup = 1024;
                // slm_rows must span several bytes to escape the
                // one-byte-per-cache-line scatter, and the quantisation phase
                // wants >= 256 work-items.
                constexpr int kMinScaleRun = 4;
                constexpr int kMinWorkGroup = 256;
                const int groups_per_row =
                    hidden / static_cast<int>(group_size);
                const int max_rows = static_cast<int>(std::min<int64_t>(
                    kMaxWorkGroup / groups_per_row, num_rows));
                int slm_rows = kMinScaleRun;
                while (slm_rows * groups_per_row < kMinWorkGroup &&
                       slm_rows * 2 <= max_rows) {
                  slm_rows *= 2;
                }
                while (slm_rows > max_rows) {
                  slm_rows /= 2;
                }
                // Rows too wide for kMinScaleRun of them have a negligible
                // scale surface, so staging would cost more than it saves.
                if (slm_rows >= kMinScaleRun) {
                  const int64_t tile_threads =
                      static_cast<int64_t>(groups_per_row) * slm_rows;
                  // Tail lanes are masked off inside the kernel.
                  const int64_t slm_wg = (tile_threads + 31) / 32 * 32;
                  const int64_t slm_wgs = (num_rows + slm_rows - 1) / slm_rows;
                  queue.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<uint8_t, 1> tile(
                        sycl::range<1>(tile_threads), cgh);
                    auto kernel =
                        vllm::mxfp4::per_token_group_quant_mxfp4_moe_slm_kernel<
                            scalar_t,
                            CPT>(
                            output_q.data_ptr<uint8_t>(),
                            input.data_ptr<scalar_t>(),
                            tile,
                            hidden,
                            static_cast<float>(eps),
                            static_cast<int>(num_rows),
                            slm_rows,
                            moe_layout);
                    cgh.parallel_for(
                        sycl::nd_range<1>(slm_wgs * slm_wg, slm_wg), kernel);
                  });
                  return;
                }
              }
            }
            queue.submit([&](sycl::handler& cgh) {
              auto kernel = vllm::mxfp4::
                  per_token_group_quant_mxfp4_vec_kernel<scalar_t, CPT>(
                      output_q.data_ptr<uint8_t>(),
                      scale_ptr,
                      input.data_ptr<scalar_t>(),
                      hidden,
                      static_cast<int>(group_size),
                      static_cast<float>(eps),
                      scale_stride_token,
                      scale_stride_group,
                      static_cast<int>(num_rows),
                      rows_per_wg,
                      moe_layout);
              cgh.parallel_for(
                  sycl::nd_range<1>(num_wgs * wg_size, wg_size), kernel);
            });
          };
          if (cpt == 4) {
            launch(std::integral_constant<int, 4>{});
          } else if (cpt == 2) {
            launch(std::integral_constant<int, 2>{});
          } else {
            launch(std::integral_constant<int, 1>{});
          }
        } else {
          queue.submit([&](sycl::handler& cgh) {
            auto kernel =
                vllm::mxfp4::per_token_group_quant_mxfp4_kernel<scalar_t>(
                    output_q.data_ptr<uint8_t>(),
                    scale_ptr,
                    input.data_ptr<scalar_t>(),
                    static_cast<int>(group_size),
                    groups_per_block,
                    static_cast<float>(eps),
                    scale_num_cols,
                    scale_stride,
                    is_column_major,
                    moe_layout);
            cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
          });
        }
      });
}

// silu_and_mul_per_token_group_quant_mxfp4
//
// Same as per_token_group_quant_mxfp4, but the input holds the gate and up
// halves of a SwiGLU back to back ([M, 2N]) and the activation is folded into
// the quantiser. This avoids materialising and re-reading the [M, N] bf16
// activation between the two grouped GEMMs of a MoE layer.
void silu_and_mul_per_token_group_quant_mxfp4(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    int64_t group_size,
    double eps,
    const c10::optional<torch::Tensor>& expert_scale_desc) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_q.is_contiguous(), "output_q must be contiguous");
  TORCH_CHECK(
      group_size == 32, "MXFP4 requires group_size == 32, got ", group_size);
  TORCH_CHECK(input.dim() == 2, "input must be 2-D");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be 2-D");
  TORCH_CHECK(
      output_q.scalar_type() == at::ScalarType::Byte, "output_q must be uint8");
  TORCH_CHECK(
      input.size(1) % 2 == 0,
      "input last dim must be even, got ",
      input.size(1));

  const int hidden = static_cast<int>(input.size(1) / 2);
  const int64_t num_rows = input.size(0);
  TORCH_CHECK(
      hidden % group_size == 0,
      "hidden_size must be divisible by group_size, got ",
      hidden);
  TORCH_CHECK(
      output_q.size(0) == num_rows && output_q.size(1) == hidden / 2,
      "output_q must have shape [M, hidden_size / 2]");
  TORCH_CHECK(
      output_s.size(1) == hidden / static_cast<int>(group_size),
      "output_s must have shape [M, hidden_size / group_size]");

  const auto moe_layout = vllm::make_moe_scale_layout(
      output_s,
      expert_scale_desc,
      hidden / static_cast<int>(group_size),
      num_rows);

  // make_moe_scale_layout already sizes and type-checks the padded e8m0
  // surface, so these only constrain the plain float32 scale tensor.
  if (!moe_layout.enabled()) {
    TORCH_CHECK(
        output_s.scalar_type() == at::ScalarType::Float,
        "output_s must be float32 without expert_scale_desc");
    TORCH_CHECK(
        output_s.size(0) == num_rows,
        "output_s must have shape [M, hidden_size / group_size]");
  }

  if (num_rows == 0 || hidden == 0) {
    return;
  }

  const at::DeviceGuard device_guard(input.device());

  const bool is_column_major =
      !moe_layout.enabled() && output_s.stride(0) < output_s.stride(1);
  float* const scale_ptr =
      moe_layout.enabled() ? nullptr : output_s.data_ptr<float>();
  const int scale_num_cols = static_cast<int>(output_s.size(1));
  const int64_t scale_stride_token =
      is_column_major ? 1LL : static_cast<int64_t>(scale_num_cols);
  const int64_t scale_stride_group = is_column_major ? output_s.stride(1) : 1LL;

  auto& queue = vllm::xpu::vllmGetQueue();

  const int eu_count = static_cast<int>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_group_quant_mxfp4", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        constexpr int VEC = 16 / sizeof(sycl_t);
        TORCH_CHECK(
            hidden % VEC == 0 && group_size % VEC == 0,
            "fused silu+mxfp4 quant needs hidden_size divisible by ",
            VEC);
        const int num_chunks = hidden / VEC;
        const int chunks_per_group = static_cast<int>(group_size) / VEC;
        // Same chunks-per-thread heuristic as per_token_group_quant_mxfp4:
        // widen the per-thread tile only while the launch still saturates.
        const int64_t saturate_wi = static_cast<int64_t>(eu_count) * 128;
        const int64_t base_wi = num_rows * num_chunks;
        int cpt = 1;
        for (int c : {4, 2}) {
          if (chunks_per_group % c == 0 && base_wi / c >= saturate_wi) {
            cpt = c;
            break;
          }
        }
        const int threads_per_row = num_chunks / cpt;
        const int rows_per_wg = threads_per_row >= 32 ? 1 : 32 / threads_per_row;
        int64_t wg_size = std::min<int64_t>(
            ((static_cast<int64_t>(threads_per_row) * rows_per_wg + 31) / 32) *
                32,
            1024);
        if (wg_size < 32) wg_size = 32;
        const int64_t num_wgs = (num_rows + rows_per_wg - 1) / rows_per_wg;
        auto launch = [&](auto cpt_tag) {
          constexpr int CPT = decltype(cpt_tag)::value;
          queue.submit([&](sycl::handler& cgh) {
            auto kernel = vllm::mxfp4::per_token_group_quant_mxfp4_vec_kernel<
                sycl_t,
                CPT,
                false,
                true>(
                output_q.data_ptr<uint8_t>(),
                scale_ptr,
                reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>()),
                hidden,
                static_cast<int>(group_size),
                static_cast<float>(eps),
                scale_stride_token,
                scale_stride_group,
                static_cast<int>(num_rows),
                rows_per_wg,
                moe_layout);
            cgh.parallel_for(
                sycl::nd_range<1>(num_wgs * wg_size, wg_size), kernel);
          });
        };
        if (cpt == 4) {
          launch(std::integral_constant<int, 4>{});
        } else if (cpt == 2) {
          launch(std::integral_constant<int, 2>{});
        } else {
          launch(std::integral_constant<int, 1>{});
        }
      });
}
