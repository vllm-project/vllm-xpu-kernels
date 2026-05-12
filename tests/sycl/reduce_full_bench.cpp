// Pure SYCL reduce benchmark with FULL attention reduction logic
// (max_logits reduction, rescale with exp2, weighted sum, normalize)
// Tests: plain sum vs full reduce vs GRF 256 impact
//
// Build:
//   icpx -fsycl -O2 -o reduce_full_bench reduce_full_bench.cpp
// Run:
//   ./reduce_full_bench

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

int main() {
    int num_heads_q = 32;
    int num_splits  = 16;
    int head_dim    = 128;
    int iterations  = 200;
    int warmup      = 50;
    int wg_size     = 64;  // match CUTLASS: 4 SG × 16

    size_t in_size = (size_t)num_heads_q * num_splits * head_dim;
    size_t out_size = (size_t)num_heads_q * head_dim;
    size_t meta_size = (size_t)num_heads_q * num_splits; // exp_sums, max_logits

    printf("=== Full Reduce Benchmark (Pure SYCL) ===\n");
    printf("  num_heads_q=%d  num_splits=%d  head_dim=%d  wg_size=%d\n",
           num_heads_q, num_splits, head_dim, wg_size);
    printf("  Oaccum: %zu floats = %.1f KB\n", in_size, in_size * 4.0 / 1024);
    printf("  Output: %zu floats = %.1f KB\n\n", out_size, out_size * 4.0 / 1024);

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};

    float* d_oaccum     = sycl::malloc_device<float>(in_size, q);
    float* d_out        = sycl::malloc_device<float>(out_size, q);
    float* d_exp_sums   = sycl::malloc_device<float>(meta_size, q);
    float* d_max_logits = sycl::malloc_device<float>(meta_size, q);

    // Init
    std::vector<float> h_oaccum(in_size), h_exp(meta_size), h_max(meta_size);
    for (auto& v : h_oaccum) v = 0.1f;
    for (int h = 0; h < num_heads_q; h++)
        for (int s = 0; s < num_splits; s++) {
            h_exp[h * num_splits + s] = 1.0f;
            h_max[h * num_splits + s] = -0.5f + 0.1f * s;
        }
    q.memcpy(d_oaccum, h_oaccum.data(), in_size * sizeof(float));
    q.memcpy(d_exp_sums, h_exp.data(), meta_size * sizeof(float));
    q.memcpy(d_max_logits, h_max.data(), meta_size * sizeof(float));
    q.wait();

    // ========== Test 1: Plain sum (same as previous pure SYCL) ==========
    auto run_plain = [&]() {
        return q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);
                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s)
                            acc += d_oaccum[head_q * num_splits * head_dim + s * head_dim + idx];
                        d_out[head_q * head_dim + idx] = acc;
                    }
                });
        });
    };

    // ========== Test 2: Full reduce (max + rescale + normalize) ==========
    auto run_full = [&]() {
        return q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);

                    // Step 1: find global max (all threads redundantly)
                    float gmax = -1e30f;
                    for (int s = 0; s < num_splits; ++s) {
                        float val = d_max_logits[head_q * num_splits + s];
                        gmax = sycl::max(gmax, val);
                    }

                    // Step 2: compute rescale + global exp sum
                    float gexp = 0.0f;
                    for (int s = 0; s < num_splits; ++s) {
                        float lmax = d_max_logits[head_q * num_splits + s];
                        float lexp = d_exp_sums[head_q * num_splits + s];
                        gexp += lexp * sycl::native::exp2(lmax - gmax);
                    }
                    float inv_gexp = 1.0f / gexp;

                    // Step 3: weighted sum + normalize
                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s) {
                            float lmax = d_max_logits[head_q * num_splits + s];
                            float lexp = d_exp_sums[head_q * num_splits + s];
                            float rescale = lexp * sycl::native::exp2(lmax - gmax);
                            float oval = d_oaccum[head_q * num_splits * head_dim + s * head_dim + idx];
                            acc += oval * rescale;
                        }
                        d_out[head_q * head_dim + idx] = acc * inv_gexp;
                    }
                });
        });
    };

    // ========== Test 3: Full reduce with GRF 256 ==========
    auto run_full_grf256 = [&]() {
        return q.submit([&](sycl::handler& cgh) {
            syclex::properties kernel_props{intelex::grf_size<256>};
            cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
                kernel_props,
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);

                    float gmax = -1e30f;
                    for (int s = 0; s < num_splits; ++s) {
                        float val = d_max_logits[head_q * num_splits + s];
                        gmax = sycl::max(gmax, val);
                    }

                    float gexp = 0.0f;
                    for (int s = 0; s < num_splits; ++s) {
                        float lmax = d_max_logits[head_q * num_splits + s];
                        float lexp = d_exp_sums[head_q * num_splits + s];
                        gexp += lexp * sycl::native::exp2(lmax - gmax);
                    }
                    float inv_gexp = 1.0f / gexp;

                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s) {
                            float lmax = d_max_logits[head_q * num_splits + s];
                            float lexp = d_exp_sums[head_q * num_splits + s];
                            float rescale = lexp * sycl::native::exp2(lmax - gmax);
                            float oval = d_oaccum[head_q * num_splits * head_dim + s * head_dim + idx];
                            acc += oval * rescale;
                        }
                        d_out[head_q * head_dim + idx] = acc * inv_gexp;
                    }
                });
        });
    };

    // Benchmark helper
    auto bench = [&](auto fn, const char* name) {
        for (int i = 0; i < warmup; ++i) fn();
        q.wait();

        std::vector<sycl::event> events;
        events.reserve(iterations);
        for (int i = 0; i < iterations; ++i)
            events.push_back(fn());
        q.wait();

        double total_ns = 0;
        for (auto& e : events) {
            uint64_t s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t en = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_ns += (en - s);
        }
        double avg_us = total_ns / iterations / 1000.0;
        printf("  %-30s %5.2f us\n", name, avg_us);
    };

    printf("\n=== Results ===\n");
    bench(run_plain, "Plain sum (no rescale):");
    bench(run_full, "Full reduce (GRF 128):");
    bench(run_full_grf256, "Full reduce (GRF 256):");
    printf("\n  CUTLASS reduce (unitrace):   %5.2f us\n", 7.4);

    sycl::free(d_oaccum, q);
    sycl::free(d_out, q);
    sycl::free(d_exp_sums, q);
    sycl::free(d_max_logits, q);
    return 0;
}
