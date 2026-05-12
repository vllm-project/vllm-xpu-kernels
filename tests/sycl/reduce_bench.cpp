// Pure SYCL reduce kernel benchmark.
// Mimics the splitKV reduction: reads [num_heads_q, num_splits, head_dim] float32,
// sums across splits, writes [num_heads_q, head_dim] float32.
//
// Build:
//   icpx -fsycl -O2 -o reduce_bench reduce_bench.cpp
//
// Run:
//   ./reduce_bench [num_heads_q] [num_splits] [head_dim] [iterations]

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int num_heads_q = argc > 1 ? atoi(argv[1]) : 32;
    int num_splits  = argc > 2 ? atoi(argv[2]) : 16;
    int head_dim    = argc > 3 ? atoi(argv[3]) : 128;
    int iterations  = argc > 4 ? atoi(argv[4]) : 200;
    int warmup      = 50;

    // Input: [num_heads_q, num_splits, head_dim] float32
    // Output: [num_heads_q, head_dim] float32
    size_t in_size = (size_t)num_heads_q * num_splits * head_dim;
    size_t out_size = (size_t)num_heads_q * head_dim;

    printf("=== Pure SYCL Reduce Benchmark ===\n");
    printf("  num_heads_q=%d  num_splits=%d  head_dim=%d\n", num_heads_q, num_splits, head_dim);
    printf("  Input: %zu floats = %.1f KB\n", in_size, in_size * 4.0 / 1024);
    printf("  Output: %zu floats = %.1f KB\n", out_size, out_size * 4.0 / 1024);
    printf("  iterations=%d  warmup=%d\n\n", iterations, warmup);

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    printf("  Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    float* d_in  = sycl::malloc_device<float>(in_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    // Initialize input with random data
    std::vector<float> h_in(in_size);
    for (auto& v : h_in) v = static_cast<float>(rand()) / RAND_MAX;
    q.memcpy(d_in, h_in.data(), in_size * sizeof(float)).wait();
    q.memset(d_out, 0, out_size * sizeof(float)).wait();

    // Kernel config: 1 WG per head_q, each WG has enough threads to cover head_dim
    // Use 128 threads per WG (8 SGs × 16 threads)
    int wg_size = 128;
    int num_wgs = num_heads_q;

    printf("  Kernel: grid=%d  block=%d\n", num_wgs, wg_size);
    printf("  Each thread: %d splits × %d elements = %d loads\n\n",
           num_splits, (head_dim + wg_size - 1) / wg_size,
           num_splits * ((head_dim + wg_size - 1) / wg_size));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);
                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s) {
                            acc += d_in[head_q * num_splits * head_dim + s * head_dim + idx];
                        }
                        d_out[head_q * head_dim + idx] = acc;
                    }
                });
        });
    }
    q.wait();

    // Timed iterations with per-kernel event profiling
    std::vector<sycl::event> events;
    events.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto evt = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);
                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s) {
                            acc += d_in[head_q * num_splits * head_dim + s * head_dim + idx];
                        }
                        d_out[head_q * head_dim + idx] = acc;
                    }
                });
        });
        events.push_back(evt);
    }
    q.wait();

    // Compute average kernel time from events
    double total_ns = 0.0;
    for (auto& evt : events) {
        uint64_t start = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
        total_ns += static_cast<double>(end - start);
    }
    double avg_us = total_ns / iterations / 1000.0;

    // Also measure with host wall clock (batch, no per-iter sync)
    q.wait();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int head_q = item.get_group(0);
                    int tid = item.get_local_id(0);
                    for (int idx = tid; idx < head_dim; idx += wg_size) {
                        float acc = 0.0f;
                        for (int s = 0; s < num_splits; ++s) {
                            acc += d_in[head_q * num_splits * head_dim + s * head_dim + idx];
                        }
                        d_out[head_q * head_dim + idx] = acc;
                    }
                });
        });
    }
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    // Also test empty kernel for baseline
    std::vector<sycl::event> empty_events;
    empty_events.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto evt = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    // empty
                });
        });
        empty_events.push_back(evt);
    }
    q.wait();
    double empty_total_ns = 0.0;
    for (auto& evt : empty_events) {
        uint64_t start = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
        empty_total_ns += static_cast<double>(end - start);
    }
    double empty_avg_us = empty_total_ns / iterations / 1000.0;

    double bytes_read = (double)in_size * sizeof(float);
    double bytes_write = (double)out_size * sizeof(float);
    double bw_gbps = (bytes_read + bytes_write) / 1e9 / (avg_us / 1e6);

    printf("=== Results ===\n");
    printf("  GPU event (kernel time):   %.2f us\n", avg_us);
    printf("  Host wall (batch avg):     %.2f us\n", wall_us);
    printf("  Empty kernel (baseline):   %.2f us\n", empty_avg_us);
    printf("  Kernel compute only:       %.2f us (= gpu_event - empty)\n", avg_us - empty_avg_us);
    printf("  Effective BW:              %.1f GB/s\n", bw_gbps);
    printf("  Data: %.1f KB read + %.1f KB write\n",
           bytes_read / 1024, bytes_write / 1024);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
    return 0;
}
