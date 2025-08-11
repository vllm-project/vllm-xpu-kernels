

#include <torch/all.h>
#include <iostream>
#include <syclcompat.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"

using namespace cute;

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(Element* block, std::size_t size, uint64_t seed = 2023) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(block, size, seed,
                                                     scope_max, scope_min, 0);

  syclcompat::wait();
  return true;
}

template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block,
                      uint64_t seed = 2023) {
  return initialize_block<Element>(block.get(), block.size(), seed);
}

template <typename T1, typename T2>
void initialize_mixed_dtype_block(
    cutlass::DeviceAllocation<T1>& block_device,
    cutlass::DeviceAllocation<T2>& block_device_dq, uint64_t seed) {
  static_assert(cute::sizeof_bits_v<T2> >= 8);

  std::ranlux24_base rng(std::random_device{}());
  rng.seed(seed);

  int bits_input = cute::sizeof_bits_v<T1>;
  T1 scope_max, scope_min;
  if (bits_input == 1) {
    scope_max = T1(2);
    scope_min = T1(0);
  } else if (bits_input <= 8) {
    scope_max = T1(2);
    scope_min = T1(-2);
  } else {
    scope_max = T1(8);
    scope_min = T1(-8);
  }

  std::uniform_int_distribution<> dist(scope_min, scope_max);

  if constexpr (cute::sizeof_bits_v<T1> >= 8) {
    auto block_host = std::vector<T1>(block_device.size());
    auto block_host_dq = std::vector<T2>(block_device.size());
    for (int i = 0; i < block_host.size(); ++i) {
      block_host[i] = static_cast<T1>(dist(rng));
      block_host_dq[i] = static_cast<T2>(block_host[i]);
    }

    block_device.copy_from_host(block_host.data());
    block_device_dq.copy_from_host(block_host_dq.data());
  } else {
    static constexpr auto array_size = 1024;

    cute::array_subbyte<T1, array_size> block_host{};
    auto block_host_dq = std::vector<T2>(array_size);

    for (int i = 0; i < block_host.size(); ++i) {
      block_host[i] = static_cast<T1>(dist(rng));
      block_host_dq[i] = static_cast<T2>(block_host[i].get());
    }

    static constexpr auto elements_per_byte =
        cute::sizeof_bits_v<int8_t> / cute::sizeof_bits_v<T1>;

    int loop_cnt = block_device.size() / array_size;
    for (int i = 0; i < loop_cnt; i++) {
      cutlass::device_memory::copy_to_device(
          block_device.get() + (i * array_size) / elements_per_byte,
          raw_pointer_cast(block_host.begin()), array_size);
      cutlass::device_memory::copy_to_device(
          block_device_dq.get() + i * array_size, block_host_dq.data(),
          array_size);
    }

    auto tail_size = block_device.size() % array_size;
    if (tail_size) {
      cutlass::device_memory::copy_to_device(
          block_device.get() + (loop_cnt * array_size) / elements_per_byte,
          raw_pointer_cast(block_host.begin()), tail_size);
      cutlass::device_memory::copy_to_device(
          block_device_dq.get() + loop_cnt * array_size, block_host_dq.data(),
          tail_size);
    }
  }
}

template <typename T>
inline bool is_close(T a, T b, float atol, float rtol) {
  return std::abs((float)a - (float)b) <= atol + rtol * std::abs((float)b);
}

// TODO(Codeplay): use on device initialisation for this
template <typename T>
inline void random_fill(T* src, int seed, size_t N, float max, float min) {
  if constexpr (std::is_same_v<T, float> ||
                std::is_same_v<T, cute::bfloat16_t> ||
                std::is_same_v<T, cute::half_t>) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min, max);
    auto buff = std::vector<T>(N);

    for (size_t i = 0; i < N; ++i) {
      buff[i] = (T)(dis(gen));
    }
    syclcompat::memcpy<T>(src, buff.data(), N);
    syclcompat::wait();
  } else {
    assert(0 & "Not supported dtype");
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {
  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options()
      : help(false),
        error(false),
        m(5120),
        n(4096),
        k(4096),
        l(1),
        iterations(20),
        alpha(1.f),
        beta(0.f) {}

  // Parses the command line
  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG GEMM Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the L extent (batch count) of "
           "the GEMM\n"
        << "  --alpha=<s32>               Epilogue scalar alpha\n"
        << "  --beta=<s32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm>
struct ExampleRunner {
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput>
      block_ref_D;  // Reference GEMM result for verification

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha,
              ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
        {M, N, K}, alpha, ref_A, cutlass::ComplexTransform::kNone, ref_B,
        cutlass::ComplexTransform::kNone, beta, ref_C, ref_D,
        ElementAccumulator(0),
        L,      // batch_count
        M * K,  // batch_stride_A
        K * N,  // batch_stride_B
        M * N,  // batch_stride_C
        M * N   // batch_stride_D
    );

    // CUTLASS on SYCL uses the compatibility library syclcompat for e.g.
    // default in-order queue
    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(
        block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    // Complete the stride by combining static layout info (StrideA) with
    // runtime size info (M,K,L)
    stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  cutlass::Status run(const Options& options,
                      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size =
        ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {block_A.get(), stride_A, block_B.get(), stride_B},
        {{options.alpha, options.beta},
         block_C.get(),
         stride_C,
         block_D.get(),
         stride_D},
        hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n
                << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed) return cutlass::Status::kErrorInternal;

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      syclcompat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops =
          (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x'
                << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n",
             tflops / cute_time, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

void cutlass_sycl_demo(torch::Tensor& a) {
  //
  // Parse options
  //
  //
  std::cout << a.sizes() << std::endl;

  Options options;

  /* options.parse(argc, argv); */

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  bool passed;

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA =
      bfloat16_t;  // <- data type of elements in input matrix A
  using ElementInputB =
      bfloat16_t;               // <- data type of elements in input matrix B
  using ElementOutput = float;  // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // The 2D block copy operations used for the A and B matrices
  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  // A TiledMMA struct defines a tiling of an MMA atom over M, N and K,
  // combining both additional hardware (sub-groups for Intel BMG) and
  // iterations by each sub-group.
  //
  // The TiledMMAHelper struct defines a specific TiledMMA for a given MMA atom
  // (XE_8x16x16_F32BF16BF16F32_TT), TileShape (<256, 256, 32>) and sub-group
  // layout (8x4x1). The TiledMMA constructed using TiledMMAHelper has the
  // property that each sub-group operates on a single contiguous chunk of the
  // work-group TileShape. For this configuration, this implies that each
  // sub-group operates on a contiguous 32x64x32 chunk (4x4x2 iterations). See
  // 0t_mma_atom.md#TiledMMAs for more info. Sub-groups are arranged row-major
  // (stride 4,1,0) for performance reasons.
  using TiledMma =  // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
      typename TiledMMAHelper<
          MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
          Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // For Intel BMG, PipelineStages defines how many k-blocks ahead to prefetch
  // from A and B.
  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // This is the 'default' epilogue operation (Linear Combination) which
  // performs everything in: (D = alpha * (A*B) + beta * C) aside from the
  // (A*B), which is handled by the GEMM. See 05_bmg_gemm_with_epilogues for
  // more complex epilogue examples.
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator,
      ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  // FusionCallbacks ties the EpilogueOp to an implementation (based on the
  // dispatch policy/architecture) and defines the epilogue arguments.
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape,
      decltype(tile_shape(TiledMma()))>;
  // GEMM Epilogue - loads & stores C/D matrices, performs epilogue operations &
  // load/stores any auxiliary data required
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy, TileShape, ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,  // Converts CUTLASS 2.x to
                                               // CUTLASS 3.x representation
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,  // Converts CUTLASS 2.x to
                                               // CUTLASS 3.x representation
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,  // The copy atom used to load matrix C
      void, void,
      XE_2D_U32x8x16_ST_N,  // The copy atom used to store matrix D
      void, void>;

  // GEMM Mainloop - iteration over blocks in K dimension
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,  // Converts CUTLASS 2.x to
                                               // CUTLASS 3.x representation
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,  // Converts CUTLASS 2.x to
                                               // CUTLASS 3.x representation
      TiledMma, GmemTiledCopyA, void, void, cute::identity,  // A
      GmemTiledCopyB, void, void, cute::identity             // B
      >;

  // Define the whole kernel (mainloop and epilogue)
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Defer global problem shape definition to
                                  // runtime
      CollectiveMainloop, CollectiveEpilogue>;

  // The GemmUniversalAdapter wraps the defined GEMM kernel and handles the
  // launch, and e.g. persistent scratch memory if required.
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));
}
