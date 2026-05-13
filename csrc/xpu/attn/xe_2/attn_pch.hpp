#pragma once

// Umbrella precompiled-header for the attn_kernels_xe_2 library.
//
// chunk_prefill.hpp and paged_decode.hpp share almost their entire
// transitive header set (cute, the CUTLASS-SYCL collective stack, sycl
// runtime). The 600+ generated TUs all pay that parse cost identically;
// surfacing it through a single PCH means icpx pre-parses once and every
// TU loads the resulting AST snapshot in milliseconds.
//
// PCH compatibility is strict on icpx — every consumer TU must be
// compiled with flags identical to the PCH (target triple, -fsycl,
// -D defines, GRF mode, optimisation level). cmake's
// target_precompile_headers() handles this automatically; downstream
// build systems that bypass cmake (e.g. per-TU dynamic-derivations
// builds in Nix) must propagate the PCH compile command's flags
// verbatim and rewrite only `-o` / dep-tracking flags.

// Pre-declare the SYCL kernel-name tags used inside
// cute/util/compat/memory.hpp's parallel_for<class T> ETS forward
// declarations. cmake injects -fpch-instantiate-templates into the PCH
// compile, which eagerly instantiates sycl::handler::parallel_for at
// PCH-emit time. icpx 2025.3 runs the SYCL kernel-name validator
// against the ETS tag before the implicit namespace-scope declaration
// has propagated, so the eager instantiation fails with "kernel name
// should be forward declarable at namespace scope". Declaring the tags
// here makes the in-body ETS a redeclaration of an already-visible
// class, which the validator accepts. Drop this prelude once icpx
// either delays kernel-name validation past PCH-emit or stops eagerly
// instantiating SYCL kernel templates under -fpch-instantiate-templates.
namespace compat {
namespace detail {
class memcpy_3d_detail;
class compat_memcpy_3d_detail_usmnone;
} // namespace detail
} // namespace compat

#include "chunk_prefill.hpp"
#include "paged_decode.hpp"
