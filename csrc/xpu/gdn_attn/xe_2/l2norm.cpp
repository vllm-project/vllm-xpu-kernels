#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "l2norm_kernel.hpp"
#include "l2norm.h"

void l2norm(
    sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k) {
  gdn::l2norm_impl(queue, q, k);
}
