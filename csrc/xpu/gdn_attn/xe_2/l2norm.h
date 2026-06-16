#include <torch/all.h>

void l2norm(sycl::queue& queue, const torch::Tensor& q, const torch::Tensor& k);
