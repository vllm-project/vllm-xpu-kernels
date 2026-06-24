#include "onednn_runtime.h"

std::string get_onednn_version() {
  std::ostringstream ss;
  {
    const dnnl_version_t* ver = dnnl_version();
    ss << ver->major << '.' << ver->minor << '.' << ver->patch << "."
       << ver->hash;
  }
  return ss.str();
}

namespace oneDNN {

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager instance;
  return instance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager instance;
  return instance;
}

}  // namespace oneDNN
