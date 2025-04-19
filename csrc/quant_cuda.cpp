#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


void vecquant3matmul_cherryq_cuda(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size, bool fp16
);

void vecquant4matmul_cherryq_cuda(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size
);

void Dequantize3Normal(
  torch::Tensor qweight,
  torch::Tensor scaling_factors,
  torch::Tensor out,
  int height, int cherry_size, int group_size
);

void Dequantize4Normal(
  torch::Tensor qweight,
  torch::Tensor scaling_factors,
  torch::Tensor out,
  int height, int cherry_size, int group_size
);


void vecquant3matmul_cherryq(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size, bool fp16
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
    vecquant3matmul_cherryq_cuda(
        vec, qweight, cherry_weight, cherry_indices, scaling_factors, mul, group_size, fp16
    );
}

void vecquant4matmul_cherryq(
  torch::Tensor vec,
  torch::Tensor qweight,
  torch::Tensor cherry_weight,
  torch::Tensor cherry_indices,
  torch::Tensor scaling_factors,
  torch::Tensor mul,
  int group_size
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
    vecquant4matmul_cherryq_cuda(
        vec, qweight, cherry_weight, cherry_indices, scaling_factors, mul, group_size
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vecquant3matmul_cherryq", &vecquant3matmul_cherryq, "a 3-bit version");
    m.def("vecquant4matmul_cherryq", &vecquant4matmul_cherryq, "a 4-bit version");
    m.def("Dequantize3Normal", &Dequantize3Normal, "Dequantize3Normal");
    m.def("Dequantize4Normal", &Dequantize4Normal, "Dequantize4Normal");
}
