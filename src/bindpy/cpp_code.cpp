// cpp_code.cpp
#include "cpp_code.h"
#include <cuda_runtime.h>

extern "C" void launch_add_vectors(float *a, float *b, float *c, int n);

std::vector<float> VectorAdder::add_vectors(const std::vector<float> &a,
                                            const std::vector<float> &b) {
  int n = a.size();
  std::vector<float> result(n);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  launch_add_vectors(d_a, d_b, d_c, n);

  cudaMemcpy(result.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return result;
}

