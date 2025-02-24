__global__ void add_vectors(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

extern "C" void launch_add_vectors(float *a, float *b, float *c, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  add_vectors<<<grid_size, block_size>>>(a, b, c, n);
}
