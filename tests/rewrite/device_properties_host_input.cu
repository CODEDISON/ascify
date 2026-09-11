// Deliberately no kernel: property tokens must request their own compat include.
cudaError_t query_vector_capacity(int device, int *capacity) {
  cudaDeviceProp properties;
  const cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess && capacity != nullptr)
    *capacity = properties.multiProcessorCount;
  return status;
}
