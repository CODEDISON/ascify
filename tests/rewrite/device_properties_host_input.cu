// Deliberately no kernel: property tokens must request their own compat include.
cudaError_t query_vector_capacity(int device, int *capacity) {
  cudaDeviceProp properties;
  const cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess && capacity != nullptr)
    *capacity = properties.multiProcessorCount;
  return status;
}

cudaError_t query_launch_limits(int device, int *limits) {
  cudaDeviceProp properties;
  const cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess && limits != nullptr) {
    limits[0] = properties.maxThreadsPerBlock;
    for (int axis = 0; axis != 3; ++axis) limits[axis + 1] = properties.maxGridSize[axis];
  }
  return status;
}
