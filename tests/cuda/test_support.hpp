#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void CheckCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline cudaDeviceProp SelectTestGpu() {
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count != 1) {
        std::fprintf(stderr, "Expected exactly one visible GPU, found %d. Check NVIDIA_GPU_DEVICE_ID in .env.\n",
                     count);
        std::exit(EXIT_FAILURE);
    }
    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    const bool approved = std::strcmp(properties.name, "NVIDIA GeForce RTX 5070 Ti") == 0 ||
                          std::strcmp(properties.name, "NVIDIA GeForce RTX 5050") == 0 ||
                          std::strcmp(properties.name, "NVIDIA GeForce RTX 5050 Laptop GPU") == 0;
    if (!approved || properties.major != 12 || properties.minor != 0) {
        std::fprintf(stderr, "Expected an RTX 5070 Ti or RTX 5050 with compute capability 12.0, found %s (%d.%d).\n",
                     properties.name, properties.major, properties.minor);
        std::exit(EXIT_FAILURE);
    }
    CheckCuda(cudaSetDevice(0), "cudaSetDevice");
    return properties;
}
