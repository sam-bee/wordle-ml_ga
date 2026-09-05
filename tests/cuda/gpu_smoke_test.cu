#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void CheckCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

bool IsApprovedGpu(const char *name) {
    return std::strcmp(name, "NVIDIA GeForce RTX 5070 Ti") == 0 || std::strcmp(name, "NVIDIA GeForce RTX 5050") == 0 ||
           std::strcmp(name, "NVIDIA GeForce RTX 5050 Laptop GPU") == 0;
}

__global__ void AddOne(int *value) { *value += 1; }

} // namespace

int main() {
    int device_count = 0;
    CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != 1) {
        std::fprintf(stderr, "Expected exactly one visible GPU, found %d. Check NVIDIA_GPU_DEVICE_ID in .env.\n",
                     device_count);
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    if (!IsApprovedGpu(properties.name) || properties.major != 12 || properties.minor != 0) {
        std::fprintf(stderr, "Expected an RTX 5070 Ti or RTX 5050 with compute capability 12.0, found %s (%d.%d).\n",
                     properties.name, properties.major, properties.minor);
        return EXIT_FAILURE;
    }

    // Validate the device before creating a context or allocating GPU memory.
    CheckCuda(cudaSetDevice(0), "cudaSetDevice");

    int result = 41;
    int *device_value = nullptr;
    CheckCuda(cudaMalloc(&device_value, sizeof(result)), "cudaMalloc");
    CheckCuda(cudaMemcpy(device_value, &result, sizeof(result), cudaMemcpyHostToDevice), "cudaMemcpy(host to device)");
    AddOne<<<1, 1>>>(device_value);
    CheckCuda(cudaGetLastError(), "AddOne kernel launch");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    CheckCuda(cudaMemcpy(&result, device_value, sizeof(result), cudaMemcpyDeviceToHost), "cudaMemcpy(device to host)");
    CheckCuda(cudaFree(device_value), "cudaFree");

    if (result != 42) {
        std::fprintf(stderr, "Unexpected kernel result: %d, expected 42.\n", result);
        return EXIT_FAILURE;
    }

    std::printf("CUDA smoke passed: gpu=%s, compute=%d.%d, memory=%.2f GiB, result=%d\n", properties.name,
                properties.major, properties.minor,
                static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0), result);
    return EXIT_SUCCESS;
}
