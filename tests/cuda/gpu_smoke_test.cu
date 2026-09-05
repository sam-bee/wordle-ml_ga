#include "test_support.hpp"

namespace {

__global__ void AddOne(int *value) { *value += 1; }

} // namespace

int main() {
    const auto properties = SelectTestGpu();

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
