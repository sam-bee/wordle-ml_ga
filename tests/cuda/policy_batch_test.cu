#include "model/policy.hpp"
#include "test_support.hpp"

#include "golden_cases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace model = wordle_ga::model;

namespace {

std::vector<float> ReadFloats(const char *filename, std::size_t count) {
    const auto path = std::string(POLICY_FIXTURE_DIR) + "/" + filename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(float))) {
        std::fprintf(stderr, "Missing or incorrectly sized fixture: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::vector<float> values(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
        std::fprintf(stderr, "Could not read fixture: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return values;
}

void Require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const auto device = SelectTestGpu();
    constexpr int kCount = static_cast<int>(std::size(kGoldenCases));
    const auto golden = ReadFloats("golden-vectors.f32le", kGoldenValueCount);

    std::array<model::Input, kCount> inputs{};
    for (int i = 0; i < kCount; ++i) {
        inputs[i].turn = kGoldenCases[i].turn;
        std::copy_n(golden.data() + kGoldenCases[i].candidates, model::kNumSolutions, inputs[i].candidate_mask);
        std::copy_n(golden.data() + kGoldenCases[i].stats, model::kCandidateStatsSize, inputs[i].candidate_stats);
        std::copy_n(golden.data() + kGoldenCases[i].remaining, model::kNumActions, inputs[i].remaining_action_mask);
    }

    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    float *weights = nullptr;
    model::Input *device_inputs = nullptr;
    model::Workspace *workspaces = nullptr;
    float *logits = nullptr;
    const float **parameters = nullptr;
    CheckCuda(cudaMalloc(&weights, model::weights::kCount * sizeof(float)), "cudaMalloc(weights)");
    CheckCuda(cudaMalloc(&device_inputs, sizeof(inputs)), "cudaMalloc(inputs)");
    CheckCuda(cudaMalloc(&workspaces, kCount * sizeof(model::Workspace)), "cudaMalloc(workspaces)");
    CheckCuda(cudaMalloc(&logits, kCount * model::kNumActions * sizeof(float)), "cudaMalloc(logits)");
    CheckCuda(cudaMalloc(&parameters, kCount * sizeof(float *)), "cudaMalloc(parameters)");

    const auto run = [&](const std::vector<float> &host_weights, const std::array<const float *, kCount> &host_ptrs,
                         const std::array<int, kCount> *active, unsigned char poison_byte,
                         bool poison_workspace = true) {
        CheckCuda(cudaMemcpyAsync(device_inputs, inputs.data(), sizeof(inputs), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(inputs)");
        CheckCuda(cudaMemcpyAsync(weights, host_weights.data(), host_weights.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(weights)");
        CheckCuda(cudaMemcpyAsync(parameters, host_ptrs.data(), sizeof(host_ptrs), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(parameters)");
        if (poison_workspace) {
            CheckCuda(cudaMemsetAsync(workspaces, 0xff, kCount * sizeof(model::Workspace), stream),
                      "cudaMemsetAsync(workspaces)");
        }
        CheckCuda(cudaMemsetAsync(logits, poison_byte, kCount * model::kNumActions * sizeof(float), stream),
                  "cudaMemsetAsync(logits)");
        int *device_active = nullptr;
        if (active != nullptr) {
            CheckCuda(cudaMallocAsync(&device_active, kCount * sizeof(int), stream), "cudaMallocAsync(active)");
            CheckCuda(
                cudaMemcpyAsync(device_active, active->data(), kCount * sizeof(int), cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(active)");
        }
        CheckCuda(model::ForwardBatch(parameters, device_inputs, workspaces, logits, kCount, device_active, stream),
                  "ForwardBatch");
        if (device_active != nullptr)
            CheckCuda(cudaFreeAsync(device_active, stream), "cudaFreeAsync(active)");
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    };

    std::vector<float> trained = ReadFloats("weights.f32le", model::weights::kCount);
    std::array<const float *, kCount> same{};
    for (auto &pointer : same)
        pointer = weights;
    run(trained, same, nullptr, 0);
    std::vector<float> output(kCount * model::kNumActions);
    CheckCuda(cudaMemcpy(output.data(), logits, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy(logits)");
    for (int i = 0; i < kCount; ++i) {
        for (int action = 0; action < model::kNumActions; ++action) {
            const float expected = golden[kGoldenCases[i].logits + action];
            Require(std::isfinite(output[i * model::kNumActions + action]) &&
                        std::fabs(output[i * model::kNumActions + action] - expected) <= 1e-3f,
                    "ForwardBatch golden mismatch");
        }
    }

    // The batch API rejects counts outside its fixed launch range before touching device memory.
    Require(model::ForwardBatch(nullptr, device_inputs, workspaces, logits, 1, nullptr, stream) ==
                cudaErrorInvalidValue,
            "ForwardBatch accepted null parameters");
    Require(model::ForwardBatch(parameters, nullptr, workspaces, logits, 1, nullptr, stream) == cudaErrorInvalidValue,
            "ForwardBatch accepted null inputs");
    Require(model::ForwardBatch(parameters, device_inputs, nullptr, logits, 1, nullptr, stream) ==
                cudaErrorInvalidValue,
            "ForwardBatch accepted null workspaces");
    Require(model::ForwardBatch(parameters, device_inputs, workspaces, nullptr, 1, nullptr, stream) ==
                cudaErrorInvalidValue,
            "ForwardBatch accepted null logits");
    Require(model::ForwardBatch(parameters, device_inputs, workspaces, logits, 0, nullptr, stream) ==
                cudaErrorInvalidValue,
            "ForwardBatch accepted a zero count");
    Require(model::ForwardBatch(parameters, device_inputs, workspaces, logits, 65536, nullptr, stream) ==
                cudaErrorInvalidValue,
            "ForwardBatch accepted a count above 65535");

    // Reuse scratch without clearing it: a forward pass must overwrite every value it reads.
    run(trained, same, nullptr, 0, false);
    CheckCuda(cudaMemcpy(output.data(), logits, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy(reused logits)");
    for (int i = 0; i < kCount; ++i)
        for (int action = 0; action < model::kNumActions; ++action)
            Require(std::isfinite(output[i * model::kNumActions + action]) &&
                        std::fabs(output[i * model::kNumActions + action] - golden[kGoldenCases[i].logits + action]) <=
                            1e-3f,
                    "ForwardBatch scratch reuse mismatch");

    // Distinct device pointers: zero networks with controlled per-action biases.
    std::vector<float> weights2(model::weights::kCount, 0.0f);
    for (int action = 0; action < model::kNumActions; ++action)
        weights2[model::weights::kLogitsBias + action] = static_cast<float>(action) * 0.001f - 1.0f;
    float *weights2_device = nullptr;
    CheckCuda(cudaMalloc(&weights2_device, weights2.size() * sizeof(float)), "cudaMalloc(weights2)");
    CheckCuda(cudaMemcpyAsync(weights2_device, weights2.data(), weights2.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream),
              "cudaMemcpyAsync(weights2)");
    std::array<const float *, kCount> varied{};
    for (int i = 0; i < kCount; ++i)
        varied[i] = (i & 1) ? weights2_device : weights;
    run(trained, varied, nullptr, 0);
    CheckCuda(cudaMemcpy(output.data(), logits, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy(varied logits)");
    for (int i = 0; i < kCount; ++i)
        for (int action = 0; action < model::kNumActions; ++action) {
            const float expected =
                (i & 1) ? weights2[model::weights::kLogitsBias + action] : golden[kGoldenCases[i].logits + action];
            Require(std::isfinite(output[i * model::kNumActions + action]) &&
                        std::fabs(output[i * model::kNumActions + action] - expected) <= 1e-3f,
                    "ForwardBatch varied pointer mismatch");
        }

    // Inactive lanes retain their poison exactly, including their workspace.
    std::array<int, kCount> active{};
    for (int i = 0; i < kCount; i += 2)
        active[i] = 1;
    for (int i = 1; i < kCount; i += 2)
        std::memset(&inputs[i], 0xA5, sizeof(inputs[i]));
    std::array<const float *, kCount> active_ptrs{};
    for (int i = 0; i < kCount; i += 2)
        active_ptrs[i] = weights2_device;
    run(weights2, active_ptrs, &active, 0xA5);
    CheckCuda(cudaMemcpy(output.data(), logits, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy(active logits)");
    const float poison = [] {
        std::uint32_t bits = 0xA5A5A5A5u;
        float value;
        std::memcpy(&value, &bits, 4);
        return value;
    }();
    for (int i = 1; i < kCount; i += 2)
        for (int action = 0; action < model::kNumActions; ++action)
            Require(std::memcmp(&output[i * model::kNumActions + action], &poison, sizeof(float)) == 0,
                    "Inactive ForwardBatch lane was written");
    for (int i = 0; i < kCount; i += 2)
        for (int action = 0; action < model::kNumActions; ++action)
            Require(std::isfinite(output[i * model::kNumActions + action]) &&
                        std::fabs(output[i * model::kNumActions + action] -
                                  weights2[model::weights::kLogitsBias + action]) <= 1e-6f,
                    "Active ForwardBatch lane mismatch");

    std::vector<std::uint8_t> workspace_bytes(kCount * sizeof(model::Workspace));
    CheckCuda(cudaMemcpy(workspace_bytes.data(), workspaces, workspace_bytes.size(), cudaMemcpyDeviceToHost),
              "cudaMemcpy(active workspaces)");
    for (int i = 1; i < kCount; i += 2)
        for (std::size_t byte = 0; byte < sizeof(model::Workspace); ++byte)
            Require(workspace_bytes[static_cast<std::size_t>(i) * sizeof(model::Workspace) + byte] == 0xff,
                    "Inactive ForwardBatch workspace was written");

    CheckCuda(cudaFree(weights2_device), "cudaFree(weights2)");
    CheckCuda(cudaFree(parameters), "cudaFree(parameters)");
    CheckCuda(cudaFree(logits), "cudaFree(logits)");
    CheckCuda(cudaFree(workspaces), "cudaFree(workspaces)");
    CheckCuda(cudaFree(device_inputs), "cudaFree(inputs)");
    CheckCuda(cudaFree(weights), "cudaFree(weights)");
    CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::printf("CUDA policy batch passed: gpu=%s, cases=%d\n", device.name, kCount);
    return EXIT_SUCCESS;
}
