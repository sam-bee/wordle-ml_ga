#include "model/policy.hpp"
#include "test_support.hpp"

#include "golden_cases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
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
    auto parameters = ReadFloats("weights.f32le", model::weights::kCount);
    const auto golden = ReadFloats("golden-vectors.f32le", kGoldenValueCount);

    float *device_weights = nullptr;
    model::Input *device_input = nullptr;
    model::Workspace *workspace = nullptr;
    float *device_logits = nullptr;
    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    CheckCuda(cudaMalloc(&device_weights, parameters.size() * sizeof(float)), "cudaMalloc(weights)");
    CheckCuda(cudaMalloc(&device_input, sizeof(model::Input)), "cudaMalloc(input)");
    CheckCuda(cudaMalloc(&workspace, sizeof(model::Workspace)), "cudaMalloc(workspace)");
    CheckCuda(cudaMalloc(&device_logits, model::kNumActions * sizeof(float)), "cudaMalloc(logits)");
    CheckCuda(cudaMemcpyAsync(device_weights, parameters.data(), parameters.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(weights)");
    // Poison scratch once: inference must initialize every value it reads.
    CheckCuda(cudaMemsetAsync(workspace, 0xff, sizeof(model::Workspace), stream), "cudaMemsetAsync(workspace)");

    std::array<float, model::kNumActions> logits{};
    const auto infer = [&](const model::Input &input) {
        CheckCuda(cudaMemcpyAsync(device_input, &input, sizeof(input), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(input)");
        CheckCuda(model::Forward(device_weights, device_input, workspace, device_logits, stream), "Forward");
        CheckCuda(cudaMemcpyAsync(logits.data(), device_logits, sizeof(logits), cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(logits)");
        CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    };

    // Expected logits were produced by the original GoMLX graph and are immutable
    // test data. This host code only copies inputs and compares CUDA outputs.
    float max_error = 0.0f;
    int case_index = 0;
    for (const auto &reference : kGoldenCases) {
        model::Input input{};
        input.turn = reference.turn;
        std::copy_n(golden.data() + reference.candidates, model::kNumSolutions, input.candidate_mask);
        std::copy_n(golden.data() + reference.stats, model::kCandidateStatsSize, input.candidate_stats);
        std::copy_n(golden.data() + reference.remaining, model::kNumActions, input.remaining_action_mask);
        infer(input);
        for (int action = 0; action < model::kNumActions; ++action) {
            const float expected = golden[reference.logits + action];
            const float error = std::fabs(logits[action] - expected);
            if (!std::isfinite(logits[action]) || error > 1e-3f) {
                std::fprintf(stderr, "Case %d action %d: got %.9g, expected %.9g (error %.9g)\n", case_index, action,
                             logits[action], expected, error);
                return EXIT_FAILURE;
            }
            max_error = std::max(max_error, error);
        }
        ++case_index;
    }

    // The source graph defines a finite forward pass for an empty candidate set.
    model::Input empty{};
    empty.turn = 5;
    infer(empty);
    for (const float logit : logits) {
        Require(std::isfinite(logit), "Empty candidate set produced a non-finite logit");
    }

    // A signed bonus changes only candidate scores. Non-candidate probe words
    // retain their base score; neither sign makes them illegal.
    std::fill(parameters.begin(), parameters.end(), 0.0f);
    std::fill_n(parameters.data() + model::weights::kLogitsBias, model::kNumActions, 0.25f);
    model::Input input{};
    input.candidate_mask[0] = 1.0f;
    input.remaining_action_mask[7] = 1.0f;
    for (const float bonus : {2.0f, -2.0f}) {
        parameters[model::weights::kBonusBias] = bonus;
        CheckCuda(cudaMemcpyAsync(device_weights, parameters.data(), parameters.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(bonus test weights)");
        infer(input);
        for (int action = 0; action < model::kNumActions; ++action) {
            const float expected = action == 7 ? 0.25f + bonus : 0.25f;
            if (logits[action] != expected) {
                std::fprintf(stderr, "Bonus %.2f action %d: got %.9g, expected %.9g\n", bonus, action, logits[action],
                             expected);
                return EXIT_FAILURE;
            }
        }
    }

    Require(model::Forward(nullptr, device_input, workspace, device_logits, stream) == cudaErrorInvalidValue,
            "Null weights were accepted");
    Require(model::Forward(device_weights, nullptr, workspace, device_logits, stream) == cudaErrorInvalidValue,
            "Null input was accepted");
    Require(model::Forward(device_weights, device_input, nullptr, device_logits, stream) == cudaErrorInvalidValue,
            "Null workspace was accepted");
    Require(model::Forward(device_weights, device_input, workspace, nullptr, stream) == cudaErrorInvalidValue,
            "Null output was accepted");

    CheckCuda(cudaFree(device_logits), "cudaFree(logits)");
    CheckCuda(cudaFree(workspace), "cudaFree(workspace)");
    CheckCuda(cudaFree(device_input), "cudaFree(input)");
    CheckCuda(cudaFree(device_weights), "cudaFree(weights)");
    CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::printf("CUDA policy passed: gpu=%s, cases=%d, logits=%d, max_absolute_error=%.9g\n", device.name, case_index,
                case_index * model::kNumActions, max_error);
    return EXIT_SUCCESS;
}
