#include "fitness/evaluator.hpp"
#include "genotype_slab/device.cuh"
#include "genotype_slab/slab.hpp"
#include "model/policy.hpp"
#include "test_support.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fitness = wordle_ga::fitness;
namespace model = wordle_ga::model;
namespace slab = wordle_ga::genotype_slab;
namespace fs = std::filesystem;

namespace {

void Require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

void Check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void AllocateBiasGenome(slab::DeviceView view, const int *actions, slab::Slot *slot) {
    if (threadIdx.x == 0)
        *slot = slab::Allocate(view);
    __syncthreads();
    const slab::Slot selected = *slot;
    float *weights = slab::Parameters(view, selected);
    if (weights == nullptr)
        return;
    for (std::size_t i = threadIdx.x; i < slab::kSlotFloats; i += blockDim.x)
        weights[i] = 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int rank = 0; rank < 6; ++rank)
            weights[model::weights::kLogitsBias + actions[rank]] = static_cast<float>(600.0 - rank);
    }
}

std::array<int, 6> TrainingActions(const fitness::Vocabulary &vocabulary) {
    std::array<int, 6> actions{};
    for (int i = 0; i < 6; ++i)
        actions[i] = vocabulary.solution_actions[vocabulary.training_solutions[i]];
    return actions;
}

std::array<int, 6> ProbeActions(const fitness::Vocabulary &vocabulary) {
    std::array<int, 6> actions{};
    int found = 0;
    for (int action = 0; action < model::kNumActions && found < 6; ++action) {
        bool is_solution = false;
        for (const fitness::Word &solution : vocabulary.solutions) {
            bool equal = true;
            for (int letter = 0; letter < 5; ++letter)
                equal = equal && vocabulary.actions[action].letters[letter] == solution.letters[letter];
            if (equal) {
                is_solution = true;
                break;
            }
        }
        if (!is_solution)
            actions[found++] = action;
    }
    Require(found == 6, "fewer than six non-solution actions available");
    return actions;
}

void CheckResult(const fitness::Result &result, bool winning) {
    Require(result.games == fitness::kTrainingSolutions, "wrong game count");
    Require(result.invalid_games == 0, "unexpected invalid games");
    Require(result.wins == (winning ? 6u : 0u), "wrong win count");
    Require(result.guess_sum == (winning ? 12639u : 12654u), "wrong guess sum");
    Require(result.score == (winning ? 63291u : 0u), "wrong score");
    for (int i = 0; i < 6; ++i)
        Require(result.solved_in[i] == (winning ? 1u : 0u), "wrong solved-in histogram");
}

void CheckInvalidResult(const fitness::Result &result) {
    Require(result.games == fitness::kTrainingSolutions, "invalid slot wrong game count");
    Require(result.wins == 0 && result.guess_sum == 12654u, "invalid slot was scored as gameplay");
    Require(result.invalid_games == fitness::kTrainingSolutions && result.score == 0,
            "invalid slot result was not marked invalid");
    for (std::uint32_t solved : result.solved_in)
        Require(solved == 0, "invalid slot solved a game");
}

void CopyResults(fitness::Result *device, std::vector<fitness::Result> &results, cudaStream_t stream) {
    Check(cudaMemcpyAsync(results.data(), device, results.size() * sizeof(fitness::Result), cudaMemcpyDeviceToHost,
                          stream),
          "copy evaluator results");
    Check(cudaStreamSynchronize(stream), "copy evaluator results synchronize");
}

void RuntimeRegression(const fitness::Vocabulary &vocabulary) {
    const auto training_actions = TrainingActions(vocabulary);
    const auto probe_actions = ProbeActions(vocabulary);
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    slab::Slab genotypes;
    Check(genotypes.Create(2, stream), "Slab::Create");
    int *device_actions = nullptr;
    slab::Slot *device_slots = nullptr;
    Check(cudaMalloc(&device_actions, 6 * sizeof(int)), "cudaMalloc(actions)");
    Check(cudaMalloc(&device_slots, 2 * sizeof(slab::Slot)), "cudaMalloc(slots)");
    Check(cudaMemcpyAsync(device_actions, training_actions.data(), 6 * sizeof(int), cudaMemcpyHostToDevice, stream),
          "copy training actions");
    AllocateBiasGenome<<<1, 256, 0, stream>>>(genotypes.view(), device_actions, device_slots);
    Check(cudaGetLastError(), "training genome launch");
    Check(cudaMemcpyAsync(device_actions, probe_actions.data(), 6 * sizeof(int), cudaMemcpyHostToDevice, stream),
          "copy probe actions");
    AllocateBiasGenome<<<1, 256, 0, stream>>>(genotypes.view(), device_actions, device_slots + 1);
    Check(cudaGetLastError(), "probe genome launch");
    Check(cudaStreamSynchronize(stream), "genome synchronize");

    std::array<slab::Slot, 3> population{{0, 0, 0}};
    Check(cudaMemcpyAsync(population.data(), device_slots, 2 * sizeof(slab::Slot), cudaMemcpyDeviceToHost, stream),
          "copy allocated slots");
    Check(cudaStreamSynchronize(stream), "slot copy synchronize");
    const std::array<slab::Slot, 3> ordered_population{{population[1], population[0], population[0]}};
    slab::Slot *device_population = nullptr;
    fitness::Result *device_results = nullptr;
    Check(cudaMalloc(&device_population, population.size() * sizeof(slab::Slot)), "cudaMalloc(population)");
    Check(cudaMalloc(&device_results, population.size() * sizeof(fitness::Result)), "cudaMalloc(results)");
    Check(cudaMemcpyAsync(device_population, ordered_population.data(), sizeof(ordered_population),
                          cudaMemcpyHostToDevice, stream),
          "copy population");

    fitness::Evaluator evaluator;
    Check(evaluator.Create(vocabulary, 256, stream), "Evaluator::Create");
    Check(evaluator.Evaluate(genotypes.view(), device_population, 3, device_results, stream), "training Evaluate");
    Check(cudaStreamSynchronize(stream), "training Evaluate synchronize");
    std::vector<fitness::Result> results(3);
    CopyResults(device_results, results, stream);
    CheckResult(results[0], false);
    CheckResult(results[1], true);
    CheckResult(results[2], true);

    const slab::Slot invalid = slab::kInvalidSlot;
    Check(cudaMemcpyAsync(device_population + 1, &invalid, sizeof(invalid), cudaMemcpyHostToDevice, stream),
          "copy invalid population slot");
    Check(evaluator.Evaluate(genotypes.view(), device_population + 1, 1, device_results, stream),
          "invalid-slot Evaluate");
    Check(cudaStreamSynchronize(stream), "invalid-slot Evaluate synchronize");
    CopyResults(device_results, results, stream);
    CheckInvalidResult(results[0]);
    Check(cudaMemcpyAsync(device_population, ordered_population.data(), sizeof(ordered_population),
                          cudaMemcpyHostToDevice, stream),
          "restore population");
    Check(evaluator.Destroy(), "first evaluator Destroy");
    Check(evaluator.Create(vocabulary, 512, stream), "second evaluator Create");
    Check(evaluator.Evaluate(genotypes.view(), device_population, 3, device_results, stream),
          "batch-invariance Evaluate");
    Check(cudaStreamSynchronize(stream), "batch-invariance Evaluate synchronize");
    CopyResults(device_results, results, stream);
    CheckResult(results[0], false);
    CheckResult(results[1], true);
    CheckResult(results[2], true);

    Check(evaluator.Destroy(), "Evaluator::Destroy");
    Check(cudaFree(device_results), "free results");
    Check(cudaFree(device_population), "free population");
    Check(cudaFree(device_slots), "free slots");
    Check(cudaFree(device_actions), "free actions");
    Check(genotypes.Destroy(), "Slab::Destroy");
    Check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

void InvalidAndLoaderChecks(const fitness::Vocabulary &vocabulary) {
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "invalid-check cudaStreamCreateWithFlags");
    fitness::Evaluator evaluator;
    Require(evaluator.Create(vocabulary, 0, stream) == cudaErrorInvalidValue, "Create(0) accepted");
    Require(evaluator.Create(vocabulary, 65536, stream) == cudaErrorInvalidValue, "Create(65536) accepted");
    Check(evaluator.Create(vocabulary, 256, stream), "valid evaluator Create");
    Require(evaluator.Create(vocabulary, 256, stream) == cudaErrorInvalidValue, "repeated Create accepted");
    Require(evaluator.Evaluate({}, nullptr, 0, nullptr) == cudaErrorInvalidValue, "invalid Evaluate accepted");
    Check(evaluator.Destroy(), "first evaluator Destroy");
    Check(evaluator.Destroy(), "idempotent evaluator Destroy");
    Check(evaluator.Create(vocabulary, 256, stream), "recreate evaluator");
    Check(evaluator.Destroy(), "recreated evaluator Destroy");
    Check(cudaStreamDestroy(stream), "invalid-check cudaStreamDestroy");

    std::array<char, 64> temporary_name{};
    std::snprintf(temporary_name.data(), temporary_name.size(), "%s/wordle-ga-vocabulary-XXXXXX",
                  fs::temp_directory_path().c_str());
    const char *created = mkdtemp(temporary_name.data());
    Require(created != nullptr, "cannot create temporary vocabulary directory");
    const fs::path temporary(created);
    for (const char *file : {"wordlist-action-space-4739.csv", "wordlist-valid-solutions-all-2309.csv",
                             "wordlist-valid-solutions-train-2109.csv"})
        fs::copy_file(fs::path(WORD_DATA_DIR) / file, temporary / file);
    Require(fitness::LoadVocabulary(temporary.string()).training_solutions.size() == fitness::kTrainingSolutions,
            "three-file vocabulary load failed");
    {
        std::ofstream broken(temporary / "wordlist-valid-solutions-train-2109.csv", std::ios::trunc);
        broken << "ZZZZZ\n";
    }
    bool rejected = false;
    try {
        (void)fitness::LoadVocabulary(temporary.string());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    Require(rejected, "perturbed training vocabulary was accepted");
    fs::remove_all(temporary);
}

} // namespace

int main() {
    SelectTestGpu();
    const auto vocabulary = fitness::LoadVocabulary(WORD_DATA_DIR);
    InvalidAndLoaderChecks(vocabulary);
    RuntimeRegression(vocabulary);
    std::puts("CUDA fitness runtime regression passed");
    return EXIT_SUCCESS;
}
