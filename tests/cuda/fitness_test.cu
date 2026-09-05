#include "fitness/evaluator.hpp"
#include "fitness/game.cuh"
#include "fitness/vocabulary.hpp"
#include "genotype_slab/device.cuh"
#include "genotype_slab/slab.hpp"
#include "model/policy.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

namespace fitness = wordle_ga::fitness;
namespace model = wordle_ga::model;
namespace slab = wordle_ga::genotype_slab;

namespace {

fitness::Word Word(const char *text) {
    fitness::Word result{};
    for (int i = 0; i < 5; ++i)
        result.letters[i] = static_cast<std::uint8_t>(text[i] - 'A');
    return result;
}

__global__ void FeedbackProbe(fitness::Word guess, fitness::Word answer, std::uint8_t *out) {
    if (threadIdx.x == 0)
        *out = fitness::Feedback(guess, answer);
}

__global__ void AllocateSlot(slab::DeviceView view, slab::Slot *slot) {
    if (threadIdx.x != 0)
        return;
    const slab::Slot allocated = slab::Allocate(view);
    *slot = allocated;
}

std::vector<float> ReadWeights() {
    const std::string path = std::string(POLICY_FIXTURE_DIR) + "/weights.f32le";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto size = static_cast<std::size_t>(file.tellg());
    if (!file || size != model::weights::kCount * sizeof(float)) {
        std::fprintf(stderr, "Missing policy fixture: %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    std::vector<float> result(model::weights::kCount);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(size));
    return result;
}

void Check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void BasicTests(const fitness::Vocabulary &vocabulary) {
    std::uint8_t *device_feedback = nullptr;
    Check(cudaMalloc(&device_feedback, sizeof(std::uint8_t)), "cudaMalloc(feedback probe)");
    const std::array<std::tuple<const char *, const char *, int>, 3> probes{{
        {"AAAAB", "AACAA", 71},  // three greens and one yellow after green consumption
        {"ABBEY", "EERIE", 27},  // the E at position three is yellow
        {"EERIE", "EERIE", 242}, // five greens, 2*(1+3+9+27+81)
    }};
    for (const auto &[guess, answer, expected] : probes) {
        FeedbackProbe<<<1, 1>>>(Word(guess), Word(answer), device_feedback);
        Check(cudaGetLastError(), "FeedbackProbe launch");
        std::uint8_t actual = 0;
        Check(cudaMemcpy(&actual, device_feedback, sizeof(actual), cudaMemcpyDeviceToHost),
              "cudaMemcpy(feedback probe)");
        if (actual != expected) {
            std::fprintf(stderr, "Feedback %s/%s: got %u expected %d\n", guess, answer, actual, expected);
            std::exit(EXIT_FAILURE);
        }
    }

    const std::size_t feedback_count = static_cast<std::size_t>(model::kNumActions) * model::kNumSolutions;
    fitness::Word *actions = nullptr;
    fitness::Word *solutions = nullptr;
    std::uint8_t *table = nullptr;
    Check(cudaMalloc(&actions, sizeof(vocabulary.actions)), "cudaMalloc(actions)");
    Check(cudaMalloc(&solutions, sizeof(vocabulary.solutions)), "cudaMalloc(solutions)");
    Check(cudaMalloc(&table, feedback_count), "cudaMalloc(feedback table)");
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    Check(cudaMemcpy(actions, vocabulary.actions.data(), sizeof(vocabulary.actions), cudaMemcpyHostToDevice),
          "copy actions");
    Check(cudaMemcpy(solutions, vocabulary.solutions.data(), sizeof(vocabulary.solutions), cudaMemcpyHostToDevice),
          "copy solutions");
    Check(fitness::BuildFeedback(actions, solutions, table, stream), "BuildFeedback");
    Check(cudaStreamSynchronize(stream), "BuildFeedback synchronize");
    const std::size_t first_solution_action =
        static_cast<std::size_t>(vocabulary.solution_actions[0]) * model::kNumSolutions;
    const std::array<std::size_t, 3> offsets{{first_solution_action, first_solution_action + 1, feedback_count - 1}};
    // ABACK/ABACK, ABACK/ABASE, ZULUS/ZONAL (green Z and yellow L).
    const std::array<std::uint8_t, 3> expected_table{{242, 26, 11}};
    for (int i = 0; i < 3; ++i) {
        std::uint8_t got = 0;
        Check(cudaMemcpy(&got, table + offsets[i], 1, cudaMemcpyDeviceToHost), "copy table probe");
        if (got != expected_table[i]) {
            std::fprintf(stderr, "Feedback table probe %d: got %u, expected %u\n", i, got, expected_table[i]);
            std::exit(EXIT_FAILURE);
        }
    }
    Check(cudaFree(table), "cudaFree(table)");
    Check(cudaFree(solutions), "cudaFree(solutions)");
    Check(cudaFree(actions), "cudaFree(actions)");
    Check(cudaFree(device_feedback), "cudaFree(feedback probe)");
    Check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

int FullTraining(const fitness::Vocabulary &vocabulary) {
    const auto weights = ReadWeights();
    slab::Slab genotype;
    Check(genotype.Create(1), "Slab::Create");
    slab::Slot *population = nullptr;
    slab::Slot *allocated_slot = nullptr;
    fitness::Result *result = nullptr;
    Check(cudaMalloc(&population, sizeof(slab::Slot)), "cudaMalloc(population)");
    Check(cudaMalloc(&allocated_slot, sizeof(slab::Slot)), "cudaMalloc(allocated slot)");
    Check(cudaMalloc(&result, sizeof(fitness::Result)), "cudaMalloc(result)");
    AllocateSlot<<<1, 1>>>(genotype.view(), allocated_slot);
    Check(cudaGetLastError(), "AllocateSlot launch");
    Check(cudaDeviceSynchronize(), "AllocateSlot synchronize");
    slab::Slot slot = slab::kInvalidSlot;
    Check(cudaMemcpy(&slot, allocated_slot, sizeof(slot), cudaMemcpyDeviceToHost), "copy allocated slot");
    if (slot == slab::kInvalidSlot)
        return EXIT_FAILURE;
    Check(cudaMemcpy(genotype.view().genotypes + static_cast<std::size_t>(slot) * slab::kSlotFloats, weights.data(),
                     weights.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy training weights");
    Check(cudaMemcpy(population, &slot, sizeof(slot), cudaMemcpyHostToDevice), "copy population");
    fitness::Evaluator evaluator;
    Check(evaluator.Create(vocabulary, 256), "Evaluator::Create");
    cudaEvent_t begin = nullptr, end = nullptr;
    Check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
    Check(cudaEventCreate(&end), "cudaEventCreate(end)");
    Check(cudaEventRecord(begin), "cudaEventRecord(begin)");
    Check(evaluator.Evaluate(genotype.view(), population, 1, result), "Evaluator::Evaluate");
    Check(cudaEventRecord(end), "cudaEventRecord(end)");
    Check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float milliseconds = 0.0f;
    Check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
    fitness::Result host{};
    Check(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost), "copy result");
    std::printf("full-training: games=%u wins=%u guess_sum=%u invalid=%u score=%u time_ms=%.3f allocated_bytes=%zu\n",
                host.games, host.wins, host.guess_sum, host.invalid_games, host.score, milliseconds,
                evaluator.allocated_bytes());
    if (host.games != fitness::kTrainingSolutions || host.invalid_games != 0 || host.wins > host.games ||
        host.guess_sum < host.wins || host.guess_sum > 6u * host.games)
        return EXIT_FAILURE;
    Check(cudaEventDestroy(end), "cudaEventDestroy(end)");
    Check(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
    Check(evaluator.Destroy(), "Evaluator::Destroy");
    Check(cudaFree(result), "cudaFree(result)");
    Check(cudaFree(allocated_slot), "cudaFree(allocated slot)");
    Check(cudaFree(population), "cudaFree(population)");
    Check(genotype.Destroy(), "Slab::Destroy");
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    SelectTestGpu();
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--full-training")) {
        std::fprintf(stderr, "Usage: fitness_test [--full-training]\n");
        return EXIT_FAILURE;
    }
    const auto vocabulary = fitness::LoadVocabulary(WORD_DATA_DIR);
    BasicTests(vocabulary);
    if (argc > 1 && std::string(argv[1]) == "--full-training")
        return FullTraining(vocabulary);
    std::puts("CUDA fitness primitives passed");
    return EXIT_SUCCESS;
}
