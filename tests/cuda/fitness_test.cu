#include "fitness/evaluator.hpp"
#include "fitness/game.cuh"
#include "fitness/vocabulary.hpp"
#include "genotype_slab/device.cuh"
#include "genotype_slab/slab.hpp"
#include "model/policy.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
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

__global__ void AllocateSlots(slab::DeviceView view, slab::Slot *slots, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        slots[index] = slab::Allocate(view);
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

bool BaselineMatches(const fitness::Result &result) {
    return result.games == 2109u && result.wins == 2101u && result.guess_sum == 7696u && result.invalid_games == 0u &&
           result.score == 22162104u && result.solved_in[0] == 1u && result.solved_in[1] == 59u &&
           result.solved_in[2] == 862u && result.solved_in[3] == 988u && result.solved_in[4] == 155u &&
           result.solved_in[5] == 36u;
}

bool ParsePositiveInt(const char *text, int maximum, int *value) {
    if (text == nullptr || *text == '\0')
        return false;
    for (const char *digit = text; *digit != '\0'; ++digit) {
        if (*digit < '0' || *digit > '9')
            return false;
    }
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed < 1 || parsed > maximum)
        return false;
    *value = static_cast<int>(parsed);
    return true;
}

int FullTraining(const fitness::Vocabulary &vocabulary, int population_count, int repetitions, bool benchmark) {
    const auto weights = ReadWeights();
    slab::Slab genotype;
    Check(genotype.Create(population_count), "Slab::Create");
    slab::Slot *population = nullptr;
    fitness::Result *results = nullptr;
    std::vector<slab::Slot> host_population(static_cast<std::size_t>(population_count));
    Check(cudaMalloc(&population, host_population.size() * sizeof(slab::Slot)), "cudaMalloc(population)");
    Check(cudaMalloc(&results, static_cast<std::size_t>(population_count) * sizeof(fitness::Result)),
          "cudaMalloc(results)");
    const int blocks = (population_count + 255) / 256;
    AllocateSlots<<<blocks, 256>>>(genotype.view(), population, population_count);
    Check(cudaGetLastError(), "AllocateSlots launch");
    Check(cudaDeviceSynchronize(), "AllocateSlots synchronize");
    Check(cudaMemcpy(host_population.data(), population, host_population.size() * sizeof(slab::Slot),
                     cudaMemcpyDeviceToHost),
          "copy allocated slots");
    bool distinct = true;
    for (int i = 0; i < population_count; ++i) {
        if (host_population[static_cast<std::size_t>(i)] == slab::kInvalidSlot) {
            distinct = false;
            break;
        }
        for (int j = 0; j < i; ++j) {
            if (host_population[static_cast<std::size_t>(i)] == host_population[static_cast<std::size_t>(j)]) {
                distinct = false;
                break;
            }
        }
    }
    if (!distinct) {
        std::fprintf(stderr, "AllocateSlots did not produce distinct live slots\n");
        cudaFree(results);
        cudaFree(population);
        genotype.Destroy();
        return EXIT_FAILURE;
    }
    for (const slab::Slot slot : host_population) {
        Check(cudaMemcpy(genotype.view().genotypes + static_cast<std::size_t>(slot) * slab::kSlotFloats, weights.data(),
                         weights.size() * sizeof(float), cudaMemcpyHostToDevice),
              "copy training weights");
    }
    fitness::Evaluator evaluator;
    Check(evaluator.Create(vocabulary, 256), "Evaluator::Create");
    cudaEvent_t begin = nullptr, end = nullptr;
    Check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
    Check(cudaEventCreate(&end), "cudaEventCreate(end)");
    const auto cleanup = [&]() {
        Check(cudaEventDestroy(end), "cudaEventDestroy(end)");
        Check(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
        Check(evaluator.Destroy(), "Evaluator::Destroy");
        Check(cudaFree(results), "cudaFree(results)");
        Check(cudaFree(population), "cudaFree(population)");
        Check(genotype.Destroy(), "Slab::Destroy");
    };
    std::vector<fitness::Result> host_results(static_cast<std::size_t>(population_count));
    const auto evaluate = [&](bool timed, float *milliseconds) {
        if (timed)
            Check(cudaEventRecord(begin), "cudaEventRecord(begin)");
        Check(evaluator.Evaluate(genotype.view(), population, population_count, results), "Evaluator::Evaluate");
        if (timed) {
            Check(cudaEventRecord(end), "cudaEventRecord(end)");
            Check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
            Check(cudaEventElapsedTime(milliseconds, begin, end), "cudaEventElapsedTime");
        } else {
            Check(cudaDeviceSynchronize(), "warmup synchronize");
        }
        Check(cudaMemcpy(host_results.data(), results, host_results.size() * sizeof(fitness::Result),
                         cudaMemcpyDeviceToHost),
              "copy results");
        for (int i = 0; i < population_count; ++i) {
            if (!BaselineMatches(host_results[static_cast<std::size_t>(i)])) {
                std::fprintf(stderr, "%s baseline mismatch at organism %d\n", timed ? "evaluation" : "warmup", i);
                return false;
            }
        }
        return true;
    };
    float milliseconds = 0.0f;
    if (benchmark && !evaluate(false, &milliseconds)) {
        cleanup();
        return EXIT_FAILURE;
    }
    std::vector<float> times;
    times.reserve(static_cast<std::size_t>(benchmark ? repetitions : 1));
    for (int repetition = 0; repetition < (benchmark ? repetitions : 1); ++repetition) {
        if (!evaluate(true, &milliseconds)) {
            cleanup();
            return EXIT_FAILURE;
        }
        times.push_back(milliseconds);
        if (!benchmark) {
            const fitness::Result &host = host_results[0];
            std::printf(
                "full-training: games=%u wins=%u guess_sum=%u invalid=%u score=%u time_ms=%.3f allocated_bytes=%zu\n",
                host.games, host.wins, host.guess_sum, host.invalid_games, host.score, milliseconds,
                evaluator.allocated_bytes());
            std::printf("solved_in: %u %u %u %u %u %u\n", host.solved_in[0], host.solved_in[1], host.solved_in[2],
                        host.solved_in[3], host.solved_in[4], host.solved_in[5]);
        }
    }
    if (benchmark) {
        std::sort(times.begin(), times.end());
        const float median = repetitions % 2 == 0 ? (times[times.size() / 2 - 1] + times[times.size() / 2]) * 0.5f
                                                  : times[times.size() / 2];
        std::printf("fitness-benchmark: population=%d repetitions=%d min_ms=%.3f median_ms=%.3f max_ms=%.3f "
                    "evaluator_bytes=%zu\n",
                    population_count, repetitions, times.front(), median, times.back(), evaluator.allocated_bytes());
    }
    cleanup();
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    SelectTestGpu();
    if ((argc != 1 && argc != 2 && argc != 4) || (argc == 2 && std::string(argv[1]) != "--full-training") ||
        (argc == 4 && std::string(argv[1]) != "--benchmark")) {
        std::fprintf(stderr, "Usage: fitness_test [--full-training | --benchmark POPULATION REPETITIONS]\n");
        return EXIT_FAILURE;
    }
    int population = 0;
    int repetitions = 0;
    if (argc == 4 && (!ParsePositiveInt(argv[2], fitness::kMaxPopulation, &population) ||
                      !ParsePositiveInt(argv[3], 20, &repetitions))) {
        std::fprintf(stderr, "Usage: fitness_test [--full-training | --benchmark POPULATION REPETITIONS]\n");
        return EXIT_FAILURE;
    }
    const auto vocabulary = fitness::LoadVocabulary(WORD_DATA_DIR);
    if (argc > 1 && std::string(argv[1]) == "--full-training")
        return FullTraining(vocabulary, 1, 1, false);
    if (argc == 4)
        return FullTraining(vocabulary, population, repetitions, true);
    BasicTests(vocabulary);
    std::puts("CUDA fitness primitives passed");
    return EXIT_SUCCESS;
}
