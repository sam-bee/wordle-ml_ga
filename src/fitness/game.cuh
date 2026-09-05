#pragma once

#include "fitness/evaluator.hpp"

#include <cuda_runtime.h>

namespace wordle_ga::fitness {

// Green matches consume letters first; yellow matches consume remaining copies
// left-to-right. Position zero is the least-significant base-3 digit (0/1/2).
__device__ inline std::uint8_t Feedback(Word guess, Word answer) {
    std::uint8_t marks[5]{};
    bool used[5]{};
    for (int i = 0; i < 5; ++i) {
        if (guess.letters[i] == answer.letters[i]) {
            marks[i] = 2;
            used[i] = true;
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (marks[i] != 0)
            continue;
        for (int j = 0; j < 5; ++j) {
            if (!used[j] && guess.letters[i] == answer.letters[j]) {
                marks[i] = 1;
                used[j] = true;
                break;
            }
        }
    }
    int code = 0;
    int factor = 1;
    for (int i = 0; i < 5; ++i) {
        code += factor * marks[i];
        factor *= 3;
    }
    return static_cast<std::uint8_t>(code);
}

struct Game {
    std::uint8_t candidates[model::kNumSolutions];
    std::uint16_t history[6];
    std::uint16_t target;
    int organism;
    int guesses;
    int won;
    int invalid;
};

struct Tables {
    const Word *actions;
    const Word *solutions;
    const std::uint16_t *solution_actions;
    const std::uint16_t *training_solutions;
    const std::uint8_t *feedback; // [action * kNumSolutions + solution]
};

cudaError_t BuildFeedback(const Word *actions, const Word *solutions, std::uint8_t *feedback, cudaStream_t stream);
cudaError_t StartGames(Tables tables, genotype_slab::DeviceView slab, const genotype_slab::Slot *population_slots,
                       std::uint64_t first_pair, int count, Game *games, int *active, const float **parameters,
                       cudaStream_t stream);
cudaError_t EncodeGames(Tables tables, const Game *games, const int *active, int count, model::Input *inputs,
                        cudaStream_t stream);
cudaError_t AdvanceGames(Tables tables, Game *games, int *active, int count, const float *logits, cudaStream_t stream);
cudaError_t AccumulateGames(const Game *games, int count, Result *results, cudaStream_t stream);
cudaError_t FinishResults(Result *results, int population_count, cudaStream_t stream);

} // namespace wordle_ga::fitness
