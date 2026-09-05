#pragma once

#include "model/policy.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace wordle_ga::fitness {

inline constexpr int kTrainingSolutions = 2109;

struct Word {
    std::uint8_t letters[5]; // A=0 through Z=25.
};

struct Vocabulary {
    std::array<Word, model::kNumActions> actions;
    std::array<Word, model::kNumSolutions> solutions;
    std::array<std::uint16_t, model::kNumSolutions> solution_actions;
    std::array<std::uint16_t, kTrainingSolutions> training_solutions;
};

// Loads the frozen action/all-solution/training CSVs only. Throws std::runtime_error
// for invalid data. File order is the canonical ID order; held-out records are unused.
Vocabulary LoadVocabulary(const std::string &data_directory);

} // namespace wordle_ga::fitness
